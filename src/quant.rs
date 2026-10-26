//! Number formats: bf16/f16 conversion and the block-quantized formats.
//!
//! Q8_0 and Q4_0 follow the layout llama.cpp made common: blocks of 32
//! weights sharing one f16 scale. Q8_0 stores `round(w / d)` as i8
//! (8.5 bits per weight); Q4_0 stores 4-bit values offset by 8 (4.5 bits per
//! weight), with element j in the low nibble of byte j and element j + 16
//! in the high nibble.

// chunks_exact(QK) reads more plainly here than as_chunks and builds on
// older compilers.
#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]

pub const QK: usize = 32;

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct BlockQ8 {
    pub d: u16, // f16 scale
    pub q: [i8; QK],
}

/// Activations quantized on the fly before an int8 matrix product. The
/// scale stays f32: activations are short-lived, so size doesn't matter.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct ActQ8 {
    pub d: f32,
    pub q: [i8; QK],
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct BlockQ4 {
    pub d: u16,
    pub q: [u8; QK / 2],
}

pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Round to nearest even, as torch does.
pub fn f32_to_bf16(f: f32) -> u16 {
    let bits = f.to_bits();
    if f.is_nan() {
        return ((bits >> 16) | 0x40) as u16;
    }
    let round = 0x7FFF + ((bits >> 16) & 1);
    ((bits + round) >> 16) as u16
}

/// f16 -> f32 through a 256 KiB table: the int8 kernels convert one weight
/// scale per 32 weights, and the branchy bit-twiddling version is slow.
pub fn f16_to_f32_fast(h: u16) -> f32 {
    static TABLE: std::sync::OnceLock<Box<[f32]>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| (0..=u16::MAX).map(f16_to_f32).collect())[h as usize]
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // Subnormal: normalize the mantissa.
            let shift = man.leading_zeros() - 21;
            let man = (man << shift) & 0x3FF;
            sign | ((127 - 15 + 1 - shift) << 23) | (man << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

pub fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = bits & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        // Subnormal result, rounded to nearest even.
        let man = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rounded = (man + half - 1 + ((man >> shift) & 1)) >> shift;
        return sign | rounded as u16;
    }
    let rounded = man + 0xFFF + ((man >> 13) & 1);
    if rounded & 0x80_0000 != 0 {
        // The mantissa overflowed into the exponent.
        let e = e + 1;
        if e >= 0x1F {
            return sign | 0x7C00;
        }
        return sign | ((e as u16) << 10);
    }
    sign | ((e as u16) << 10) | (rounded >> 13) as u16
}

pub fn quantize_q8(x: &[f32], out: &mut [BlockQ8]) {
    assert_eq!(x.len(), out.len() * QK);
    for (blk, xs) in out.iter_mut().zip(x.chunks_exact(QK)) {
        let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        blk.d = f32_to_f16(d);
        for (q, &v) in blk.q.iter_mut().zip(xs) {
            *q = (v * id).round() as i8;
        }
    }
}

pub fn quantize_act(x: &[f32], out: &mut [ActQ8]) {
    assert_eq!(x.len(), out.len() * QK);
    for (blk, xs) in out.iter_mut().zip(x.chunks_exact(QK)) {
        let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        blk.d = d;
        for (q, &v) in blk.q.iter_mut().zip(xs) {
            *q = (v * id).round() as i8;
        }
    }
}

pub fn quantize_q4(x: &[f32], out: &mut [BlockQ4]) {
    assert_eq!(x.len(), out.len() * QK);
    for (blk, xs) in out.iter_mut().zip(x.chunks_exact(QK)) {
        // Like llama.cpp: the largest-magnitude value maps to -8 exactly.
        let mut max = 0f32;
        for &v in xs {
            if v.abs() > max.abs() {
                max = v;
            }
        }
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        blk.d = f32_to_f16(d);
        for j in 0..QK / 2 {
            let lo = ((xs[j] * id + 8.5) as i32).clamp(0, 15) as u8;
            let hi = ((xs[j + QK / 2] * id + 8.5) as i32).clamp(0, 15) as u8;
            blk.q[j] = lo | (hi << 4);
        }
    }
}

pub fn dequantize_q8(blocks: &[BlockQ8], out: &mut [f32]) {
    for (blk, o) in blocks.iter().zip(out.chunks_exact_mut(QK)) {
        let d = f16_to_f32(blk.d);
        for (v, &q) in o.iter_mut().zip(&blk.q) {
            *v = q as f32 * d;
        }
    }
}

pub fn dequantize_q4(blocks: &[BlockQ4], out: &mut [f32]) {
    for (blk, o) in blocks.iter().zip(out.chunks_exact_mut(QK)) {
        let d = f16_to_f32(blk.d);
        for j in 0..QK / 2 {
            o[j] = ((blk.q[j] & 0xF) as i32 - 8) as f32 * d;
            o[j + QK / 2] = ((blk.q[j] >> 4) as i32 - 8) as f32 * d;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip_and_rounding() {
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            if f.is_nan() {
                assert!(f16_to_f32(f32_to_f16(f)).is_nan());
            } else {
                assert_eq!(f32_to_f16(f), h, "{h:#06x} -> {f}");
            }
        }
        assert_eq!(f32_to_f16(1.0 + 1.0 / 2048.0), 0x3C00); // tie -> even
        assert_eq!(f32_to_f16(65520.0), 0x7C00); // overflows to inf
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(f32_to_bf16(1.0), 0x3F80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F80_8000)), 0x3F80); // tie, even stays
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F81_8000)), 0x3F82); // tie, odd rounds up
    }

    #[test]
    fn quantization_error_is_small() {
        let x: Vec<f32> = (0..256).map(|i| ((i * 37 % 101) as f32 - 50.0) / 13.0).collect();
        let mut q8 = vec![BlockQ8::default(); 8];
        let mut q4 = vec![BlockQ4::default(); 8];
        quantize_q8(&x, &mut q8);
        quantize_q4(&x, &mut q4);
        let mut y = vec![0f32; 256];
        dequantize_q8(&q8, &mut y);
        let e8 = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        dequantize_q4(&q4, &mut y);
        let e4 = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        // At most half a step for q8; one step (amax / 8) for q4, whose
        // positive side only reaches +7.
        assert!(e8 < 0.02, "q8 error {e8}");
        assert!(e4 <= 50.0 / 13.0 / 8.0 + 1e-3, "q4 error {e4}");
    }
}
