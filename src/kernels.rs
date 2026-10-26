//! Dot-product kernels: one weight row against up to four activation rows.
//!
//! Computing four dot products per pass over a weight row lets prompt
//! processing reuse each loaded (and converted) weight four times. There are
//! NEON versions (with the SDOT instruction for the quantized formats), AVX2
//! versions for f32/bf16 on x86-64, and portable scalar versions that the
//! tests use as the reference.

use crate::quant::{ActQ8, BlockQ4, BlockQ8, QK, bf16_to_f32, f16_to_f32};

pub mod scalar {
    use super::*;

    pub fn dot_f32(w: &[f32], x: &[f32]) -> f32 {
        // Four partial sums, like the SIMD versions, for similar rounding.
        let mut acc = [0f32; 8];
        let n = w.len() / 8 * 8;
        for i in (0..n).step_by(8) {
            for k in 0..8 {
                acc[k] += w[i + k] * x[i + k];
            }
        }
        let mut s: f32 = acc.iter().sum();
        for i in n..w.len() {
            s += w[i] * x[i];
        }
        s
    }

    pub fn dot_bf16(w: &[u16], x: &[f32]) -> f32 {
        let mut acc = [0f32; 8];
        let n = w.len() / 8 * 8;
        for i in (0..n).step_by(8) {
            for k in 0..8 {
                acc[k] += bf16_to_f32(w[i + k]) * x[i + k];
            }
        }
        let mut s: f32 = acc.iter().sum();
        for i in n..w.len() {
            s += bf16_to_f32(w[i]) * x[i];
        }
        s
    }

    pub fn dot_q8(w: &[BlockQ8], x: &[ActQ8]) -> f32 {
        w.iter()
            .zip(x)
            .map(|(a, b)| {
                let s: i32 = a.q.iter().zip(&b.q).map(|(&p, &q)| p as i32 * q as i32).sum();
                s as f32 * f16_to_f32(a.d) * b.d
            })
            .sum()
    }

    pub fn dot_q4(w: &[BlockQ4], x: &[ActQ8]) -> f32 {
        w.iter()
            .zip(x)
            .map(|(a, b)| {
                let mut s = 0i32;
                for j in 0..QK / 2 {
                    s += ((a.q[j] & 0xF) as i32 - 8) * b.q[j] as i32;
                    s += ((a.q[j] >> 4) as i32 - 8) * b.q[j + QK / 2] as i32;
                }
                s as f32 * f16_to_f32(a.d) * b.d
            })
            .sum()
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use crate::quant::f16_to_f32_fast;
    use std::arch::aarch64::*;

    #[inline(always)]
    unsafe fn bf16x8(p: *const u16) -> (float32x4_t, float32x4_t) {
        unsafe {
            let v = vld1q_u16(p);
            (
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(v))),
                vreinterpretq_f32_u32(vshll_high_n_u16::<16>(v)),
            )
        }
    }

    /// Up to four dot products of one f32 weight row; `n` must be a multiple of 8.
    pub unsafe fn dot_f32_n<const R: usize>(w: &[f32], xs: [&[f32]; R]) -> [f32; R] {
        unsafe {
            let n = w.len();
            let mut acc0 = [vdupq_n_f32(0.0); R];
            let mut acc1 = [vdupq_n_f32(0.0); R];
            let mut i = 0;
            while i + 8 <= n {
                let w0 = vld1q_f32(w.as_ptr().add(i));
                let w1 = vld1q_f32(w.as_ptr().add(i + 4));
                for r in 0..R {
                    acc0[r] = vfmaq_f32(acc0[r], w0, vld1q_f32(xs[r].as_ptr().add(i)));
                    acc1[r] = vfmaq_f32(acc1[r], w1, vld1q_f32(xs[r].as_ptr().add(i + 4)));
                }
                i += 8;
            }
            let mut out = [0f32; R];
            for r in 0..R {
                out[r] = vaddvq_f32(vaddq_f32(acc0[r], acc1[r]));
                for j in i..n {
                    out[r] += w[j] * xs[r][j];
                }
            }
            out
        }
    }

    pub unsafe fn dot_bf16_n<const R: usize>(w: &[u16], xs: [&[f32]; R]) -> [f32; R] {
        unsafe {
            let n = w.len();
            let mut acc0 = [vdupq_n_f32(0.0); R];
            let mut acc1 = [vdupq_n_f32(0.0); R];
            let mut i = 0;
            while i + 8 <= n {
                let (w0, w1) = bf16x8(w.as_ptr().add(i));
                for r in 0..R {
                    acc0[r] = vfmaq_f32(acc0[r], w0, vld1q_f32(xs[r].as_ptr().add(i)));
                    acc1[r] = vfmaq_f32(acc1[r], w1, vld1q_f32(xs[r].as_ptr().add(i + 4)));
                }
                i += 8;
            }
            let mut out = [0f32; R];
            for r in 0..R {
                out[r] = vaddvq_f32(vaddq_f32(acc0[r], acc1[r]));
                for j in i..n {
                    out[r] += bf16_to_f32(w[j]) * xs[r][j];
                }
            }
            out
        }
    }

    #[target_feature(enable = "dotprod")]
    pub unsafe fn dot_q8_n<const R: usize>(w: &[BlockQ8], xs: [&[ActQ8]; R]) -> [f32; R] {
        unsafe {
            let mut acc = [vdupq_n_f32(0.0); R];
            for (b, blk) in w.iter().enumerate() {
                let w0 = vld1q_s8(blk.q.as_ptr());
                let w1 = vld1q_s8(blk.q.as_ptr().add(16));
                let dw = f16_to_f32_fast(blk.d);
                for r in 0..R {
                    let x = &xs[r][b];
                    let s = vdotq_s32(
                        vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(x.q.as_ptr())),
                        w1,
                        vld1q_s8(x.q.as_ptr().add(16)),
                    );
                    acc[r] = vfmaq_n_f32(acc[r], vcvtq_f32_s32(s), dw * x.d);
                }
            }
            acc.map(|a| vaddvq_f32(a))
        }
    }

    #[target_feature(enable = "dotprod")]
    pub unsafe fn dot_q4_n<const R: usize>(w: &[BlockQ4], xs: [&[ActQ8]; R]) -> [f32; R] {
        unsafe {
            let mut acc = [vdupq_n_f32(0.0); R];
            let mask = vdupq_n_u8(0x0F);
            let eight = vdupq_n_s8(8);
            for (b, blk) in w.iter().enumerate() {
                let v = vld1q_u8(blk.q.as_ptr());
                let lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, mask)), eight);
                let hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v)), eight);
                let dw = f16_to_f32_fast(blk.d);
                for r in 0..R {
                    let x = &xs[r][b];
                    let s = vdotq_s32(
                        vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(x.q.as_ptr())),
                        hi,
                        vld1q_s8(x.q.as_ptr().add(16)),
                    );
                    acc[r] = vfmaq_n_f32(acc[r], vcvtq_f32_s32(s), dw * x.d);
                }
            }
            acc.map(|a| vaddvq_f32(a))
        }
    }

    pub fn has_dotprod() -> bool {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn hsum(v: __m256) -> f32 {
        unsafe {
            let lo = _mm256_castps256_ps128(v);
            let hi = _mm256_extractf128_ps::<1>(v);
            let s = _mm_add_ps(lo, hi);
            let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            let s = _mm_add_ss(s, _mm_shuffle_ps::<1>(s, s));
            _mm_cvtss_f32(s)
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_f32_n<const R: usize>(w: &[f32], xs: [&[f32]; R]) -> [f32; R] {
        unsafe {
            let n = w.len();
            let mut acc = [_mm256_setzero_ps(); R];
            let mut i = 0;
            while i + 8 <= n {
                let wv = _mm256_loadu_ps(w.as_ptr().add(i));
                for r in 0..R {
                    acc[r] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xs[r].as_ptr().add(i)), acc[r]);
                }
                i += 8;
            }
            let mut out = [0f32; R];
            for r in 0..R {
                out[r] = hsum(acc[r]);
                for j in i..n {
                    out[r] += w[j] * xs[r][j];
                }
            }
            out
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_bf16_n<const R: usize>(w: &[u16], xs: [&[f32]; R]) -> [f32; R] {
        unsafe {
            let n = w.len();
            let mut acc = [_mm256_setzero_ps(); R];
            let mut i = 0;
            while i + 8 <= n {
                let h = _mm_loadu_si128(w.as_ptr().add(i) as *const __m128i);
                let wv = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(h)));
                for r in 0..R {
                    acc[r] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xs[r].as_ptr().add(i)), acc[r]);
                }
                i += 8;
            }
            let mut out = [0f32; R];
            for r in 0..R {
                out[r] = hsum(acc[r]);
                for j in i..n {
                    out[r] += bf16_to_f32(w[j]) * xs[r][j];
                }
            }
            out
        }
    }

    pub fn available() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
}

fn simd_f32() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *OK.get_or_init(avx2::available)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        cfg!(target_arch = "aarch64")
    }
}

#[cfg(target_arch = "aarch64")]
fn simd_int8() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(neon::has_dotprod)
}

/// Set WICK_SCALAR=1 to force the portable kernels (for testing them).
fn forced_scalar() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| std::env::var_os("WICK_SCALAR").is_some_and(|v| v != "0"))
}

pub fn dot_f32_n<const R: usize>(w: &[f32], xs: [&[f32]; R]) -> [f32; R] {
    if simd_f32() && !forced_scalar() {
        #[cfg(target_arch = "aarch64")]
        return unsafe { neon::dot_f32_n(w, xs) };
        #[cfg(target_arch = "x86_64")]
        return unsafe { avx2::dot_f32_n(w, xs) };
    }
    xs.map(|x| scalar::dot_f32(w, x))
}

pub fn dot_bf16_n<const R: usize>(w: &[u16], xs: [&[f32]; R]) -> [f32; R] {
    if simd_f32() && !forced_scalar() {
        #[cfg(target_arch = "aarch64")]
        return unsafe { neon::dot_bf16_n(w, xs) };
        #[cfg(target_arch = "x86_64")]
        return unsafe { avx2::dot_bf16_n(w, xs) };
    }
    xs.map(|x| scalar::dot_bf16(w, x))
}

pub fn dot_q8_n<const R: usize>(w: &[BlockQ8], xs: [&[ActQ8]; R]) -> [f32; R] {
    #[cfg(target_arch = "aarch64")]
    if simd_int8() && !forced_scalar() {
        return unsafe { neon::dot_q8_n(w, xs) };
    }
    xs.map(|x| scalar::dot_q8(w, x))
}

pub fn dot_q4_n<const R: usize>(w: &[BlockQ4], xs: [&[ActQ8]; R]) -> [f32; R] {
    #[cfg(target_arch = "aarch64")]
    if simd_int8() && !forced_scalar() {
        return unsafe { neon::dot_q4_n(w, xs) };
    }
    xs.map(|x| scalar::dot_q4(w, x))
}

/// Which kernels this machine uses, for `wick bench` and logs.
pub fn describe() -> &'static str {
    if forced_scalar() {
        return "scalar (forced)";
    }
    #[cfg(target_arch = "aarch64")]
    {
        if simd_int8() { "NEON + dotprod" } else { "NEON" }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if simd_f32() { "AVX2 + FMA (scalar int8)" } else { "scalar" }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        "scalar"
    }
}

// ---- small vector ops ----

pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    for ((o, &v), &g) in out.iter_mut().zip(x).zip(w) {
        *o = v * scale * g;
    }
}

pub fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    for (v, &g) in x.iter_mut().zip(w) {
        *v = *v * scale * g;
    }
}

/// Softmax in place over `x`.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// SwiGLU: gate = silu(gate) * up.
pub fn silu_mul(gate: &mut [f32], up: &[f32]) {
    for (g, &u) in gate.iter_mut().zip(up) {
        *g = *g / (1.0 + (-*g).exp()) * u;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{quantize_act, quantize_q4, quantize_q8};

    fn data(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s % 2000) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn simd_matches_scalar() {
        for n in [8, 64, 96, 1000, 4096 + 8] {
            let w = data(n, 1);
            let xs: Vec<Vec<f32>> = (0..4).map(|k| data(n, 10 + k)).collect();
            let got = dot_f32_n(&w, [&xs[0], &xs[1], &xs[2], &xs[3]]);
            for r in 0..4 {
                let want = scalar::dot_f32(&w, &xs[r]);
                assert!((got[r] - want).abs() < 1e-3 * (1.0 + want.abs()), "f32 n={n}: {} vs {want}", got[r]);
            }
            let wb: Vec<u16> = w.iter().map(|&v| crate::quant::f32_to_bf16(v)).collect();
            let got = dot_bf16_n(&wb, [&xs[0], &xs[1], &xs[2], &xs[3]]);
            for r in 0..4 {
                let want = scalar::dot_bf16(&wb, &xs[r]);
                assert!((got[r] - want).abs() < 1e-3 * (1.0 + want.abs()), "bf16 n={n}");
            }
        }
        let n = 1024;
        let w = data(n, 3);
        let x = data(n, 4);
        let mut wq8 = vec![BlockQ8::default(); n / QK];
        let mut wq4 = vec![BlockQ4::default(); n / QK];
        let mut xq = vec![ActQ8::default(); n / QK];
        quantize_q8(&w, &mut wq8);
        quantize_q4(&w, &mut wq4);
        quantize_act(&x, &mut xq);
        let exact: f32 = w.iter().zip(&x).map(|(a, b)| a * b).sum();
        let [q8] = dot_q8_n(&wq8, [&xq]);
        let [q4] = dot_q4_n(&wq4, [&xq]);
        assert!((q8 - scalar::dot_q8(&wq8, &xq)).abs() < 1e-3);
        assert!((q4 - scalar::dot_q4(&wq4, &xq)).abs() < 1e-3);
        // Quantization error grows with the vectors' norms, not the result.
        let norms = w.iter().map(|v| v * v).sum::<f32>().sqrt() * x.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((q8 - exact).abs() < 0.003 * norms, "q8 {q8} vs {exact}");
        assert!((q4 - exact).abs() < 0.03 * norms, "q4 {q4} vs {exact}");
    }
}
