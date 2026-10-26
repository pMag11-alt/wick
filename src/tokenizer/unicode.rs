//! Unicode character classes and NFC normalization, from generated tables.

use super::unicode_tables::{COMBINING_CLASS, COMPOSITION, DECOMPOSITION, LETTER, NUMBER};
use std::borrow::Cow;

fn in_ranges(table: &[(u32, u32)], cp: u32) -> bool {
    table
        .binary_search_by(|&(lo, hi)| {
            if hi < cp {
                std::cmp::Ordering::Less
            } else if lo > cp {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// `\p{L}`: general category Letter.
pub fn is_letter(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphabetic();
    }
    in_ranges(LETTER, c as u32)
}

/// `\p{N}`: general category Number.
pub fn is_number(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    in_ranges(NUMBER, c as u32)
}

/// `\s`: the White_Space property.
pub fn is_space(c: char) -> bool {
    c.is_whitespace()
}

fn combining_class(cp: u32) -> u8 {
    if cp < 0x300 {
        return 0;
    }
    match COMBINING_CLASS.binary_search_by(|&(lo, hi, _)| {
        if hi < cp {
            std::cmp::Ordering::Less
        } else if lo > cp {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }) {
        Ok(i) => COMBINING_CLASS[i].2,
        Err(_) => 0,
    }
}

const S_BASE: u32 = 0xAC00;
const L_BASE: u32 = 0x1100;
const V_BASE: u32 = 0x1161;
const T_BASE: u32 = 0x11A7;
const L_COUNT: u32 = 19;
const V_COUNT: u32 = 21;
const T_COUNT: u32 = 28;
const N_COUNT: u32 = V_COUNT * T_COUNT;
const S_COUNT: u32 = L_COUNT * N_COUNT;

fn decompose(cp: u32, out: &mut Vec<u32>) {
    if (S_BASE..S_BASE + S_COUNT).contains(&cp) {
        // Hangul syllables decompose algorithmically.
        let s = cp - S_BASE;
        out.push(L_BASE + s / N_COUNT);
        out.push(V_BASE + (s % N_COUNT) / T_COUNT);
        if !s.is_multiple_of(T_COUNT) {
            out.push(T_BASE + s % T_COUNT);
        }
        return;
    }
    match DECOMPOSITION.binary_search_by_key(&cp, |&(c, _, _)| c) {
        Ok(i) => {
            let (_, a, b) = DECOMPOSITION[i];
            decompose(a, out);
            if b != 0 {
                decompose(b, out);
            }
        }
        Err(_) => out.push(cp),
    }
}

fn compose_pair(a: u32, b: u32) -> Option<u32> {
    if (L_BASE..L_BASE + L_COUNT).contains(&a) && (V_BASE..V_BASE + V_COUNT).contains(&b) {
        return Some(S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT);
    }
    if (S_BASE..S_BASE + S_COUNT).contains(&a)
        && (a - S_BASE).is_multiple_of(T_COUNT)
        && (T_BASE + 1..T_BASE + T_COUNT).contains(&b)
    {
        return Some(a + (b - T_BASE));
    }
    COMPOSITION.binary_search_by_key(&(a, b), |&(x, y, _)| (x, y)).ok().map(|i| COMPOSITION[i].2)
}

/// Normalization Form C (canonical decomposition, then canonical
/// composition), as Qwen's tokenizers apply to their input.
pub fn nfc(s: &str) -> Cow<'_, str> {
    // Everything below U+0300 is already in NFC and stays that way.
    if s.chars().all(|c| (c as u32) < 0x300) {
        return Cow::Borrowed(s);
    }
    let mut buf = Vec::with_capacity(s.len());
    for c in s.chars() {
        decompose(c as u32, &mut buf);
    }
    // Canonical ordering: stable-sort each run of non-starters by class.
    let mut i = 0;
    while i < buf.len() {
        if combining_class(buf[i]) == 0 {
            i += 1;
            continue;
        }
        let start = i;
        while i < buf.len() && combining_class(buf[i]) != 0 {
            i += 1;
        }
        buf[start..i].sort_by_key(|&cp| combining_class(cp));
    }
    // Canonical composition.
    let mut out: Vec<u32> = Vec::with_capacity(buf.len());
    let mut starter: Option<usize> = None;
    let mut prev_class: i32 = -1; // class of the last char after the starter; -1 if adjacent
    for &cp in &buf {
        let class = combining_class(cp) as i32;
        if let Some(si) = starter
            && (prev_class < class || prev_class == -1)
            && let Some(c) = compose_pair(out[si], cp)
        {
            out[si] = c;
            continue;
        }
        if class == 0 {
            starter = Some(out.len());
            prev_class = -1;
        } else {
            prev_class = class;
        }
        out.push(cp);
    }
    let result: String = out.into_iter().filter_map(char::from_u32).collect();
    if result == s { Cow::Borrowed(s) } else { Cow::Owned(result) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes() {
        assert!(is_letter('a') && is_letter('é') && is_letter('ж') && is_letter('中') && is_letter('ㅎ'));
        assert!(!is_letter('1') && !is_letter(' ') && !is_letter('!') && !is_letter('😀'));
        assert!(is_number('7') && is_number('٣') && is_number('Ⅻ') && is_number('½'));
        assert!(!is_number('x'));
    }

    #[test]
    fn nfc_composes_and_orders() {
        assert_eq!(nfc("e\u{301}"), "é");
        assert_eq!(nfc("\u{212B}"), "\u{C5}"); // ANGSTROM SIGN is a singleton
        assert_eq!(nfc("\u{1100}\u{1161}\u{11A8}"), "각");
        // Combining marks are reordered by class before composing.
        assert_eq!(nfc("a\u{323}\u{302}"), nfc("a\u{302}\u{323}"));
        assert_eq!(nfc("a\u{302}\u{323}"), "\u{1EAD}");
        assert!(matches!(nfc("plain ascii"), Cow::Borrowed(_)));
    }
}
