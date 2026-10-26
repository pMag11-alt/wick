//! A small, strict JSON parser and serializer (RFC 8259).
//!
//! Model configs, tokenizer files, safetensors headers and the HTTP API all
//! speak JSON, and this module is all wick needs for it. Objects keep their
//! keys in insertion order.

use std::fmt::{self, Write as _};

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub offset: usize,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "JSON error at byte {}: {}", self.offset, self.message)
    }
}

impl std::error::Error for ParseError {}

impl Json {
    pub fn parse(text: &str) -> Result<Json, ParseError> {
        let mut p = Parser { s: text.as_bytes(), i: 0, depth: 0 };
        p.skip_ws();
        let v = p.value()?;
        p.skip_ws();
        if p.i != p.s.len() {
            return Err(p.err("trailing characters after the JSON value"));
        }
        Ok(v)
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Follow a path of object keys.
    pub fn path(&self, keys: &[&str]) -> Option<&Json> {
        keys.iter().try_fold(self, |v, k| v.get(k))
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Json::Number(n) if n.fract() == 0.0 && n.abs() < 9.007e15 => Some(*n as i64),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        self.as_i64().and_then(|n| usize::try_from(n).ok())
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Object(o) => Some(o),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }

    /// Build an object from (key, value) pairs.
    pub fn object<K: Into<String>>(pairs: impl IntoIterator<Item = (K, Json)>) -> Json {
        Json::Object(pairs.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    pub fn str(s: impl Into<String>) -> Json {
        Json::String(s.into())
    }

    pub fn num(n: impl Into<f64>) -> Json {
        Json::Number(n.into())
    }
}

impl fmt::Display for Json {
    /// Compact serialization.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Json::Null => f.write_str("null"),
            Json::Bool(b) => write!(f, "{b}"),
            Json::Number(n) => {
                if !n.is_finite() {
                    f.write_str("null")
                } else if n.fract() == 0.0 && n.abs() < 1e15 {
                    write!(f, "{}", *n as i64)
                } else {
                    write!(f, "{n}")
                }
            }
            Json::String(s) => write_escaped(f, s),
            Json::Array(a) => {
                f.write_char('[')?;
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        f.write_char(',')?;
                    }
                    write!(f, "{v}")?;
                }
                f.write_char(']')
            }
            Json::Object(o) => {
                f.write_char('{')?;
                for (i, (k, v)) in o.iter().enumerate() {
                    if i > 0 {
                        f.write_char(',')?;
                    }
                    write_escaped(f, k)?;
                    write!(f, ":{v}")?;
                }
                f.write_char('}')
            }
        }
    }
}

fn write_escaped(f: &mut impl fmt::Write, s: &str) -> fmt::Result {
    f.write_char('"')?;
    for c in s.chars() {
        match c {
            '"' => f.write_str("\\\"")?,
            '\\' => f.write_str("\\\\")?,
            '\n' => f.write_str("\\n")?,
            '\r' => f.write_str("\\r")?,
            '\t' => f.write_str("\\t")?,
            c if (c as u32) < 0x20 => write!(f, "\\u{:04x}", c as u32)?,
            c => f.write_char(c)?,
        }
    }
    f.write_char('"')
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
    depth: usize,
}

impl Parser<'_> {
    fn err(&self, message: &str) -> ParseError {
        ParseError { offset: self.i, message: message.to_string() }
    }

    fn skip_ws(&mut self) {
        while let Some(&c) = self.s.get(self.i) {
            if c == b' ' || c == b'\n' || c == b'\r' || c == b'\t' {
                self.i += 1;
            } else {
                break;
            }
        }
    }

    fn eat(&mut self, lit: &str) -> Result<(), ParseError> {
        if self.s[self.i..].starts_with(lit.as_bytes()) {
            self.i += lit.len();
            Ok(())
        } else {
            Err(self.err(&format!("expected `{lit}`")))
        }
    }

    fn value(&mut self) -> Result<Json, ParseError> {
        match self.s.get(self.i) {
            None => Err(self.err("unexpected end of input")),
            Some(b'n') => self.eat("null").map(|_| Json::Null),
            Some(b't') => self.eat("true").map(|_| Json::Bool(true)),
            Some(b'f') => self.eat("false").map(|_| Json::Bool(false)),
            Some(b'"') => self.string().map(Json::String),
            Some(b'[') => self.nested(Self::array),
            Some(b'{') => self.nested(Self::object),
            Some(c) if *c == b'-' || c.is_ascii_digit() => self.number(),
            Some(_) => Err(self.err("unexpected character")),
        }
    }

    fn nested(&mut self, f: fn(&mut Self) -> Result<Json, ParseError>) -> Result<Json, ParseError> {
        self.depth += 1;
        if self.depth > 512 {
            return Err(self.err("nesting too deep"));
        }
        let v = f(self);
        self.depth -= 1;
        v
    }

    fn array(&mut self) -> Result<Json, ParseError> {
        self.i += 1;
        let mut items = Vec::new();
        self.skip_ws();
        if self.s.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Json::Array(items));
        }
        loop {
            self.skip_ws();
            items.push(self.value()?);
            self.skip_ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Array(items));
                }
                _ => return Err(self.err("expected `,` or `]`")),
            }
        }
    }

    fn object(&mut self) -> Result<Json, ParseError> {
        self.i += 1;
        let mut pairs = Vec::new();
        self.skip_ws();
        if self.s.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(Json::Object(pairs));
        }
        loop {
            self.skip_ws();
            if self.s.get(self.i) != Some(&b'"') {
                return Err(self.err("expected a string key"));
            }
            let k = self.string()?;
            self.skip_ws();
            if self.s.get(self.i) != Some(&b':') {
                return Err(self.err("expected `:`"));
            }
            self.i += 1;
            self.skip_ws();
            let v = self.value()?;
            pairs.push((k, v));
            self.skip_ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Object(pairs));
                }
                _ => return Err(self.err("expected `,` or `}`")),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, ParseError> {
        let h = self.s.get(self.i..self.i + 4).ok_or_else(|| self.err("short \\u escape"))?;
        let h = std::str::from_utf8(h).map_err(|_| self.err("bad \\u escape"))?;
        let v = u32::from_str_radix(h, 16).map_err(|_| self.err("bad \\u escape"))?;
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String, ParseError> {
        self.i += 1;
        let mut out = String::new();
        loop {
            // Copy a run of plain bytes in one go.
            let start = self.i;
            while let Some(&c) = self.s.get(self.i) {
                if c == b'"' || c == b'\\' || c < 0x20 {
                    break;
                }
                self.i += 1;
            }
            if self.i > start {
                // The input is a &str, and we only stop on ASCII, so this is valid UTF-8.
                out.push_str(std::str::from_utf8(&self.s[start..self.i]).unwrap());
            }
            match self.s.get(self.i) {
                None => return Err(self.err("unterminated string")),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    let esc = *self.s.get(self.i).ok_or_else(|| self.err("unterminated escape"))?;
                    self.i += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                // A surrogate pair.
                                if self.s.get(self.i..self.i + 2) != Some(b"\\u") {
                                    return Err(self.err("unpaired surrogate"));
                                }
                                self.i += 2;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err(self.err("invalid low surrogate"));
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else if (0xDC00..0xE000).contains(&hi) {
                                return Err(self.err("unpaired surrogate"));
                            } else {
                                hi
                            };
                            out.push(char::from_u32(cp).ok_or_else(|| self.err("invalid code point"))?);
                        }
                        _ => return Err(self.err("invalid escape")),
                    }
                }
                Some(_) => return Err(self.err("control character in string")),
            }
        }
    }

    fn number(&mut self) -> Result<Json, ParseError> {
        let start = self.i;
        if self.s[self.i] == b'-' {
            self.i += 1;
        }
        match self.s.get(self.i) {
            Some(b'0') => self.i += 1,
            Some(c) if c.is_ascii_digit() => {
                while self.s.get(self.i).is_some_and(u8::is_ascii_digit) {
                    self.i += 1;
                }
            }
            _ => return Err(self.err("invalid number")),
        }
        if self.s.get(self.i) == Some(&b'.') {
            self.i += 1;
            if !self.s.get(self.i).is_some_and(u8::is_ascii_digit) {
                return Err(self.err("digit expected after `.`"));
            }
            while self.s.get(self.i).is_some_and(u8::is_ascii_digit) {
                self.i += 1;
            }
        }
        if matches!(self.s.get(self.i), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.s.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if !self.s.get(self.i).is_some_and(u8::is_ascii_digit) {
                return Err(self.err("digit expected in exponent"));
            }
            while self.s.get(self.i).is_some_and(u8::is_ascii_digit) {
                self.i += 1;
            }
        }
        let text = std::str::from_utf8(&self.s[start..self.i]).unwrap();
        text.parse::<f64>().map(Json::Number).map_err(|_| self.err("invalid number"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let text = r#"{"a":[1,2.5,-3e2,true,false,null],"b":{"c":"x\"y\\z\n\u00e9\ud83d\ude00"}}"#;
        let v = Json::parse(text).unwrap();
        assert_eq!(v.path(&["b", "c"]).unwrap().as_str().unwrap(), "x\"y\\z\né😀");
        assert_eq!(v.get("a").unwrap().as_array().unwrap()[2].as_f64(), Some(-300.0));
        assert_eq!(Json::parse(&v.to_string()).unwrap(), v);
    }

    #[test]
    fn rejects_malformed() {
        for bad in ["", "{", "[1,]", "{\"a\" 1}", "01", "1.", "\"\\x\"", "\"a", "nul", "[1] x", "\"\\ud800\""] {
            assert!(Json::parse(bad).is_err(), "{bad:?} should not parse");
        }
    }

    #[test]
    fn deep_nesting_is_bounded() {
        let text = "[".repeat(10_000);
        assert!(Json::parse(&text).is_err());
    }
}
