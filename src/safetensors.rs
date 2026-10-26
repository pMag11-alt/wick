//! The safetensors format: an 8-byte little-endian header length, a JSON
//! header describing every tensor, then the raw tensor data. Sharded models
//! (`model.safetensors.index.json`) are supported too.

use crate::json::Json;
use crate::mmap::{Bytes, Mmap};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
        }
    }

    fn parse(s: &str) -> Option<DType> {
        match s {
            "F32" => Some(DType::F32),
            "F16" => Some(DType::F16),
            "BF16" => Some(DType::BF16),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub struct TensorInfo {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data: Bytes,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

pub struct SafeTensors {
    tensors: HashMap<String, TensorInfo>,
}

impl SafeTensors {
    /// Open `model.safetensors`, or every shard listed in
    /// `model.safetensors.index.json`, inside a model directory.
    pub fn open_dir(dir: &Path) -> Result<SafeTensors, String> {
        let index = dir.join("model.safetensors.index.json");
        let files: Vec<PathBuf> = if index.exists() {
            let text = std::fs::read_to_string(&index).map_err(|e| format!("{}: {e}", index.display()))?;
            let j = Json::parse(&text).map_err(|e| format!("{}: {e}", index.display()))?;
            let map = j.get("weight_map").and_then(Json::as_object).ok_or("index has no weight_map")?;
            let mut shards: Vec<&str> = map.iter().filter_map(|(_, v)| v.as_str()).collect();
            shards.sort();
            shards.dedup();
            shards.iter().map(|s| dir.join(s)).collect()
        } else {
            vec![dir.join("model.safetensors")]
        };
        let mut tensors = HashMap::new();
        for f in files {
            tensors.extend(Self::open_file(&f)?.tensors);
        }
        Ok(SafeTensors { tensors })
    }

    pub fn open_file(path: &Path) -> Result<SafeTensors, String> {
        let map = Arc::new(Mmap::open(path).map_err(|e| format!("{}: {e}", path.display()))?);
        Self::parse(map).map_err(|e| format!("{}: {e}", path.display()))
    }

    fn parse(map: Arc<Mmap>) -> Result<SafeTensors, String> {
        if map.len() < 8 {
            return Err("file too short".into());
        }
        let hlen = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
        if hlen > map.len() - 8 || hlen > 100 << 20 {
            return Err("bad header length".into());
        }
        let header = std::str::from_utf8(&map[8..8 + hlen]).map_err(|_| "header is not UTF-8")?;
        let j = Json::parse(header).map_err(|e| e.to_string())?;
        let base = 8 + hlen;
        let mut tensors = HashMap::new();
        for (name, info) in j.as_object().ok_or("header is not an object")? {
            if name == "__metadata__" {
                continue;
            }
            let dtype_s = info.get("dtype").and_then(Json::as_str).ok_or("missing dtype")?;
            let dtype = DType::parse(dtype_s).ok_or_else(|| format!("{name}: unsupported dtype {dtype_s}"))?;
            let shape: Vec<usize> = info
                .get("shape")
                .and_then(Json::as_array)
                .ok_or("missing shape")?
                .iter()
                .map(|d| d.as_usize().ok_or("bad shape"))
                .collect::<Result<_, _>>()?;
            let offs = info.get("data_offsets").and_then(Json::as_array).ok_or("missing data_offsets")?;
            let (begin, end) = match offs {
                [b, e] => (b.as_usize().ok_or("bad offset")?, e.as_usize().ok_or("bad offset")?),
                _ => return Err("bad data_offsets".into()),
            };
            let numel: usize = shape.iter().product();
            if end < begin || base + end > map.len() || end - begin != numel * dtype.size() {
                return Err(format!("{name}: data range does not match its shape"));
            }
            let data = Bytes::slice_of(&map, base + begin, end - begin);
            tensors.insert(name.clone(), TensorInfo { dtype, shape, data });
        }
        Ok(SafeTensors { tensors })
    }

    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Write tensors in safetensors format (used by tests and `wick quantize`).
pub fn write(path: &Path, tensors: &[(&str, DType, &[usize], &[u8])]) -> std::io::Result<()> {
    let mut entries = Vec::new();
    let mut offset = 0usize;
    for (name, dtype, shape, data) in tensors {
        let dt = match dtype {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
        };
        entries.push((
            name.to_string(),
            Json::object([
                ("dtype", Json::str(dt)),
                ("shape", Json::Array(shape.iter().map(|&d| Json::num(d as f64)).collect())),
                ("data_offsets", Json::Array(vec![Json::num(offset as f64), Json::num((offset + data.len()) as f64)])),
            ]),
        ));
        offset += data.len();
    }
    let mut header = Json::Object(entries).to_string();
    while !(8 + header.len()).is_multiple_of(8) {
        header.push(' ');
    }
    let mut out = Vec::with_capacity(8 + header.len() + offset);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for (_, _, _, data) in tensors {
        out.extend_from_slice(data);
    }
    std::fs::write(path, out)
}
