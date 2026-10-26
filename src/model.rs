//! The decoder-only transformer shared by Llama, Qwen2 and Qwen3, and its
//! weights in one of four formats.

use crate::config::{Arch, Config};
use crate::kernels::{self, dot_bf16_n, dot_f32_n, dot_q4_n, dot_q8_n, rmsnorm, rmsnorm_inplace, silu_mul, softmax};
use crate::mmap::Bytes;
use crate::pool::{Pool, SyncPtr};
use crate::quant::{self, ActQ8, BlockQ4, BlockQ8, QK, bf16_to_f32, f16_to_f32, f32_to_bf16};
use crate::safetensors::{DType, SafeTensors, TensorInfo};
use std::path::Path;

/// How to hold the big matrices in memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    F32,
    BF16,
    Q8,
    Q4,
}

impl Format {
    pub fn parse(s: &str) -> Option<Format> {
        match s.to_ascii_lowercase().as_str() {
            "f32" => Some(Format::F32),
            "bf16" => Some(Format::BF16),
            "q8" | "q8_0" => Some(Format::Q8),
            "q4" | "q4_0" => Some(Format::Q4),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Format::F32 => "f32",
            Format::BF16 => "bf16",
            Format::Q8 => "q8_0",
            Format::Q4 => "q4_0",
        }
    }
}

pub enum Data {
    F32(Bytes),
    BF16(Bytes),
    Q8(Vec<BlockQ8>),
    Q4(Vec<BlockQ4>),
}

/// A row-major [rows, cols] matrix.
pub struct Weight {
    pub rows: usize,
    pub cols: usize,
    pub data: Data,
}

impl Weight {
    fn load(t: &TensorInfo, format: Format) -> Result<Weight, String> {
        let (rows, cols) = match t.shape[..] {
            [r, c] => (r, c),
            _ => return Err(format!("expected a matrix, got shape {:?}", t.shape)),
        };
        // Quantized formats need whole blocks per row.
        let format = if matches!(format, Format::Q8 | Format::Q4) && cols % QK != 0 { Format::BF16 } else { format };
        let data = match (format, t.dtype) {
            (Format::F32, DType::F32) | (Format::BF16, DType::F32) => Data::F32(t.data.clone().aligned_to::<f32>()),
            (Format::BF16, DType::BF16) => Data::BF16(t.data.clone().aligned_to::<u16>()),
            (Format::BF16, DType::F16) => {
                let src = t.data.clone().aligned_to::<u16>();
                let v: Vec<u16> = src.as_u16().iter().map(|&h| f32_to_bf16(f16_to_f32(h))).collect();
                Data::BF16(Bytes::copy_from(u16_bytes(&v)))
            }
            (Format::F32, _) => Data::F32(Bytes::copy_from(f32_bytes(&to_f32(t)))),
            (Format::Q8, _) | (Format::Q4, _) => {
                let f = to_f32(t);
                if format == Format::Q8 {
                    let mut b = vec![BlockQ8::default(); rows * cols / QK];
                    quant::quantize_q8(&f, &mut b);
                    Data::Q8(b)
                } else {
                    let mut b = vec![BlockQ4::default(); rows * cols / QK];
                    quant::quantize_q4(&f, &mut b);
                    Data::Q4(b)
                }
            }
        };
        Ok(Weight { rows, cols, data })
    }

    pub fn bytes(&self) -> usize {
        match &self.data {
            Data::F32(b) | Data::BF16(b) => b.len(),
            Data::Q8(v) => v.len() * std::mem::size_of::<BlockQ8>(),
            Data::Q4(v) => v.len() * std::mem::size_of::<BlockQ4>(),
        }
    }

    /// Copy row r out as f32 (for the embedding lookup).
    pub fn row(&self, r: usize, out: &mut [f32]) {
        let c = self.cols;
        match &self.data {
            Data::F32(b) => out.copy_from_slice(&b.as_f32()[r * c..(r + 1) * c]),
            Data::BF16(b) => {
                for (o, &v) in out.iter_mut().zip(&b.as_u16()[r * c..(r + 1) * c]) {
                    *o = bf16_to_f32(v);
                }
            }
            Data::Q8(v) => quant::dequantize_q8(&v[r * c / QK..(r + 1) * c / QK], out),
            Data::Q4(v) => quant::dequantize_q4(&v[r * c / QK..(r + 1) * c / QK], out),
        }
    }

    fn is_quantized(&self) -> bool {
        matches!(self.data, Data::Q8(_) | Data::Q4(_))
    }
}

fn to_f32(t: &TensorInfo) -> Vec<f32> {
    let d = t.data.clone();
    match t.dtype {
        DType::F32 => d.aligned_to::<f32>().as_f32().to_vec(),
        DType::BF16 => d.aligned_to::<u16>().as_u16().iter().map(|&v| bf16_to_f32(v)).collect(),
        DType::F16 => d.aligned_to::<u16>().as_u16().iter().map(|&v| f16_to_f32(v)).collect(),
    }
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding or invalid bit patterns.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn u16_bytes(v: &[u16]) -> &[u8] {
    // SAFETY: as above.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
}

fn vector(st: &SafeTensors, name: &str, len: usize) -> Result<Vec<f32>, String> {
    let t = st.get(name).ok_or_else(|| format!("missing tensor {name}"))?;
    if t.numel() != len {
        return Err(format!("{name}: expected {len} values, found {}", t.numel()));
    }
    Ok(to_f32(t))
}

pub struct Layer {
    attn_norm: Vec<f32>,
    wq: Weight,
    wk: Weight,
    wv: Weight,
    wo: Weight,
    bq: Option<Vec<f32>>,
    bk: Option<Vec<f32>>,
    bv: Option<Vec<f32>>,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    ffn_norm: Vec<f32>,
    w_gate: Weight,
    w_up: Weight,
    w_down: Weight,
}

pub struct Model {
    pub cfg: Config,
    pub format: Format,
    embed: Weight,
    lm_head: Option<Weight>, // None: tied to `embed`
    layers: Vec<Layer>,
    norm: Vec<f32>,
    inv_freq: Vec<f32>,
}

impl Model {
    pub fn load(dir: &Path, format: Format) -> Result<Model, String> {
        let cfg = Config::load(dir)?;
        let st = SafeTensors::open_dir(dir)?;
        Model::from_tensors(cfg, &st, format)
    }

    pub fn from_tensors(cfg: Config, st: &SafeTensors, format: Format) -> Result<Model, String> {
        let mat = |name: &str, rows: usize, cols: usize, fmt: Format| -> Result<Weight, String> {
            let t = st.get(name).ok_or_else(|| format!("missing tensor {name}"))?;
            if t.shape != [rows, cols] {
                return Err(format!("{name}: expected shape [{rows}, {cols}], found {:?}", t.shape));
            }
            Weight::load(t, fmt).map_err(|e| format!("{name}: {e}"))
        };
        let (h, q, kv, ff) = (cfg.hidden, cfg.q_dim(), cfg.kv_dim(), cfg.intermediate);
        // The output projection is the most sensitive matrix: keep it at 8
        // bits even when everything else is 4-bit, as llama.cpp does.
        let head_fmt = if format == Format::Q4 { Format::Q8 } else { format };
        let tied = cfg.tie_embeddings || st.get("lm_head.weight").is_none();
        let embed = mat("model.embed_tokens.weight", cfg.vocab, h, if tied { head_fmt } else { format })?;
        let lm_head = if tied { None } else { Some(mat("lm_head.weight", cfg.vocab, h, head_fmt)?) };
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("model.layers.{i}");
            let opt = |name: &str, len: usize| -> Result<Option<Vec<f32>>, String> {
                match st.get(&format!("{p}.{name}")) {
                    Some(_) => vector(st, &format!("{p}.{name}"), len).map(Some),
                    None => Ok(None),
                }
            };
            let layer = Layer {
                attn_norm: vector(st, &format!("{p}.input_layernorm.weight"), h)?,
                wq: mat(&format!("{p}.self_attn.q_proj.weight"), q, h, format)?,
                wk: mat(&format!("{p}.self_attn.k_proj.weight"), kv, h, format)?,
                wv: mat(&format!("{p}.self_attn.v_proj.weight"), kv, h, format)?,
                wo: mat(&format!("{p}.self_attn.o_proj.weight"), h, q, format)?,
                bq: opt("self_attn.q_proj.bias", q)?,
                bk: opt("self_attn.k_proj.bias", kv)?,
                bv: opt("self_attn.v_proj.bias", kv)?,
                q_norm: opt("self_attn.q_norm.weight", cfg.head_dim)?,
                k_norm: opt("self_attn.k_norm.weight", cfg.head_dim)?,
                ffn_norm: vector(st, &format!("{p}.post_attention_layernorm.weight"), h)?,
                w_gate: mat(&format!("{p}.mlp.gate_proj.weight"), ff, h, format)?,
                w_up: mat(&format!("{p}.mlp.up_proj.weight"), ff, h, format)?,
                w_down: mat(&format!("{p}.mlp.down_proj.weight"), h, ff, format)?,
            };
            if cfg.attn_bias && (layer.bq.is_none() || layer.bk.is_none() || layer.bv.is_none()) {
                return Err(format!("{p}: missing attention biases"));
            }
            if cfg.arch == Arch::Qwen3 && (layer.q_norm.is_none() || layer.k_norm.is_none()) {
                return Err(format!("{p}: missing q_norm/k_norm"));
            }
            layers.push(layer);
        }
        let norm = vector(st, "model.norm.weight", h)?;
        let inv_freq = cfg.rope_inv_freq();
        Ok(Model { cfg, format, embed, lm_head, layers, norm, inv_freq })
    }

    /// Bytes of weights held (shared mappings count fully).
    pub fn weight_bytes(&self) -> usize {
        let mut n = self.embed.bytes() + self.lm_head.as_ref().map_or(0, Weight::bytes);
        for l in &self.layers {
            n += [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down].iter().map(|w| w.bytes()).sum::<usize>();
        }
        n
    }

    pub fn param_count(&self) -> usize {
        let c = &self.cfg;
        let per_layer = c.hidden * (2 * c.q_dim() + 2 * c.kv_dim()) + 3 * c.hidden * c.intermediate + 2 * c.hidden;
        c.vocab * c.hidden * if self.lm_head.is_some() { 2 } else { 1 } + c.layers * per_layer + c.hidden
    }

    fn head(&self) -> &Weight {
        self.lm_head.as_ref().unwrap_or(&self.embed)
    }
}

/// Keys and values of every processed position, per layer.
pub struct KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    pub capacity: usize,
    pub len: usize,
    layer_stride: usize,
}

impl KvCache {
    pub fn new(cfg: &Config, capacity: usize) -> KvCache {
        let layer_stride = capacity * cfg.kv_dim();
        KvCache {
            k: vec![0.0; cfg.layers * layer_stride],
            v: vec![0.0; cfg.layers * layer_stride],
            capacity,
            len: 0,
            layer_stride,
        }
    }

    pub fn bytes(&self) -> usize {
        (self.k.len() + self.v.len()) * 4
    }
}

/// Scratch buffers, sized for a batch of `t` tokens.
#[derive(Default)]
struct Scratch {
    t: usize,
    x: Vec<f32>,
    xn: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    o: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    xq: Vec<ActQ8>,
}

/// One sequence being processed: a KV cache plus scratch space.
pub struct State {
    pub cache: KvCache,
    scratch: Scratch,
}

impl State {
    pub fn new(model: &Model, capacity: usize) -> State {
        State { cache: KvCache::new(&model.cfg, capacity.min(model.cfg.max_position)), scratch: Scratch::default() }
    }
}

/// Matrix product for a batch: y[t][..] = W x[t] for each of the t rows.
fn matmul_into(
    pool: &Pool,
    jobs: &[(&Weight, SyncPtr<f32>, usize)],
    x: &[f32],
    xq: &mut [ActQ8],
    t: usize,
    cols: usize,
) {
    if jobs.iter().any(|j| j.0.is_quantized()) {
        quant::quantize_act(&x[..t * cols], &mut xq[..t * cols / QK]);
    }
    let xq: &[ActQ8] = xq;
    // Work items are (job, chunk of rows), so several matrices that share
    // the same input (q/k/v, gate/up) run in a single parallel region.
    const ROWS: usize = 16;
    let mut items = Vec::new();
    for (j, (w, _, _)) in jobs.iter().enumerate() {
        for r in (0..w.rows).step_by(ROWS) {
            items.push((j, r));
        }
    }
    let nb = cols / QK;
    pool.run(items.len(), &|i| {
        let (j, r0) = items[i];
        let (w, out, stride) = &jobs[j];
        let r1 = (r0 + ROWS).min(w.rows);
        let c = w.cols;
        for r in r0..r1 {
            let mut tt = 0;
            while tt < t {
                let n = (t - tt).min(4);
                let res: [f32; 4] = match &w.data {
                    Data::F32(b) => {
                        let row = &b.as_f32()[r * c..(r + 1) * c];
                        dot4(
                            n,
                            |k| &x[(tt + k) * cols..(tt + k + 1) * cols],
                            |xs| dot_f32_n(row, xs),
                            |x1| dot_f32_n(row, x1),
                        )
                    }
                    Data::BF16(b) => {
                        let row = &b.as_u16()[r * c..(r + 1) * c];
                        dot4(
                            n,
                            |k| &x[(tt + k) * cols..(tt + k + 1) * cols],
                            |xs| dot_bf16_n(row, xs),
                            |x1| dot_bf16_n(row, x1),
                        )
                    }
                    Data::Q8(v) => {
                        let row = &v[r * nb..(r + 1) * nb];
                        dot4(
                            n,
                            |k| &xq[(tt + k) * nb..(tt + k + 1) * nb],
                            |xs| dot_q8_n(row, xs),
                            |x1| dot_q8_n(row, x1),
                        )
                    }
                    Data::Q4(v) => {
                        let row = &v[r * nb..(r + 1) * nb];
                        dot4(
                            n,
                            |k| &xq[(tt + k) * nb..(tt + k + 1) * nb],
                            |xs| dot_q4_n(row, xs),
                            |x1| dot_q4_n(row, x1),
                        )
                    }
                };
                for (k, &v) in res.iter().enumerate().take(n) {
                    // SAFETY: each (row, token) output cell is written by exactly one item.
                    unsafe { *out.0.add((tt + k) * stride + r) = v };
                }
                tt += n;
            }
        }
    });
}

/// Dispatch four rows at once, or one at a time for a ragged tail.
#[inline(always)]
fn dot4<'a, T: ?Sized + 'a>(
    n: usize,
    xs: impl Fn(usize) -> &'a T,
    four: impl Fn([&'a T; 4]) -> [f32; 4],
    one: impl Fn([&'a T; 1]) -> [f32; 1],
) -> [f32; 4] {
    if n == 4 {
        four([xs(0), xs(1), xs(2), xs(3)])
    } else {
        let mut r = [0f32; 4];
        for (k, v) in r.iter_mut().enumerate().take(n) {
            *v = one([xs(k)])[0];
        }
        r
    }
}

fn rope(v: &mut [f32], head_dim: usize, pos: usize, inv_freq: &[f32]) {
    let half = head_dim / 2;
    for head in v.chunks_exact_mut(head_dim) {
        for i in 0..half {
            let (sin, cos) = (pos as f32 * inv_freq[i]).sin_cos();
            let (a, b) = (head[i], head[i + half]);
            head[i] = a * cos - b * sin;
            head[i + half] = b * cos + a * sin;
        }
    }
}

impl Model {
    /// Run `tokens` (at positions cache.len ..) through the model, append
    /// them to the cache, and return the logits of the last token, or of
    /// every token when `all_logits` is set.
    pub fn forward(&self, pool: &Pool, state: &mut State, tokens: &[u32], all_logits: bool) -> Vec<f32> {
        let c = &self.cfg;
        let t = tokens.len();
        assert!(t > 0);
        assert!(state.cache.len + t <= state.cache.capacity, "context length exceeded");
        let (h, qd, kvd, ff, hd) = (c.hidden, c.q_dim(), c.kv_dim(), c.intermediate, c.head_dim);
        let start = state.cache.len;
        let s = &mut state.scratch;
        if s.t < t {
            *s = Scratch {
                t,
                x: vec![0.0; t * h],
                xn: vec![0.0; t * h.max(qd)],
                q: vec![0.0; t * qd],
                k: vec![0.0; t * kvd],
                v: vec![0.0; t * kvd],
                att: vec![0.0; t * qd],
                o: vec![0.0; t * h],
                gate: vec![0.0; t * ff],
                up: vec![0.0; t * ff],
                xq: vec![ActQ8::default(); t * h.max(qd).max(ff) / QK + 1],
            };
        }
        for (i, &tok) in tokens.iter().enumerate() {
            self.embed.row(tok as usize, &mut s.x[i * h..(i + 1) * h]);
        }

        for (li, l) in self.layers.iter().enumerate() {
            // ---- attention ----
            for i in 0..t {
                rmsnorm(&s.x[i * h..(i + 1) * h], &l.attn_norm, c.rms_eps, &mut s.xn[i * h..(i + 1) * h]);
            }
            matmul_into(
                pool,
                &[
                    (&l.wq, SyncPtr(s.q.as_mut_ptr()), qd),
                    (&l.wk, SyncPtr(s.k.as_mut_ptr()), kvd),
                    (&l.wv, SyncPtr(s.v.as_mut_ptr()), kvd),
                ],
                &s.xn,
                &mut s.xq,
                t,
                h,
            );
            for i in 0..t {
                let (q, k, v) = (
                    &mut s.q[i * qd..(i + 1) * qd],
                    &mut s.k[i * kvd..(i + 1) * kvd],
                    &mut s.v[i * kvd..(i + 1) * kvd],
                );
                if let (Some(bq), Some(bk), Some(bv)) = (&l.bq, &l.bk, &l.bv) {
                    q.iter_mut().zip(bq).for_each(|(a, b)| *a += b);
                    k.iter_mut().zip(bk).for_each(|(a, b)| *a += b);
                    v.iter_mut().zip(bv).for_each(|(a, b)| *a += b);
                }
                if let (Some(qn), Some(kn)) = (&l.q_norm, &l.k_norm) {
                    q.chunks_exact_mut(hd).for_each(|hq| rmsnorm_inplace(hq, qn, c.rms_eps));
                    k.chunks_exact_mut(hd).for_each(|hk| rmsnorm_inplace(hk, kn, c.rms_eps));
                }
                rope(q, hd, start + i, &self.inv_freq);
                rope(k, hd, start + i, &self.inv_freq);
                let base = li * state.cache.layer_stride + (start + i) * kvd;
                state.cache.k[base..base + kvd].copy_from_slice(k);
                state.cache.v[base..base + kvd].copy_from_slice(v);
            }
            self.attention(pool, state_cache_layer(&state.cache, li), s, t, start);
            matmul_into(pool, &[(&l.wo, SyncPtr(s.o.as_mut_ptr()), h)], &s.att, &mut s.xq, t, qd);
            s.x.iter_mut().zip(&s.o[..t * h]).for_each(|(a, b)| *a += b);

            // ---- feed-forward (SwiGLU) ----
            for i in 0..t {
                rmsnorm(&s.x[i * h..(i + 1) * h], &l.ffn_norm, c.rms_eps, &mut s.xn[i * h..(i + 1) * h]);
            }
            matmul_into(
                pool,
                &[(&l.w_gate, SyncPtr(s.gate.as_mut_ptr()), ff), (&l.w_up, SyncPtr(s.up.as_mut_ptr()), ff)],
                &s.xn,
                &mut s.xq,
                t,
                h,
            );
            silu_mul(&mut s.gate[..t * ff], &s.up[..t * ff]);
            matmul_into(pool, &[(&l.w_down, SyncPtr(s.o.as_mut_ptr()), h)], &s.gate, &mut s.xq, t, ff);
            s.x.iter_mut().zip(&s.o[..t * h]).for_each(|(a, b)| *a += b);
        }
        state.cache.len += t;

        // ---- output ----
        let rows: Vec<usize> = if all_logits { (0..t).collect() } else { vec![t - 1] };
        let n = rows.len();
        for (j, &i) in rows.iter().enumerate() {
            rmsnorm(&s.x[i * h..(i + 1) * h], &self.norm, c.rms_eps, &mut s.xn[j * h..(j + 1) * h]);
        }
        let mut logits = vec![0f32; n * c.vocab];
        matmul_into(pool, &[(self.head(), SyncPtr(logits.as_mut_ptr()), c.vocab)], &s.xn, &mut s.xq, n, h);
        logits
    }

    /// Causal multi-head attention with grouped KV heads, parallel over
    /// (token, head) pairs.
    fn attention(&self, pool: &Pool, (keys, values): (&[f32], &[f32]), s: &mut Scratch, t: usize, start: usize) {
        thread_local! {
            static SCORES: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let c = &self.cfg;
        let (hd, kvd, qd) = (c.head_dim, c.kv_dim(), c.q_dim());
        let group = c.heads / c.kv_heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let att = SyncPtr(s.att.as_mut_ptr());
        let q = &s.q;
        pool.run(t * c.heads, &|item| {
            let (i, head) = (item / c.heads, item % c.heads);
            let kvh = head / group;
            let qv = &q[i * qd + head * hd..i * qd + (head + 1) * hd];
            let len = start + i + 1; // causal: positions 0 ..= start + i
            SCORES.with_borrow_mut(|sc| {
                sc.resize(len, 0.0);
                for (p, s) in sc.iter_mut().enumerate() {
                    let kv = &keys[p * kvd + kvh * hd..p * kvd + (kvh + 1) * hd];
                    *s = dot_f32_n(kv, [qv])[0] * scale;
                }
                softmax(sc);
                // SAFETY: each (token, head) output slice is written by one item.
                let out = unsafe { att.slice(i * qd + head * hd, hd) };
                out.fill(0.0);
                for (p, &w) in sc.iter().enumerate() {
                    let vv = &values[p * kvd + kvh * hd..p * kvd + (kvh + 1) * hd];
                    for (o, &x) in out.iter_mut().zip(vv) {
                        *o += w * x;
                    }
                }
            });
        });
    }
}

fn state_cache_layer(cache: &KvCache, layer: usize) -> (&[f32], &[f32]) {
    let a = layer * cache.layer_stride;
    (&cache.k[a..a + cache.layer_stride], &cache.v[a..a + cache.layer_stride])
}

/// The kernels in use, for logs.
pub fn kernel_name() -> &'static str {
    kernels::describe()
}
