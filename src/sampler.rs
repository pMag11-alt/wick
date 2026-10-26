//! Turning logits into the next token.

#[derive(Clone, Debug)]
pub struct SamplerConfig {
    /// 0 means greedy decoding.
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    /// Hugging Face style: logits of tokens already in the context are
    /// divided (if positive) or multiplied (if negative) by this.
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig { temperature: 0.7, top_k: 40, top_p: 0.9, min_p: 0.0, repetition_penalty: 1.0, seed: 0x5EED }
    }
}

impl SamplerConfig {
    pub fn greedy() -> Self {
        SamplerConfig { temperature: 0.0, ..Default::default() }
    }
}

/// xoshiro256** seeded through splitmix64.
pub struct Rng([u64; 4]);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng([next(), next(), next(), next()])
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.0;
        let out = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        out
    }

    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub struct Sampler {
    pub cfg: SamplerConfig,
    rng: Rng,
}

impl Sampler {
    pub fn new(cfg: SamplerConfig) -> Sampler {
        let rng = Rng::new(cfg.seed);
        Sampler { cfg, rng }
    }

    pub fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> u32 {
        let cfg = &self.cfg;
        if cfg.repetition_penalty != 1.0 {
            let mut seen = std::collections::HashSet::new();
            for &t in history {
                if seen.insert(t) && (t as usize) < logits.len() {
                    let l = &mut logits[t as usize];
                    *l = if *l > 0.0 { *l / cfg.repetition_penalty } else { *l * cfg.repetition_penalty };
                }
            }
        }
        if cfg.temperature <= 0.0 {
            return argmax(logits);
        }
        // Candidates sorted by logit, cut by top-k first (cheap).
        let k = if cfg.top_k == 0 { logits.len() } else { cfg.top_k.min(logits.len()) };
        let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
        if k < idx.len() {
            idx.select_nth_unstable_by(k - 1, |&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
            idx.truncate(k);
        }
        idx.sort_unstable_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
        let max = logits[idx[0] as usize];
        let mut probs: Vec<f32> = idx.iter().map(|&i| ((logits[i as usize] - max) / cfg.temperature).exp()).collect();
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);
        // min-p: drop tokens far less likely than the best one.
        let mut keep = probs.len();
        if cfg.min_p > 0.0 {
            let floor = probs[0] * cfg.min_p;
            keep = probs.iter().position(|&p| p < floor).unwrap_or(keep).max(1);
        }
        // top-p: the smallest prefix with at least top_p of the mass.
        if cfg.top_p < 1.0 {
            let mut acc = 0.0;
            for (i, &p) in probs[..keep].iter().enumerate() {
                acc += p;
                if acc >= cfg.top_p {
                    keep = i + 1;
                    break;
                }
            }
        }
        let total: f32 = probs[..keep].iter().sum();
        let mut r = self.rng.next_f32() * total;
        for i in 0..keep {
            r -= probs[i];
            if r <= 0.0 {
                return idx[i];
            }
        }
        idx[keep - 1]
    }
}

pub fn argmax(x: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best as u32
}

/// log-softmax of logits at `target`, for perplexity.
pub fn log_prob(logits: &[f32], target: u32) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&l| (l as f64 - max).exp()).sum();
    logits[target as usize] as f64 - max - sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_and_filters() {
        let mut logits = vec![0.0, 3.0, 1.0, 2.9];
        assert_eq!(Sampler::new(SamplerConfig::greedy()).sample(&mut logits.clone(), &[]), 1);
        // Repetition penalty demotes token 1 below token 3.
        let mut s = Sampler::new(SamplerConfig { temperature: 0.0, repetition_penalty: 1.5, ..Default::default() });
        assert_eq!(s.sample(&mut logits, &[1]), 3);
        // top_k = 1 always returns the best token.
        let mut s = Sampler::new(SamplerConfig { temperature: 1.0, top_k: 1, ..Default::default() });
        for _ in 0..50 {
            assert_eq!(s.sample(&mut [0.0, 3.0, 1.0, 2.9], &[]), 1);
        }
    }

    #[test]
    fn sampling_follows_the_distribution() {
        let mut s = Sampler::new(SamplerConfig { temperature: 1.0, top_k: 0, top_p: 1.0, ..Default::default() });
        let logits = [0.0f32, (3.0f32).ln()]; // probabilities 1/4 and 3/4
        let n = 20000;
        let ones = (0..n).filter(|_| s.sample(&mut logits.to_vec(), &[]) == 1).count();
        let frac = ones as f64 / n as f64;
        assert!((frac - 0.75).abs() < 0.02, "{frac}");
    }
}
