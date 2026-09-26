//! pankhllm's own decision model. No external service, no GPU.
//!
//! A multinomial logistic regression over hashed word and bigram features of the
//! question's *shape* (entities abstracted to <code>, <date>, <num>, <text>). It is
//! trained by the router itself: the generative planner is the teacher, every plan
//! it makes is a labelled example, catalog `examples` seed it on day one, and the
//! overnight miner retrains it. Inference is a few dozen multiply-adds.
//!
//! At decision time it only ever chooses among the candidates the caller passes
//! (the operations the question structurally fits, plus UNSUPPORTED), with
//! probabilities renormalised over those candidates.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::cache::hash64;

const DIM: usize = 1 << 16;

fn words_of(shape: &str) -> Vec<String> {
    shape
        .split(|c: char| !(c.is_alphanumeric() || c == '<' || c == '>' || c == '_'))
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

fn word_id(w: &str) -> u32 {
    (hash64(&format!("u:{w}")) as usize % DIM) as u32
}

/// Tokens of the abstracted question: words, typed slot markers and bigrams.
pub fn features(shape: &str) -> Vec<u32> {
    let words: Vec<String> = shape
        .split(|c: char| !(c.is_alphanumeric() || c == '<' || c == '>' || c == '_'))
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect();
    let mut f: Vec<u32> = Vec::with_capacity(words.len() * 2 + 1);
    f.push((hash64("__bias__") as usize % DIM) as u32);
    for (i, w) in words.iter().enumerate() {
        f.push((hash64(&format!("u:{w}")) as usize % DIM) as u32);
        if i + 1 < words.len() {
            f.push((hash64(&format!("b:{w} {}", words[i + 1])) as usize % DIM) as u32);
        }
    }
    f.sort_unstable();
    f.dedup();
    f
}

/// Portable bundle of trained models: `pankhllm export-models`, `decisions.models_file`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsFile {
    pub format: String,
    pub exported_at_ms: u64,
    /// "plan", "tool", "skill".
    pub models: std::collections::BTreeMap<String, NativeModel>,
}

impl ModelsFile {
    pub const FORMAT: &'static str = "pankhllm-models/1";

    pub fn read(path: &str) -> anyhow::Result<Self> {
        let f: ModelsFile = serde_json::from_slice(&std::fs::read(path)?)?;
        anyhow::ensure!(f.format == Self::FORMAT, "unsupported models file format {}", f.format);
        Ok(f)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeModel {
    pub labels: Vec<String>,
    /// Sparse weights per label: feature index -> weight.
    pub weights: Vec<HashMap<u32, f32>>,
    pub trained_on: usize,
    /// Held-out accuracy of the top choice, when a holdout was large enough.
    pub holdout_accuracy: Option<f64>,
    /// Lowest threshold whose held-out precision reached the target, if any.
    pub recommended_threshold: Option<f64>,
    pub trained_at_ms: i64,
    /// Word features seen in questions some operation answered (sorted). Used to notice
    /// questions made mostly of words the model has never seen in its domain.
    #[serde(default)]
    pub known_words: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct Example {
    pub shape: String,
    pub label: String,
}

impl NativeModel {
    /// Share of the question's words the model has seen in answerable questions.
    pub fn known_ratio(&self, shape: &str) -> f64 {
        if self.known_words.is_empty() {
            return 1.0; // models stored before this field existed: no guard
        }
        let ws = words_of(shape);
        if ws.is_empty() {
            return 0.0;
        }
        ws.iter().filter(|w| self.known_words.binary_search(&word_id(w)).is_ok()).count() as f64 / ws.len() as f64
    }

    fn score(&self, feats: &[u32], li: usize) -> f32 {
        let w = &self.weights[li];
        feats.iter().map(|f| w.get(f).copied().unwrap_or(0.0)).sum()
    }

    /// Probabilities over `candidates` only (unknown candidates get no mass).
    /// Returns (best label, its probability) when at least one candidate is known.
    pub fn choose(&self, shape: &str, candidates: &[String]) -> Option<(String, f64)> {
        let feats = features(shape);
        let scored: Vec<(String, f32)> = candidates
            .iter()
            .filter_map(|c| self.labels.iter().position(|l| l == c).map(|i| (c.clone(), self.score(&feats, i))))
            .collect();
        if scored.is_empty() {
            return None;
        }
        let max = scored.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = scored.iter().map(|(_, s)| (s - max).exp()).sum();
        let (best, s) = scored.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?;
        // A single known candidate would always get probability 1: that is not evidence.
        if scored.len() == 1 {
            return None;
        }
        Some((best.clone(), (((s - max).exp()) / z) as f64))
    }
}

/// Train with SGD on softmax cross-entropy, class-balanced, with L2. Deterministic.
pub fn train(examples: &[Example], epochs: usize) -> Option<NativeModel> {
    // Distinct (shape, label) pairs: a template repeated ten thousand times must not
    // teach ten thousand times more than a phrasing seen once.
    let mut seen = std::collections::HashSet::new();
    let distinct: Vec<Example> = examples.iter().filter(|e| seen.insert((e.shape.clone(), e.label.clone()))).cloned().collect();
    let trained_on = examples.len();
    let examples = &distinct[..];
    let mut labels: Vec<String> = examples.iter().map(|e| e.label.clone()).collect();
    labels.sort();
    labels.dedup();
    if labels.len() < 2 || examples.len() < 2 {
        return None;
    }
    let data: Vec<(Vec<u32>, usize)> = examples.iter().map(|e| (features(&e.shape), labels.iter().position(|l| *l == e.label).unwrap())).collect();
    let mut counts = vec![0usize; labels.len()];
    for (_, y) in &data {
        counts[*y] += 1;
    }
    // Balanced, but capped: a rare class must not dominate by weight alone.
    let class_w: Vec<f32> = counts.iter().map(|c| ((data.len() as f32) / (labels.len() as f32 * (*c).max(1) as f32)).clamp(0.5, 3.0)).collect();
    let mut w: Vec<HashMap<u32, f32>> = vec![HashMap::new(); labels.len()];
    let l2 = 1e-3f32;
    let mut order: Vec<usize> = (0..data.len()).collect();
    for epoch in 0..epochs {
        // Deterministic shuffle.
        order.sort_by_key(|i| hash64(&format!("{epoch}:{i}")));
        let lr = 0.5 / (1.0 + epoch as f32 * 0.2);
        for &i in &order {
            let (feats, y) = &data[i];
            let scores: Vec<f32> = (0..labels.len()).map(|k| feats.iter().map(|f| w[k].get(f).copied().unwrap_or(0.0)).sum()).collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let z: f32 = exps.iter().sum();
            for k in 0..labels.len() {
                let p = exps[k] / z;
                let g = (p - if k == *y { 1.0 } else { 0.0 }) * class_w[*y];
                for f in feats {
                    let e = w[k].entry(*f).or_insert(0.0);
                    *e -= lr * (g + l2 * *e);
                }
            }
        }
    }
    for wk in w.iter_mut() {
        wk.retain(|_, v| v.abs() > 1e-4);
    }
    let mut known_words: Vec<u32> = examples.iter().filter(|e| e.label != "UNSUPPORTED").flat_map(|e| words_of(&e.shape).into_iter().map(|w| word_id(&w))).collect();
    known_words.sort_unstable();
    known_words.dedup();
    Some(NativeModel { labels, weights: w, trained_on, holdout_accuracy: None, recommended_threshold: None, trained_at_ms: crate::store::now_ms(), known_words })
}

/// Train on 80%, measure on 20% (when there is enough data), then retrain on all of it.
/// The recommended threshold is the lowest at which held-out precision reaches `target`.
pub fn train_with_holdout(examples: &[Example], target_precision: f64) -> Option<NativeModel> {
    let (mut hold, mut fit): (Vec<Example>, Vec<Example>) = (Vec::new(), Vec::new());
    for e in examples {
        if hash64(&e.shape).is_multiple_of(5) { hold.push(e.clone()) } else { fit.push(e.clone()) }
    }
    let mut model = train(examples, 30)?;
    if hold.len() >= 20 {
        if let Some(m) = train(&fit, 30) {
            let preds: Vec<(f64, bool)> = hold
                .iter()
                .filter_map(|e| m.choose(&e.shape, &m.labels).map(|(l, p)| (p, l == e.label)))
                .collect();
            let acc = preds.iter().filter(|(_, ok)| *ok).count() as f64 / preds.len().max(1) as f64;
            model.holdout_accuracy = Some(acc);
            for th in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] {
                let cov: Vec<&(f64, bool)> = preds.iter().filter(|(p, _)| *p >= th).collect();
                if cov.len() >= 10 && cov.iter().filter(|(_, ok)| *ok).count() as f64 / cov.len() as f64 >= target_precision {
                    model.recommended_threshold = Some(th);
                    break;
                }
            }
        }
    }
    Some(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(shape: &str, label: &str) -> Example {
        Example { shape: shape.into(), label: label.into() }
    }

    #[test]
    fn learns_shapes_and_respects_candidates() {
        let data = vec![
            ex("what were calls for <code> in <date>", "metric"),
            ex("reach for <code> in <date>", "metric"),
            ex("revenue of <code> during <date>", "metric"),
            ex("how many calls did <code> log in <date>", "metric"),
            ex("top <num> territories by calls in <date>", "top"),
            ex("which <num> territories had the most revenue in <date>", "top"),
            ex("best <num> territories by reach in <date>", "top"),
            ex("who covers <code>", "owner"),
            ex("which rep owns territory <code>", "owner"),
            ex("owner of <code>", "owner"),
            ex("why did calls fall in <code>", "UNSUPPORTED"),
            ex("write a poem about sales", "UNSUPPORTED"),
        ];
        let m = train(&data, 30).unwrap();
        let all: Vec<String> = m.labels.clone();
        assert_eq!(m.choose("calls for <code> in <date>", &all).unwrap().0, "metric");
        assert_eq!(m.choose("top <num> territories by revenue in <date>", &all).unwrap().0, "top");
        assert_eq!(m.choose("who owns <code>", &all).unwrap().0, "owner");
        // Restricted to two candidates, probabilities renormalise over them.
        let (l, p) = m.choose("who owns <code>", &["owner".into(), "UNSUPPORTED".into()]).unwrap();
        assert_eq!(l, "owner");
        assert!(p > 0.5);
        // One known candidate is not evidence.
        assert!(m.choose("who owns <code>", &["owner".into(), "never_seen".into()]).is_none());
        // Familiar wording vs mostly unseen words.
        assert!(m.known_ratio("who owns <code>") > 0.99);
        assert!(m.known_ratio("give me tips for better time management") < 0.5);
        // Round-trips through JSON (how it is stored).
        let back: NativeModel = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
        assert_eq!(back.choose("owner of <code>", &all).unwrap().0, "owner");
    }
}
