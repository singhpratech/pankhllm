//! Response cache: exact-key LRU plus optional semantic (paraphrase) lookup.
//! Everything is in memory and per instance; entries expire by TTL.

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::config::CacheConfig;
use crate::types::{CacheInfo, Confidence, ToolCall, Usage};

#[derive(Debug, Clone)]
pub struct CachedAnswer {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub model: String,
    pub provider_model: String,
    pub usage: Usage,
    pub confidence: Confidence,
}

#[derive(Debug, Clone)]
struct Entry {
    answer: CachedAnswer,
    created: Instant,
    /// Everything except the question text: context, tools, prompt, tags, cache_key.
    scope: u64,
    vector: Option<Vec<f32>>,
}

#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct CacheStats {
    pub entries: usize,
    pub exact_hits: u64,
    pub semantic_hits: u64,
    pub misses: u64,
    pub stores: u64,
}

pub struct ResponseCache {
    cfg: CacheConfig,
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    map: HashMap<u64, Entry>,
    order: VecDeque<u64>,
    stats: CacheStats,
}

/// Stable 64-bit hash of a canonical string. Collisions are theoretically
/// possible; the exact key also stores the scope so a collision across scopes
/// cannot serve the wrong answer.
pub fn hash64(s: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

impl ResponseCache {
    pub fn new(cfg: CacheConfig) -> Self {
        Self { cfg, inner: Mutex::new(Inner::default()) }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn ttl(&self) -> Duration {
        Duration::from_secs(self.cfg.ttl_secs.max(1))
    }

    pub fn stats(&self) -> CacheStats {
        let mut s = self.lock().stats.clone();
        s.entries = self.lock().map.len();
        s
    }

    /// Exact lookup. `key` is the full request hash (scope + question).
    pub fn get_exact(&self, key: u64, scope: u64) -> Option<(CachedAnswer, CacheInfo)> {
        let ttl = self.ttl();
        let mut g = self.lock();
        let hit = match g.map.get(&key) {
            Some(e) if e.scope == scope && e.created.elapsed() < ttl => Some((e.answer.clone(), e.created.elapsed().as_secs())),
            _ => None,
        };
        match hit {
            Some((a, age)) => {
                g.stats.exact_hits += 1;
                Some((a, CacheInfo { hit: true, kind: "exact".into(), similarity: None, age_secs: age }))
            }
            None => None,
        }
    }

    /// Semantic lookup within the same scope. Brute force over live entries; fine
    /// for tens of thousands of entries at a few hundred dimensions.
    pub fn get_semantic(&self, vector: &[f32], scope: u64) -> Option<(CachedAnswer, CacheInfo)> {
        let threshold = self.cfg.semantic.as_ref()?.threshold;
        let ttl = self.ttl();
        let mut g = self.lock();
        let mut best: Option<(f64, CachedAnswer, u64)> = None;
        for e in g.map.values() {
            if e.scope != scope || e.created.elapsed() >= ttl {
                continue;
            }
            if let Some(v) = &e.vector {
                let sim = cosine(vector, v);
                if sim >= threshold && best.as_ref().is_none_or(|b| sim > b.0) {
                    best = Some((sim, e.answer.clone(), e.created.elapsed().as_secs()));
                }
            }
        }
        match best {
            Some((sim, a, age)) => {
                g.stats.semantic_hits += 1;
                Some((a, CacheInfo { hit: true, kind: "semantic".into(), similarity: Some(sim), age_secs: age }))
            }
            None => None,
        }
    }

    pub fn miss(&self) {
        self.lock().stats.misses += 1;
    }

    pub fn put(&self, key: u64, scope: u64, answer: CachedAnswer, vector: Option<Vec<f32>>) {
        let mut g = self.lock();
        if !g.map.contains_key(&key) {
            g.order.push_back(key);
        }
        g.map.insert(key, Entry { answer, created: Instant::now(), scope, vector });
        g.stats.stores += 1;
        while g.map.len() > self.cfg.max_entries.max(1) {
            match g.order.pop_front() {
                Some(old) => {
                    g.map.remove(&old);
                }
                None => break,
            }
        }
    }

    pub fn cacheable(&self, c: &Confidence, clarification: bool, abstained: bool, has_tool_calls: bool) -> bool {
        if clarification || abstained || c.hedged {
            return false;
        }
        has_tool_calls || c.score >= self.cfg.min_confidence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ans(t: &str) -> CachedAnswer {
        CachedAnswer { text: t.into(), tool_calls: vec![], model: "m".into(), provider_model: "pm".into(), usage: Usage::default(), confidence: Confidence { score: 0.9, ..Default::default() } }
    }

    #[test]
    fn exact_hit_requires_same_scope_and_respects_ttl_and_capacity() {
        let mut cfg = CacheConfig { enabled: true, ..Default::default() };
        cfg.max_entries = 2;
        let c = ResponseCache::new(cfg);
        c.put(1, 10, ans("a"), None);
        assert_eq!(c.get_exact(1, 10).unwrap().0.text, "a");
        assert!(c.get_exact(1, 11).is_none(), "different scope must not hit");
        c.put(2, 10, ans("b"), None);
        c.put(3, 10, ans("c"), None);
        assert!(c.get_exact(1, 10).is_none(), "oldest evicted at capacity");
        assert_eq!(c.stats().entries, 2);
    }

    #[test]
    fn semantic_hit_by_cosine_threshold() {
        let cfg = CacheConfig { enabled: true, semantic: Some(crate::config::SemanticCache { embedding_model: "e".into(), threshold: 0.9 }), ..Default::default() };
        let c = ResponseCache::new(cfg);
        c.put(1, 10, ans("east"), Some(vec![1.0, 0.0, 0.0]));
        c.put(2, 10, ans("west"), Some(vec![0.0, 1.0, 0.0]));
        let (a, info) = c.get_semantic(&[0.98, 0.05, 0.0], 10).unwrap();
        assert_eq!(a.text, "east");
        assert!(info.similarity.unwrap() > 0.9);
        assert!(c.get_semantic(&[0.6, 0.6, 0.0], 10).is_none(), "below threshold");
        assert!(c.get_semantic(&[1.0, 0.0, 0.0], 99).is_none(), "other scope");
    }

    #[test]
    fn not_cacheable_when_hedged_or_low_confidence() {
        let c = ResponseCache::new(CacheConfig { enabled: true, ..Default::default() });
        assert!(c.cacheable(&Confidence { score: 0.9, ..Default::default() }, false, false, false));
        assert!(!c.cacheable(&Confidence { score: 0.3, ..Default::default() }, false, false, false));
        assert!(c.cacheable(&Confidence { score: 0.3, ..Default::default() }, false, false, true), "tool calls are structured: cache them");
        assert!(!c.cacheable(&Confidence { score: 0.9, hedged: true, ..Default::default() }, false, false, false));
        assert!(!c.cacheable(&Confidence { score: 0.9, ..Default::default() }, true, false, false));
    }
}
