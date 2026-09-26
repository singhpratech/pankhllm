//! Turn a request into the numbers the policy reasons about.

use crate::classifier::{classify, looks_ambiguous, looks_like_code};
use crate::config::Config;
use crate::types::{Intent, Phase, RouteRequest, Signals};

/// Cheap token estimate. Good enough for tier and context-window decisions;
/// swap in a real tokenizer if you need billing-grade counts.
pub fn estimate_tokens(text: &str) -> u32 {
    let chars = text.chars().count() as f64;
    (chars / 3.6).ceil() as u32
}

pub fn extract(req: &RouteRequest, cfg: &Config, rule_index: Option<usize>) -> Signals {
    let query = req.query();
    let query_tokens = estimate_tokens(query);
    let context_tokens: u32 = req.context.iter().map(|c| estimate_tokens(&c.text)).sum();
    let history_tokens: u32 = req
        .messages
        .iter()
        .map(|m| estimate_tokens(&m.content))
        .sum::<u32>()
        .saturating_sub(query_tokens);

    let top_score = req
        .context
        .iter()
        .filter_map(|c| c.score)
        .fold(None, |acc: Option<f64>, s| Some(acc.map_or(s, |a| a.max(s))));

    let weak_retrieval = match top_score {
        Some(s) => s < cfg.routing.weak_retrieval.threshold,
        // Chunks without scores can't be judged. No chunks at all is weak, unless the
        // request carries tools: in an agent loop the tools are the retrieval.
        None => req.context.is_empty() && req.tools.is_empty(),
    };

    // Precedence: explicit override > config rule > built-in heuristic.
    let rule = rule_index.and_then(|i| cfg.routing.rules.get(i));
    let (intent, intent_source) = match (req.intent, rule.and_then(|r| r.intent)) {
        (Some(i), _) => (i, "override"),
        (None, Some(i)) => (i, "rule"),
        (None, None) => (classify(query, req.context.len()), "heuristic"),
    };

    let phase = if req.tools.is_empty() {
        Phase::Plain
    } else if req.messages.last().is_some_and(|m| m.tool_call_id.is_some()) {
        Phase::AfterToolResult
    } else {
        Phase::ToolSelect
    };

    Signals {
        intent,
        phase,
        intent_source: intent_source.to_string(),
        matched_rule: rule.map(|r| r.name.clone()),
        query_tokens,
        context_tokens,
        history_tokens,
        total_input_tokens: query_tokens + context_tokens + history_tokens,
        n_chunks: req.context.len(),
        top_score,
        weak_retrieval,
        has_code: looks_like_code(query),
        required_tags: req.tags.clone(),
        ambiguous: req.tools.is_empty()
            && intent != Intent::Chitchat
            && looks_ambiguous(query, req.messages.iter().any(|m| m.role == "assistant"), req.context.len()),
    }
}
