//! Heuristic intent classifier. Zero latency, zero cost, explainable.
//! Order matters: the more specific patterns are checked first.

use crate::types::Intent;

const CODE_MARKERS: &[&str] = &[
    "```", "stack trace", "traceback", "compile error", "segfault", "nullpointer",
    "function ", "fn ", "def ", "class ", "import ", "select ", "regex", "refactor",
    "implement ", "write a script", "write code", "unit test", "typescript", "python",
    "rust", "golang", "javascript", "java ", "c#", "sql", "bash ", "dockerfile", "kubernetes",
    "dax", "pandas", "snippet", "measure for", "write a query", "query for", "powershell", "excel formula",
    "vba", "spark", "pyspark", "notebook", "stored procedure", "generate a", "pivot ", "dataframe",
];

const SUMMARY_MARKERS: &[&str] = &[
    "summarize", "summarise", "summary", "tl;dr", "tldr", "overview of", "key points",
    "main points", "in brief", "recap", "give me the gist",
];

const EXTRACTION_MARKERS: &[&str] = &[
    "list all", "list every", "extract", "enumerate", "all the ", "every ", "table of",
    "as json", "in json", "as a table", "as csv", "pull out",
];

const SYNTHESIS_MARKERS: &[&str] = &[
    "compare", "comparison", "difference between", "differences between", "contrast",
    " versus ", " vs ", " vs. ", "pros and cons", "trade-off", "tradeoff", "relationship between",
    "how does", "how do", "how is", "how are", "relate",
];

const REASONING_MARKERS: &[&str] = &[
    "why ", "why?", "explain", "what would happen", "what happens if", "should we",
    "should i", "prove", "implication", "predict", "estimate", "calculate", "derive",
    "reason", "justify", "argue", "recommend", "best way", "optimal", "strategy", "plan ",
    "root cause", "diagnose", "cause", "likely", "driver", "hypothes", "analy", "impact",
    "what should", "next step", "investigate", "assess",
];

const CHITCHAT: &[&str] = &[
    "hi", "hello", "hey", "hi there", "hello there", "hey there", "thanks", "thank you", "thanks a lot", "ok", "okay",
    "cool", "bye", "goodbye", "good morning", "good evening", "good afternoon", "yo", "sup", "how are you", "great", "nice",
];

const WH_WORDS: &[&str] = &["what", "when", "where", "who", "which", "how many", "how much", "whom", "whose"];

/// Under-specified questions: too short to carry a subject, or leaning on a
/// reference ("it", "that one", "same for") with no earlier turn to resolve it.
pub fn looks_ambiguous(query: &str, has_history: bool, n_chunks: usize) -> bool {
    let lower = query.trim().to_ascii_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    if words.is_empty() {
        return true;
    }
    let padded = format!(" {lower} ");
    let dangling = [" it ", " it?", " that ", " that?", " this ", " this?", " those ", " same ", " what about ", " and for ", " how about ", " the other ", " them "];
    let leans_on_reference = dangling.iter().any(|d| padded.contains(d)) || lower.starts_with("and ") || lower.starts_with("also ");
    if leans_on_reference && !has_history {
        return true;
    }
    // Two or three words with no context: "numbers?", "status east", "trend". A short
    // question that names something specific (a code, a date, a number) is not vague.
    let names_something = query.chars().any(|c| c.is_ascii_digit());
    words.len() <= 3 && n_chunks == 0 && !has_history && !names_something && !CHITCHAT.contains(&lower.trim_end_matches(['!', '.', '?']))
}

pub fn looks_like_code(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    if lower.contains("```") {
        return true;
    }
    let symbol_hits = ["{", "}", "();", "=>", "->", "::", "</", "/>", "==", "!="]
        .iter()
        .filter(|s| lower.contains(**s))
        .count();
    symbol_hits >= 2
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

pub fn classify(query: &str, n_chunks: usize) -> Intent {
    let lower = query.trim().to_ascii_lowercase();
    let padded = format!(" {lower} ");
    let words: Vec<&str> = lower.split_whitespace().collect();

    if words.is_empty() {
        return Intent::Chitchat;
    }
    if words.len() <= 4 && n_chunks == 0 && CHITCHAT.contains(&lower.trim_end_matches(['!', '.', '?'])) {
        return Intent::Chitchat;
    }
    if looks_like_code(query) || contains_any(&padded, CODE_MARKERS) {
        return Intent::Code;
    }
    if contains_any(&padded, SUMMARY_MARKERS) {
        return Intent::Summarization;
    }
    if contains_any(&padded, EXTRACTION_MARKERS) {
        return Intent::Extraction;
    }

    // Two or more questions, or two wh-words joined by a connective, is a chain.
    let question_marks = lower.matches('?').count();
    let wh_hits: usize = WH_WORDS.iter().map(|w| padded.matches(&format!(" {w}")).count()).sum();
    let has_connective = contains_any(&padded, &[" and then ", " after that ", " then ", " and also ", " and what ", " and who ", " and how ", " and which ", ", and "]);
    if question_marks >= 2 || (wh_hits >= 2 && has_connective) {
        return Intent::MultiHop;
    }

    if contains_any(&padded, SYNTHESIS_MARKERS) {
        return Intent::Synthesis;
    }
    if contains_any(&padded, REASONING_MARKERS) {
        return Intent::Reasoning;
    }

    // Long, open-ended prompts without a clear pattern deserve a stronger model.
    if words.len() > 60 {
        return Intent::Synthesis;
    }
    Intent::Lookup
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(q: &str) -> Intent {
        classify(q, 3)
    }

    #[test]
    fn lookups() {
        assert_eq!(c("What is the refund window for annual plans?"), Intent::Lookup);
        assert_eq!(c("When was the contract signed?"), Intent::Lookup);
        assert_eq!(c("Who is the account owner for ACME?"), Intent::Lookup);
    }

    #[test]
    fn extraction_and_summary() {
        assert_eq!(c("List all the dates mentioned in the policy"), Intent::Extraction);
        assert_eq!(c("Extract the parties and amounts as JSON"), Intent::Extraction);
        assert_eq!(c("Summarize the onboarding guide"), Intent::Summarization);
        assert_eq!(c("tl;dr of the incident report"), Intent::Summarization);
    }

    #[test]
    fn synthesis_and_reasoning() {
        assert_eq!(c("Compare the enterprise and team plans"), Intent::Synthesis);
        assert_eq!(c("What is the difference between SLA and SLO here?"), Intent::Synthesis);
        assert_eq!(c("Why did the migration fail according to the postmortem?"), Intent::Reasoning);
        assert_eq!(c("Should we move the billing service to the new region?"), Intent::Reasoning);
        assert_eq!(c("NRx fell 18% while calls rose. Two most likely causes, one line each."), Intent::Reasoning);
        assert_eq!(c("What should the MSL check first after the formulary change?"), Intent::Reasoning);
    }

    #[test]
    fn multi_hop_and_code() {
        assert_eq!(c("Who approved the Q3 budget, and what did they say about hiring?"), Intent::MultiHop);
        assert_eq!(c("Which team owns the API? What is their on-call rotation?"), Intent::MultiHop);
        assert_eq!(c("Fix this python function so it handles None"), Intent::Code);
        assert_eq!(c("```\nfn main() {}\n```\nwhy does this not compile"), Intent::Code);
    }

    #[test]
    fn ambiguity() {
        assert!(looks_ambiguous("what about west?", false, 0));
        assert!(looks_ambiguous("numbers?", false, 0));
        assert!(looks_ambiguous("and the same for Q3", false, 2));
        assert!(!looks_ambiguous("what about west?", true, 0), "resolvable with history");
        assert!(!looks_ambiguous("What was call activity for East in August?", false, 0));
        assert!(!looks_ambiguous("hello", false, 0));
        assert!(!looks_ambiguous("owner of T-300", false, 0), "short but specific");
        assert!(!looks_ambiguous("Who covers T-112?", false, 0));
    }

    #[test]
    fn chitchat() {
        assert_eq!(classify("hello", 0), Intent::Chitchat);
        assert_eq!(classify("thanks!", 0), Intent::Chitchat);
        // With context present, a short greeting is still treated as a real query.
        assert_eq!(classify("hello", 2), Intent::Lookup);
    }
}
