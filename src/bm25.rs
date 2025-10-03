use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Bm25Index {
    doc_frequencies: HashMap<String, usize>,
    doc_lengths: Vec<usize>,
    avg_doc_length: f32,
    total_docs: usize,
    inverted_index: HashMap<String, Vec<(usize, usize)>>,
    k1: f32,
    b: f32,
}

impl Bm25Index {
    pub fn new() -> Self {
        Self {
            doc_frequencies: HashMap::new(),
            doc_lengths: Vec::new(),
            avg_doc_length: 0.0,
            total_docs: 0,
            inverted_index: HashMap::new(),
            k1: 1.2,
            b: 0.75,
        }
    }

    pub fn search(&self, query: &str, limit: usize) -> Vec<(usize, f32)> {
        if self.total_docs == 0 {
            return Vec::new();
        }

        let query_tokens = tokenize(query);
        if query_tokens.is_empty() {
            return Vec::new();
        }

        let mut scores: HashMap<usize, f32> = HashMap::new();

        for doc_id in 0..self.total_docs {
            let score = self.calculate_score(&query_tokens, doc_id);
            if score > 0.0 {
                scores.insert(doc_id, score);
            }
        }

        let mut results: Vec<(usize, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    fn calculate_score(&self, query_tokens: &[String], doc_id: usize) -> f32 {
        let mut score = 0.0;
        let doc_length = self.doc_lengths[doc_id] as f32;
        let length_norm = 1.0 - self.b + self.b * (doc_length / self.avg_doc_length);

        for term in query_tokens {
            let tf = self.get_term_frequency(term, doc_id) as f32;
            if tf == 0.0 {
                continue;
            }

            let df = self.doc_frequencies.get(term).copied().unwrap_or(0) as f32;
            if df == 0.0 {
                continue;
            }

            let idf = ((self.total_docs as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();
            let normalized_tf = (tf * (self.k1 + 1.0)) / (tf + self.k1 * length_norm);
            score += idf * normalized_tf;
        }

        score
    }

    fn get_term_frequency(&self, term: &str, doc_id: usize) -> usize {
        self.inverted_index
            .get(term)
            .and_then(|postings| {
                postings
                    .iter()
                    .find(|(id, _)| *id == doc_id)
                    .map(|(_, freq)| *freq)
            })
            .unwrap_or(0)
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty() && s.len() >= 2)
        .map(|s| s.to_string())
        .collect()
}