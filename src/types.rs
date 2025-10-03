use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub metadata: ChunkMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChunkMetadata {
    #[serde(default)]
    pub source_file: String,
    
    #[serde(default)]
    pub page_number: u32,
    
    #[serde(default)]
    pub document_title: Option<String>,
    
    #[serde(flatten)]
    pub extra_fields: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub score: f32,
    pub chunk: Chunk,
    pub search_type: SearchType,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SearchType {
    Keyword,
    Semantic,
    Hybrid,
}