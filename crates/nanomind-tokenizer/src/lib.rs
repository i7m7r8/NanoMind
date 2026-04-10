//! BPE / token-level tokenizer.
//!
//! Loads vocabulary from GGUF metadata or tokenizer.json files.

use std::collections::HashMap;

/// Simple token-level tokenizer.
///
// For full BPE support, load a HuggingFace tokenizer.json alongside the model.
/// This implementation handles direct token mapping which works for most GGUF models
/// that include their vocab in metadata.
pub struct Tokenizer {
    /// Token ID → string.
    id_to_token: Vec<String>,
    /// String → token ID.
    token_to_id: HashMap<String, u32>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub unk_token_id: u32,
    /// Special token markers.
    pub special_tokens: Vec<String>,
}

impl Tokenizer {
    /// Create from vocab arrays (as stored in GGUF metadata).
    pub fn from_vocab(tokens: Vec<String>, bos_id: u32, eos_id: u32, unk_id: u32) -> Self {
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, token) in tokens.iter().enumerate() {
            token_to_id.insert(token.clone(), i as u32);
        }

        let special_tokens: Vec<String> = tokens
            .iter()
            .filter(|t| t.starts_with('<') && t.ends_with('>'))
            .cloned()
            .collect();

        Self {
            id_to_token: tokens,
            token_to_id,
            bos_token_id: bos_id,
            eos_token_id: eos_id,
            unk_token_id: unk_id,
            special_tokens,
        }
    }

    /// Load from HuggingFace tokenizer.json content.
    pub fn from_hf_json(json: &str) -> Result<Self, String> {
        // Parse the simplified format
        let vocab = extract_vocab_from_json(json)?;
        let (bos, eos, unk) = extract_special_tokens(json);

        Ok(Self::from_vocab(vocab, bos, eos, unk))
    }

    /// Encode text into token IDs.
    ///
    /// For a simple tokenizer, this tries to match the longest possible token
    /// from the vocabulary (greedy longest-match tokenization).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            // Try longest match first
            let mut found = false;

            // Check if the whole remaining text is a token
            if let Some(&id) = self.token_to_id.get(remaining) {
                tokens.push(id);
                break;
            }

            // Try prefixes of decreasing length
            for len in (1..=remaining.len().min(20)).rev() {
                let prefix = &remaining[..len];
                if let Some(&id) = self.token_to_id.get(prefix) {
                    tokens.push(id);
                    remaining = &remaining[len..];
                    found = true;
                    break;
                }
            }

            if !found {
                // Single character fallback
                let ch = remaining.chars().next().unwrap();
                let ch_str = ch.to_string();
                if let Some(&id) = self.token_to_id.get(&ch_str) {
                    tokens.push(id);
                } else {
                    // Try byte-level encoding
                    for byte in ch_str.bytes() {
                        let byte_token = format!("<0x{:02X}>", byte);
                        if let Some(&id) = self.token_to_id.get(&byte_token) {
                            tokens.push(id);
                        } else {
                            tokens.push(self.unk_token_id);
                        }
                    }
                }
                remaining = &remaining[ch.len_utf8()..];
            }
        }

        tokens
    }

    /// Decode token IDs to text.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut result = String::new();
        for &token_id in tokens {
            if let Some(token) = self.id_to_token.get(token_id as usize) {
                // Skip special tokens
                if token.starts_with('<') && token.ends_with('>') {
                    continue;
                }
                // Handle byte-level tokens
                if token.starts_with("Ġ") {
                    result.push(' ');
                    result.push_str(&token[1..]);
                } else if token.starts_with("<0x") && token.ends_with('>') {
                    // Byte token
                    let hex = &token[3..token.len() - 1];
                    if let Ok(byte) = u8::from_str_radix(hex, 16) {
                        result.push(byte as char);
                    }
                } else {
                    result.push_str(token);
                }
            }
        }
        result
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Get token string by ID.
    pub fn token_str(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    /// Get token ID by string.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }
}

/// Simple JSON vocabulary extractor.
fn extract_vocab_from_json(json: &str) -> Result<Vec<String>, String> {
    // Find "model" → "vocab" section
    if let Some(pos) = json.find("\"vocab\"") {
        if let Some(brace) = json[pos..].find('{') {
            let start = pos + brace + 1;
            // Find matching close brace
            let mut depth = 0;
            let mut in_str = false;
            let mut escaped = false;
            for (i, ch) in json[start..].char_indices() {
                if escaped {
                    escaped = false;
                    continue;
                }
                if ch == '\\' && in_str {
                    escaped = true;
                    continue;
                }
                if ch == '"' {
                    in_str = !in_str;
                    continue;
                }
                if in_str {
                    continue;
                }
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth < 0 {
                            let vocab_str = &json[start..start + i];
                            return Ok(parse_simple_vocab(vocab_str));
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    Err("Could not find vocab in JSON".into())
}

fn parse_simple_vocab(s: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for entry in s.split(',') {
        if let Some(colon) = entry.find(':') {
            let key = entry[..colon].trim().trim_matches('"');
            // Unescape the key
            let token = key
                .replace("\\u00", "\\x")
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\\"", "\"")
                .replace("\\\\", "\\");
            tokens.push(token);
        }
    }
    tokens
}

fn extract_special_tokens(json: &str) -> (u32, u32, u32) {
    let mut bos = 1;
    let mut eos = 2;
    let mut unk = 0;

    if let Some(pos) = json.find("\"added_tokens\"") {
        let section = &json[pos..];
        for obj in section.split('{') {
            if obj.contains("\"special\"") && obj.contains("\"id\"") {
                if let Some(id_pos) = obj.find("\"id\"") {
                    let id_str = &obj[id_pos + 4..];
                    if let Some(end) = id_str.find(|c: char| !c.is_ascii_digit()) {
                        if let Ok(id) = id_str[..end].parse::<u32>() {
                            if obj.contains("\"<s>\"") || obj.contains("\"<|begin|>") {
                                bos = id;
                            } else if obj.contains("\"</s>\"") || obj.contains("\"<|end|>") {
                                eos = id;
                            } else if obj.contains("\"<unk>\"") {
                                unk = id;
                            }
                        }
                    }
                }
            }
        }
    }

    (bos, eos, unk)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenize() {
        let vocab = vec![
            "hello".into(),
            "world".into(),
            " test".into(),
            "hello world".into(),
        ];
        let tokenizer = Tokenizer::from_vocab(vocab, 0, 1, 0);

        let tokens = tokenizer.encode("hello");
        assert_eq!(tokens, vec![0]);

        let text = tokenizer.decode(&[0]);
        assert_eq!(text, "hello");
    }

    #[test]
    fn test_decode_special() {
        let vocab = vec!["<s>".into(), "</s>".into(), "hello".into()];
        let tokenizer = Tokenizer::from_vocab(vocab, 0, 1, 0);

        // Special tokens should be skipped
        let text = tokenizer.decode(&[0, 1, 2]);
        assert_eq!(text, "hello");
    }
}
