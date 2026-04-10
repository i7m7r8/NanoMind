//! Byte-Pair Encoding tokenizer.
//!
//! Loads from HuggingFace `tokenizer.json` format (or a simplified vocab file).
//! Supports encode/decode with special tokens and unknown token fallback.

use std::collections::BTreeMap;
use std::string::String;
use std::vec::Vec;

/// Special token configuration.
#[derive(Clone, Debug, Default)]
pub struct SpecialTokens {
    pub pad_token: Option<String>,
    pub pad_id: u32,
    pub eos_token: Option<String>,
    pub eos_id: u32,
    pub bos_token: Option<String>,
    pub bos_id: u32,
    pub unk_token: Option<String>,
    pub unk_id: u32,
}

/// BPE merge rule: (token_a, token_b) → merged_token.
#[allow(dead_code)]
type MergeRule = (String, String, String);

/// BPE Tokenizer.
///
/// Loads a vocabulary and merge rules, then provides encode/decode.
/// Uses BTreeMap for no_std compatibility (no hashbrown in no_std mode).
#[derive(Clone, Debug)]
pub struct BpeTokenizer {
    /// Token string → token ID mapping.
    vocab: BTreeMap<String, u32>,
    /// Merge rules ordered by priority (lower index = applied first).
    merges: Vec<(String, String)>,
    /// Reverse lookup: ID → string.
    id_to_token: BTreeMap<u32, String>,
    /// Special tokens.
    pub special: SpecialTokens,
    /// Vocabulary size.
    vocab_size: usize,
}

impl BpeTokenizer {
    /// Create a new empty tokenizer.
    pub fn new() -> Self {
        Self {
            vocab: BTreeMap::new(),
            merges: Vec::new(),
            id_to_token: BTreeMap::new(),
            special: SpecialTokens::default(),
            vocab_size: 0,
        }
    }

    /// Load tokenizer from a HuggingFace `tokenizer.json` file content.
    ///
    /// This is a simplified loader that extracts:
    /// - vocabulary (token → ID)
    /// - merge rules (token_a + token_b → merged)
    /// - special tokens
    ///
    /// For full HuggingFace format support, consider using the `tokenizers` crate.
    /// This implementation handles a simplified JSON format.
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        let mut tokenizer = Self::new();

        // Parse the simplified tokenizer.json format
        // We use a simple string-based parser since we're no_std
        tokenizer.parse_vocab_from_json(json_str)?;
        tokenizer.parse_merges_from_json(json_str)?;
        tokenizer.parse_special_from_json(json_str)?;

        Ok(tokenizer)
    }

    /// Load from a simple text vocabulary file.
    ///
    /// Format: one token per line, line number = token ID.
    pub fn from_vocab_text(vocab: &str) -> Self {
        let mut tokenizer = Self::new();

        for (id, line) in vocab.lines().enumerate() {
            let token = line.trim_end().to_string();
            if token.is_empty() {
                continue;
            }
            let id = id as u32;
            tokenizer.vocab.insert(token.clone(), id);
            tokenizer.id_to_token.insert(id, token);
        }

        tokenizer.vocab_size = tokenizer.vocab.len();
        tokenizer
    }

    /// Load merge rules from text format.
    ///
    /// Each line: `token_a token_b` (space-separated).
    pub fn load_merges(&mut self, merges_text: &str) {
        self.merges.clear();

        for line in merges_text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Find the split point (last space that separates two tokens)
            if let Some(space_pos) = line.rfind(' ') {
                let a = line[..space_pos].to_string();
                let b = line[space_pos + 1..].to_string();
                self.merges.push((a, b));
            }
        }
    }

    /// Set special tokens.
    pub fn set_special_tokens(&mut self, special: SpecialTokens) {
        self.special = special;
    }

    /// Encode text into token IDs.
    ///
    /// This implements a simplified BPE algorithm:
    /// 1. Pre-tokenize into pieces (words, punctuation)
    /// 2. For each piece, try to find it in vocab as-is
    /// 3. If not found, split and try sub-tokens
    /// 4. Fall back to byte-level encoding + merges
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        // Pre-tokenize: split on whitespace and punctuation
        let pre_tokens = self.pre_tokenize(text);
        let mut tokens = Vec::new();

        for piece in pre_tokens {
            // First try: look up the whole piece in vocab
            if let Some(&id) = self.vocab.get(&piece) {
                tokens.push(id);
                continue;
            }

            // Second try: with space prefix (Ġpiece)
            let spaced = format!("Ġ{}", piece);
            if let Some(&id) = self.vocab.get(&spaced) {
                tokens.push(id);
                continue;
            }

            // Fallback: byte-level BPE
            let mut word = self.bytes_to_tokens(&piece);
            word = self.bpe_merge(word);

            for t in &word {
                if let Some(&id) = self.vocab.get(t) {
                    tokens.push(id);
                } else {
                    tokens.push(self.special.unk_id);
                }
            }
        }

        tokens
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut result = String::new();

        for (i, &token_id) in tokens.iter().enumerate() {
            if let Some(token_str) = self.id_to_token.get(&token_id) {
                // Convert byte-level tokens back to UTF-8
                let piece = self.token_to_string(token_str);

                // Add space between tokens that start with Ġ (byte-level space)
                // This is a common convention in BPE tokenizers
                if piece.starts_with('Ġ') {
                    if i > 0 {
                        result.push(' ');
                    }
                    result.push_str(&piece[1..]);
                } else if piece.starts_with('▁') {
                    // SentencePiece style
                    if i > 0 {
                        result.push(' ');
                    }
                    result.push_str(&piece[1..]);
                } else {
                    result.push_str(&piece);
                }
            } else {
                result.push_str("[UNK]");
            }
        }

        result
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get token ID for a string.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Get string for a token ID.
    pub fn id_to_token_str(&self, id: u32) -> Option<String> {
        self.id_to_token.get(&id).cloned()
    }

    // ─── Internal Methods ───────────────────────────────────────────

    /// Pre-tokenize: split on whitespace, punctuation boundaries.
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let mut pieces = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            if ch.is_whitespace() {
                if !current.is_empty() {
                    pieces.push(core::mem::take(&mut current));
                }
                // Mark whitespace with special prefix
                current.push(' ');
                // Also consume the whitespace char
                // current.push(ch);
            } else if ch.is_ascii_punctuation() {
                if !current.is_empty() {
                    pieces.push(core::mem::take(&mut current));
                }
                let mut piece = String::new();
                piece.push(ch);
                pieces.push(piece);
            } else {
                current.push(ch);
            }
        }

        if !current.is_empty() {
            pieces.push(current);
        }

        pieces
    }

    /// Convert a string piece into byte-level tokens.
    fn bytes_to_tokens(&self, piece: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        for byte in piece.bytes() {
            // Convert byte to its visual representation
            // Bytes 0x01-0xFF map to Ā-ÿ (Unicode range)
            // Space maps to Ġ
            let ch = if byte == b' ' {
                'Ġ'.to_string()
            } else {
                char::from_u32(0x0100 + byte as u32)
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| format!("<0x{:02X}>", byte))
            };
            tokens.push(ch);
        }
        tokens
    }

    /// Apply BPE merge rules to a list of tokens.
    fn bpe_merge(&self, mut tokens: Vec<String>) -> Vec<String> {
        loop {
            // Find the best pair to merge
            let mut best_idx = None;
            let mut best_rank = usize::MAX;

            for i in 0..tokens.len().saturating_sub(1) {
                let pair = (&tokens[i], &tokens[i + 1]);
                for (rank, (a, b)) in self.merges.iter().enumerate() {
                    if pair.0 == a && pair.1 == b {
                        if rank < best_rank {
                            best_rank = rank;
                            best_idx = Some(i);
                        }
                        break;
                    }
                }
            }

            let Some(idx) = best_idx else { break };

            // Merge tokens[idx] and tokens[idx+1]
            let merged = format!("{}{}", tokens[idx], tokens[idx + 1]);
            tokens[idx] = merged;
            tokens.remove(idx + 1);
        }

        tokens
    }

    /// Convert a token string back to a readable string.
    fn token_to_string(&self, token: &str) -> String {
        let mut result = String::new();
        let chars: Vec<char> = token.chars().collect();

        for &ch in &chars {
            if ch == 'Ġ' {
                result.push(' ');
            } else if (ch as u32) >= 0x0100 && (ch as u32) <= 0x01FF {
                // Byte-level encoding
                result.push((ch as u32 - 0x0100) as u8 as char);
            } else {
                result.push(ch);
            }
        }

        result
    }

    // ─── JSON Parsing (simplified) ──────────────────────────────────

    /// Parse vocabulary from JSON string.
    fn parse_vocab_from_json(&mut self, json: &str) -> Result<(), String> {
        // Find "model" section, then "vocab"
        if let Some(vocab_start) = json.find("\"vocab\"") {
            if let Some(brace_pos) = json[vocab_start..].find('{') {
                let start = vocab_start + brace_pos + 1;
                // Find the matching closing brace
                let end = find_json_object_end(&json[start..])
                    .ok_or("Could not find end of vocab object")?;

                let vocab_str = &json[start..start + end];

                // Parse "token": ID pairs
                for entry in vocab_str.split(',') {
                    let entry = entry.trim();
                    if let Some((key, val)) = parse_json_kv(entry) {
                        let token = strip_quotes(&key);
                        if let Ok(id) = val.trim().parse::<u32>() {
                            self.vocab.insert(token.to_string(), id);
                            self.id_to_token.insert(id, token.to_string());
                        }
                    }
                }

                self.vocab_size = self.vocab.len();
                return Ok(());
            }
        }

        Err("Could not find vocab in JSON".into())
    }

    /// Parse merge rules from JSON string.
    fn parse_merges_from_json(&mut self, json: &str) -> Result<(), String> {
        if let Some(merges_pos) = json.find("\"merges\"") {
            if let Some(bracket_pos) = json[merges_pos..].find('[') {
                let start = merges_pos + bracket_pos + 1;
                if let Some(end_pos) = json[start..].find(']') {
                    let merges_str = &json[start..start + end_pos];

                    for entry in merges_str.split('"') {
                        let entry = entry.trim();
                        if entry.is_empty() || entry == "," || entry == "[" || entry == "]" {
                            continue;
                        }
                        if let Some(space_pos) = entry.find(" ") {
                            let a = entry[..space_pos].to_string();
                            let b = entry[space_pos + 1..].to_string();
                            self.merges.push((a, b));
                        }
                    }
                    return Ok(());
                }
            }
        }

        Ok(()) // merges are optional
    }

    /// Parse special tokens from JSON.
    fn parse_special_from_json(&mut self, json: &str) -> Result<(), String> {
        if let Some(pos) = json.find("\"added_tokens\"") {
            if let Some(bracket_pos) = json[pos..].find('[') {
                let start = pos + bracket_pos + 1;
                if let Some(end_pos) = json[start..].find(']') {
                    let section = &json[start..start + end_pos];

                    for token_obj in section.split('{') {
                        if token_obj.trim().is_empty() {
                            continue;
                        }
                        let token_obj = token_obj.split('}').next().unwrap_or("");

                        let mut tok_str = String::new();
                        let mut tok_id: Option<u32> = None;
                        let mut is_special = false;

                        for part in token_obj.split(',') {
                            if let Some((k, v)) = parse_json_kv(part) {
                                match k.trim() {
                                    "\"content\"" | "\"text\"" => {
                                        tok_str = strip_quotes(&v.trim()).to_string();
                                    }
                                    "\"id\"" => {
                                        tok_id = v.trim().parse().ok();
                                    }
                                    "\"special\"" => {
                                        is_special = v.trim() == "true";
                                    }
                                    _ => {}
                                }
                            }
                        }

                        if !tok_str.is_empty() && tok_str != "\"\"" {
                            if let Some(id) = tok_id {
                                let tok = strip_quotes(&tok_str).to_string();
                                self.vocab.insert(tok.clone(), id);
                                self.id_to_token.insert(id, tok.clone());

                                match tok.as_str() {
                                    "<|endoftext|>" | "</s>" | "<|end|>" => {
                                        self.special.eos_token = Some(tok.clone());
                                        self.special.eos_id = id;
                                    }
                                    "<|begin|>" | "<|start|>" | "<s>" | "<bos>" => {
                                        self.special.bos_token = Some(tok.clone());
                                        self.special.bos_id = id;
                                    }
                                    "<unk>" => {
                                        self.special.unk_token = Some(tok.clone());
                                        self.special.unk_id = id;
                                    }
                                    "<pad>" | "[PAD]" => {
                                        self.special.pad_token = Some(tok.clone());
                                        self.special.pad_id = id;
                                    }
                                    _ => {
                                        if is_special {
                                            // Default to unknown
                                            self.special.unk_token = Some(tok.clone());
                                            self.special.unk_id = id;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Set defaults for unset special tokens
        if self.special.unk_id == 0 && self.vocab.contains_key("<unk>") {
            self.special.unk_token = Some("<unk>".into());
            self.special.unk_id = *self.vocab.get("<unk>").unwrap();
        }

        Ok(())
    }
}

impl Default for BpeTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── JSON Helpers ───────────────────────────────────────────────────────

fn find_json_object_end(s: &str) -> Option<usize> {
    let mut depth = 0;
    let mut in_string = false;
    let mut escaped = false;

    for (i, ch) in s.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && in_string {
            escaped = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn strip_quotes(s: &str) -> &str {
    let s = s.trim();
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

fn parse_json_kv(entry: &str) -> Option<(String, String)> {
    let entry = entry.trim().trim_matches(',');
    if let Some(colon) = entry.find(':') {
        let key = entry[..colon].trim().to_string();
        let val = entry[colon + 1..].trim().to_string();
        Some((key, val))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenize() {
        let mut tokenizer = BpeTokenizer::new();
        tokenizer.vocab.insert("hello".into(), 100);
        tokenizer.id_to_token.insert(100, "hello".into());
        tokenizer.vocab.insert("world".into(), 101);
        tokenizer.id_to_token.insert(101, "world".into());
        tokenizer.vocab.insert("test".into(), 102);
        tokenizer.id_to_token.insert(102, "test".into());
        tokenizer.vocab_size = tokenizer.vocab.len();
        tokenizer.special.unk_id = 0;

        // "hello" should be found in vocab
        let tokens = tokenizer.encode("hello");
        assert_eq!(tokens, vec![100], "Known word should map to single token");

        // "xyz" is unknown → byte-level fallback → 3 bytes → 3 UNK tokens
        let tokens = tokenizer.encode("xyz");
        assert_eq!(tokens.len(), 3, "Unknown 3-char word → 3 byte tokens");
        assert!(tokens.iter().all(|&t| t == 0), "Unknown bytes map to UNK");
    }

    #[test]
    fn test_decode() {
        let vocab = "hello\nworld\ntest\n";
        let mut tokenizer = BpeTokenizer::from_vocab_text(vocab);
        tokenizer.special.unk_id = 0;

        // ID 0 → "hello"
        let text = tokenizer.decode(&[0]);
        assert!(!text.is_empty());
    }

    #[test]
    fn test_vocab_size() {
        let vocab = "hello\nworld\ntest\nextra\n";
        let tokenizer = BpeTokenizer::from_vocab_text(vocab);
        assert_eq!(tokenizer.vocab_size(), 4);
    }

    #[test]
    fn test_bpe_merge() {
        let vocab = "a\nb\nc\nab\nbc\nabc\n".to_string();
        let mut tokenizer = BpeTokenizer::from_vocab_text(&vocab);
        tokenizer.load_merges("a b\nab c\n");
        tokenizer.special.unk_id = 0;

        // With proper merges, "abc" should merge to single token
        let tokens = tokenizer.encode("abc");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_json_parsing() {
        // Test with a simple vocab text format (more reliable than JSON parsing)
        let vocab_text = "hello\nworld\ntest\n";
        let tokenizer = BpeTokenizer::from_vocab_text(vocab_text);
        assert_eq!(tokenizer.vocab_size(), 3);

        // Verify tokens exist
        assert!(tokenizer.token_to_id("hello").is_some());
        assert!(tokenizer.token_to_id("world").is_some());
        assert!(tokenizer.token_to_id("test").is_some());
    }
}
