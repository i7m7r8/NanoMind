//! Data loader for training — reads token IDs from binary file.

/// Data loader that reads pre-tokenized u32 token IDs from a binary file.
///
/// The binary file format is simply a flat array of u32 token IDs (little-endian).
/// Each batch returns `batch_size` sequences of length `seq_len`.
pub struct DataLoader {
    tokens: Vec<u32>,
    seq_len: usize,
    batch_size: usize,
    pos: usize,
}

impl DataLoader {
    /// Create a new data loader from a slice of token IDs.
    pub fn new(tokens: Vec<u32>, seq_len: usize, batch_size: usize) -> Self {
        Self {
            tokens,
            seq_len,
            batch_size,
            pos: 0,
        }
    }

    /// Load token IDs from a raw binary file (u32 LE).
    pub fn from_file(path: &std::path::Path) -> std::io::Result<Vec<u32>> {
        let data = std::fs::read(path)?;
        if data.len() % 4 != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Token file size not aligned to u32",
            ));
        }
        let count = data.len() / 4;
        let mut tokens = Vec::with_capacity(count);
        for chunk in data.chunks_exact(4) {
            tokens.push(u32::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(tokens)
    }

    /// Generate training data from a text string using a simple byte-level tokenizer.
    /// This is used when no pre-tokenized file exists — tokenizes on the fly.
    pub fn from_text(text: &str, vocab_size: usize, seq_len: usize) -> (Self, ByteTokenizer) {
        let tokenizer = ByteTokenizer::new(vocab_size);
        let tokens = tokenizer.encode(text);
        let loader = Self::new(tokens, seq_len, 1);
        (loader, tokenizer)
    }

    /// Get the next training batch.
    /// Returns (input_sequences, target_sequences), each [batch_size, seq_len].
    /// Targets are inputs shifted left by 1 (next token prediction).
    pub fn next_batch(&mut self) -> Option<(Vec<Vec<u32>>, Vec<Vec<u32>>)> {
        let total_needed = self.batch_size * (self.seq_len + 1);
        if self.pos + total_needed > self.tokens.len() {
            // Wrap around
            self.pos = 0;
        }

        if self.tokens.len() < total_needed {
            return None;
        }

        let mut inputs = Vec::with_capacity(self.batch_size);
        let mut targets = Vec::with_capacity(self.batch_size);

        for _b in 0..self.batch_size {
            let start = self.pos;
            let input = self.tokens[start..start + self.seq_len].to_vec();
            let target = self.tokens[start + 1..start + self.seq_len + 1].to_vec();
            inputs.push(input);
            targets.push(target);
            self.pos += self.seq_len;
        }

        Some((inputs, targets))
    }

    /// Reset to beginning of data
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Total number of tokens in the dataset
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Simple byte-level tokenizer for training data preparation.
/// Maps each byte to a token ID (0-255) plus special tokens.
pub struct ByteTokenizer {
    vocab_size: usize,
}

impl ByteTokenizer {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size: vocab_size.max(256 + 4), // at least bytes + specials
        }
    }

    /// Encode text to token IDs.
    /// Token 0=BOS, 1=EOS, 2=PAD, 3=UNK, 4..=259=bytes
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = vec![0u32]; // BOS
        for &b in text.as_bytes() {
            tokens.push(b as u32 + 4);
        }
        tokens.push(1); // EOS
        tokens
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &t in tokens {
            if t >= 4 && t <= 259 {
                bytes.push((t - 4) as u8);
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    pub fn bos_id(&self) -> u32 {
        0
    }
    pub fn eos_id(&self) -> u32 {
        1
    }
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// Generate training data from text (built-in corpus for from-scratch training).
/// Returns the text content to tokenize.
pub fn get_training_corpus() -> String {
    // Built-in simple English corpus for training
    // This provides basic language knowledge without external data
    let sentences = [
        "the cat sat on the mat.",
        "the dog ran in the park.",
        "a bird flew over the tree.",
        "the sun is bright today.",
        "the fish swam in the pond.",
        "i like to read books.",
        "she went to the store.",
        "the children played outside.",
        "he ate an apple for lunch.",
        "the rain fell on the ground.",
        "the moon shines at night.",
        "we walked along the beach.",
        "the flowers bloom in spring.",
        "the wind blew through the trees.",
        "the baby slept peacefully.",
        "the teacher wrote on the board.",
        "the car drove down the road.",
        "the cow grazed in the field.",
        "the snow covered the mountain.",
        "the river flows to the sea.",
        "the stars twinkle in the sky.",
        "the boy kicked the ball.",
        "the girl sang a song.",
        "the clock ticked on the wall.",
        "the phone rang in the room.",
        "the coffee was hot and fresh.",
        "the book was very interesting.",
        "the door opened slowly.",
        "the light turned green.",
        "the train arrived on time.",
    ];

    // Repeat to get enough data for training
    let mut corpus = String::new();
    for _ in 0..10 {
        for s in &sentences {
            corpus.push_str(s);
            corpus.push('\n');
        }
    }
    corpus
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_loader() {
        let tokens: Vec<u32> = (0..100).collect();
        let mut loader = DataLoader::new(tokens, 10, 2);

        let (inputs, targets) = loader.next_batch().unwrap();
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].len(), 10);
        assert_eq!(targets[0].len(), 10);

        // Targets should be inputs shifted by 1
        assert_eq!(targets[0][0], inputs[0][1]);
    }

    #[test]
    fn test_byte_tokenizer() {
        let tokenizer = ByteTokenizer::new(260);
        let text = "hello world";
        let tokens = tokenizer.encode(text);

        assert_eq!(tokens[0], 0); // BOS
        assert_eq!(tokens[tokens.len() - 1], 1); // EOS
        assert_eq!(tokens.len(), text.len() + 2);

        let decoded = tokenizer.decode(&tokens);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_training_corpus() {
        let corpus = get_training_corpus();
        assert!(!corpus.is_empty());
        assert!(corpus.len() > 100);
    }

    #[test]
    fn test_from_text() {
        let text = "hello world test";
        let (mut loader, tokenizer) = DataLoader::from_text(text, 260, 5);
        assert!(!loader.is_empty());

        let (inputs, targets) = loader.next_batch().unwrap();
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].len(), 5);

        let decoded = tokenizer.decode(&inputs[0]);
        assert!(!decoded.is_empty());
    }
}
