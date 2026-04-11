#!/bin/bash
# Download training data for NanoMind
# All sources are public domain (no copyright restrictions)

set -e

mkdir -p data/raw

echo "=== Downloading Training Data ==="

# 1. Alice in Wonderland (Project Gutenberg, public domain)
echo "[1/4] Alice in Wonderland..."
curl -sL "https://www.gutenberg.org/cache/epub/11/pg11.txt" \
    -o data/raw/alice.txt

# 2. Shakespeare Complete Works (Project Gutenberg, public domain)
echo "[2/4] Shakespeare Complete Works..."
curl -sL "https://www.gutenberg.org/cache/epub/100/pg100.txt" \
    -o data/raw/shakespeare.txt

# 3. The Art of War by Sun Tzu (public domain)
echo "[3/4] The Art of War..."
curl -sL "https://www.gutenberg.org/cache/epub/132/pg132.txt" \
    -o data/raw/artofwar.txt

# 4. Aesop's Fables (public domain)
echo "[4/4] Aesop's Fables..."
curl -sL "https://www.gutenberg.org/cache/epub/19528/pg19528.txt" \
    -o data/raw/aesop.txt

# Combine all texts
echo "Combining..."
cat data/raw/*.txt > data/corpus.txt

# Show stats
echo ""
echo "=== Training Data Stats ==="
echo "Files: $(ls data/raw/*.txt | wc -l)"
echo "Total size: $(du -h data/corpus.txt | cut -f1)"
echo "Characters: $(wc -c < data/corpus.txt)"
echo "Words: $(wc -w < data/corpus.txt)"
echo "Lines: $(wc -l < data/corpus.txt)"
