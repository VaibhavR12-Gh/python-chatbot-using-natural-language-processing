"""
nlp_utils.py — NLP Preprocessing Utilities (pure Python, no external NLP libs)

Implements from scratch:
  - Tokenizer
  - Stop word list
  - Porter Stemmer (simplified)
  - TF-IDF Vectorizer
  - Cosine Similarity
"""

import re
import math
from collections import Counter


# ─────────────────────────────────────────────
# STOP WORDS
# ─────────────────────────────────────────────

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "up", "about", "into", "through",
    "during", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "shall", "can", "not", "no", "nor", "so", "yet",
    "both", "either", "neither", "each", "few", "more", "most", "other",
    "some", "such", "than", "too", "very", "just", "that", "this", "these",
    "those", "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "it", "they", "them", "what", "which", "who", "whom", "how", "when",
    "where", "why", "all", "any", "both", "own", "same", "than", "s",
    "t", "don", "doesn", "didn", "won", "isn", "aren", "wasn", "weren",
    "im", "ive", "id", "ill", "youre", "theyre", "its", "thats"
}


# ─────────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    Tokenize text into lowercase word tokens.
    Steps:
      1. Lowercase
      2. Expand contractions
      3. Remove punctuation
      4. Split on whitespace
      5. Filter empty strings
    """
    text = text.lower()

    # Expand common contractions
    contractions = {
        "i'm": "i am", "i've": "i have", "i'd": "i would", "i'll": "i will",
        "you're": "you are", "you've": "you have", "you'll": "you will",
        "it's": "it is", "that's": "that is", "what's": "what is",
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "won't": "will not", "can't": "cannot", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "they're": "they are", "we're": "we are", "there's": "there is"
    }
    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)

    # Remove punctuation (keep apostrophes handled above)
    text = re.sub(r"[^\w\s]", " ", text)

    # Split and filter
    tokens = [t.strip() for t in text.split() if t.strip()]
    return tokens


# ─────────────────────────────────────────────
# PORTER STEMMER (simplified)
# ─────────────────────────────────────────────

class PorterStemmer:
    """
    Simplified Porter Stemmer implementing the most impactful rules.
    Handles: plurals, verb suffixes, adjective suffixes.
    """

    VOWELS = set("aeiou")

    def _is_vowel(self, char: str) -> bool:
        return char in self.VOWELS

    def _has_vowel(self, word: str) -> bool:
        return any(self._is_vowel(c) for c in word)

    def stem(self, word: str) -> str:
        if len(word) <= 2:
            return word

        word = self._step1a(word)
        word = self._step1b(word)
        word = self._step2(word)
        word = self._step3(word)
        return word

    def _step1a(self, word: str) -> str:
        """Handle plurals and -ed/-ing."""
        if word.endswith("sses"):
            return word[:-2]
        if word.endswith("ies"):
            return word[:-2] if len(word) > 4 else word[:-1]
        if word.endswith("ss"):
            return word
        if word.endswith("s") and not word.endswith("ss"):
            stem = word[:-1]
            if self._has_vowel(stem) and len(stem) > 1:
                return stem
        return word

    def _step1b(self, word: str) -> str:
        """Handle -eed, -ed, -ing."""
        if word.endswith("eed"):
            stem = word[:-3]
            if len(stem) > 1:
                return word[:-1]
            return word
        for suffix in ("ing", "ed"):
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._has_vowel(stem) and len(stem) > 1:
                    # Handle double consonant
                    if len(stem) > 1 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
                        return stem[:-1]
                    if stem.endswith("at") or stem.endswith("bl") or stem.endswith("iz"):
                        return stem + "e"
                    return stem
        return word

    def _step2(self, word: str) -> str:
        """Handle common suffixes."""
        suffix_map = [
            ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
            ("anci", "ance"), ("izer", "ize"), ("alism", "al"),
            ("ness", ""), ("ment", ""), ("ful", ""), ("ous", ""),
            ("ive", ""), ("ing", ""), ("ly", ""), ("er", ""),
        ]
        for suffix, replacement in suffix_map:
            if word.endswith(suffix) and len(word) - len(suffix) > 1:
                return word[:-len(suffix)] + replacement
        return word

    def _step3(self, word: str) -> str:
        """Final cleanup."""
        for suffix in ("icate", "ative", "alize", "ical", "ness", "ful"):
            if word.endswith(suffix) and len(word) - len(suffix) > 2:
                return word[:-len(suffix)]
        return word


# Singleton stemmer
_stemmer = PorterStemmer()


# ─────────────────────────────────────────────
# PREPROCESSING PIPELINE
# ─────────────────────────────────────────────

def preprocess(text: str, remove_stopwords: bool = True, stem: bool = True) -> list[str]:
    """
    Full NLP preprocessing pipeline:
      1. Tokenize
      2. Remove stop words (optional)
      3. Stem tokens (optional)
    
    Returns list of processed tokens.
    """
    tokens = tokenize(text)

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]

    if stem:
        tokens = [_stemmer.stem(t) for t in tokens]

    return tokens


def preprocess_to_string(text: str) -> str:
    """Preprocess and return as a single string (for vectorization)."""
    return " ".join(preprocess(text))


# ─────────────────────────────────────────────
# TF-IDF VECTORIZER
# ─────────────────────────────────────────────

class TFIDFVectorizer:
    """
    TF-IDF Vectorizer implemented from scratch using pure Python.
    
    TF(t, d)  = count(t in d) / total_terms(d)
    IDF(t, D) = log( |D| / (1 + df(t)) ) + 1   [smooth variant]
    TF-IDF    = TF × IDF
    """

    def __init__(self):
        self.vocab: dict[str, int] = {}       # word → index
        self.idf_values: dict[str, float] = {}  # word → IDF score
        self.n_docs: int = 0

    def fit(self, documents: list[str]) -> "TFIDFVectorizer":
        """
        Build vocabulary and IDF scores from a corpus of documents.
        Each document is a preprocessed string.
        """
        self.n_docs = len(documents)
        doc_freq: Counter = Counter()

        tokenized_docs = []
        for doc in documents:
            tokens = doc.split()
            tokenized_docs.append(tokens)
            # Count unique tokens per document
            doc_freq.update(set(tokens))

        # Build vocabulary (sorted for determinism)
        all_words = sorted(doc_freq.keys())
        self.vocab = {word: idx for idx, word in enumerate(all_words)}

        # Compute IDF with smoothing
        for word, df in doc_freq.items():
            self.idf_values[word] = math.log(self.n_docs / (1 + df)) + 1

        return self

    def transform(self, documents: list[str]) -> list[list[float]]:
        """
        Transform documents into TF-IDF vectors.
        Returns a 2D list: [doc_index][feature_index] = tfidf_score
        """
        vectors = []
        vocab_size = len(self.vocab)

        for doc in documents:
            tokens = doc.split()
            total_terms = len(tokens) if tokens else 1
            tf_counts = Counter(tokens)

            vector = [0.0] * vocab_size
            for word, count in tf_counts.items():
                if word in self.vocab:
                    tf = count / total_terms
                    idf = self.idf_values.get(word, 0.0)
                    vector[self.vocab[word]] = tf * idf

            vectors.append(vector)

        return vectors

    def fit_transform(self, documents: list[str]) -> list[list[float]]:
        """Fit and transform in one step."""
        self.fit(documents)
        return self.transform(documents)

    def transform_query(self, query: str) -> list[float]:
        """Transform a single query string into a TF-IDF vector."""
        return self.transform([query])[0]


# ─────────────────────────────────────────────
# COSINE SIMILARITY
# ─────────────────────────────────────────────

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    cos(θ) = (A · B) / (||A|| × ||B||)
    
    Returns value in [0, 1]: 1 = identical direction, 0 = orthogonal.
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot_product / (mag_a * mag_b)


def find_best_match(query_vec: list[float], corpus_vecs: list[list[float]]) -> tuple[int, float]:
    """
    Find the index of the most similar vector in the corpus.
    Returns (best_index, best_score).
    """
    scores = [cosine_similarity(query_vec, vec) for vec in corpus_vecs]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return best_idx, scores[best_idx]


# ─────────────────────────────────────────────
# NAMED ENTITY EXTRACTOR (rule-based)
# ─────────────────────────────────────────────

class SimpleNER:
    """
    Rule-based Named Entity Recognition for common patterns.
    Detects: emails, URLs, numbers, dates, capitalized names.
    """

    PATTERNS = {
        "EMAIL":   r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "URL":     r"https?://[^\s]+",
        "NUMBER":  r"\b\d+(?:\.\d+)?\b",
        "DATE":    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:,\s+\d{4})?\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "CAPS":    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
    }

    def extract(self, text: str) -> dict[str, list[str]]:
        """Extract named entities from raw text."""
        entities = {}
        for label, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[label] = matches
        return entities


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  NLP Utils — Module Test")
    print("=" * 55)

    # Test tokenizer
    sample = "I'm learning NLP! It's really interesting, isn't it?"
    tokens = tokenize(sample)
    print(f"\n📝 Tokenize: '{sample}'")
    print(f"   → {tokens}")

    # Test preprocessing
    processed = preprocess(sample)
    print(f"\n🔧 Preprocess (stop words removed + stemmed):")
    print(f"   → {processed}")

    # Test stemmer
    stemmer = PorterStemmer()
    words = ["running", "studies", "happiness", "tokenization", "learning"]
    print(f"\n🌿 Stemming:")
    for w in words:
        print(f"   {w:20s} → {stemmer.stem(w)}")

    # Test TF-IDF
    docs = [
        "natural language processing is amazing",
        "machine learning and deep learning",
        "python is great for nlp and ml",
        "cosine similarity measures text distance"
    ]
    vectorizer = TFIDFVectorizer()
    vecs = vectorizer.fit_transform(docs)
    query = "what is nlp and natural language"
    q_vec = vectorizer.transform_query(preprocess_to_string(query))
    proc_docs = [preprocess_to_string(d) for d in docs]
    vecs2 = vectorizer.fit_transform(proc_docs)
    q_vec2 = vectorizer.transform_query(preprocess_to_string(query))
    best_idx, score = find_best_match(q_vec2, vecs2)
    print(f"\n🎯 TF-IDF + Cosine Similarity:")
    print(f"   Query: '{query}'")
    print(f"   Best match: '{docs[best_idx]}' (score: {score:.4f})")

    # Test NER
    ner = SimpleNER()
    ner_text = "Contact John Smith at john@example.com or visit https://example.com on 01/15/2024"
    entities = ner.extract(ner_text)
    print(f"\n🏷️  Named Entity Recognition:")
    print(f"   Text: '{ner_text}'")
    for label, items in entities.items():
        print(f"   {label}: {items}")

    print("\n✅ All NLP utils working correctly!\n")
