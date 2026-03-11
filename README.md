# 🤖 PyBot — NLP Chatbot from Scratch

A **hands-on NLP chatbot** built entirely in pure Python — no NLTK, no scikit-learn,
no TensorFlow. Every algorithm is implemented from scratch so you can see *exactly*
how it works.

```
User: "What is NLP?"
  ↓  Tokenize → ["what", "is", "nlp"]
  ↓  Stop Word Removal → ["nlp"]
  ↓  Stem → ["nlp"]
  ↓  TF-IDF Vectorize → [0.0, 0.82, 0.0, 0.45, ...]
  ↓  Cosine Similarity → best match: intent "nlp" (score: 0.87)
Bot: "NLP (Natural Language Processing) is AI that enables computers..."
```

---

## 📁 Project Structure

```
nlp_chatbot/
├── intents.json      ← Training data (patterns + responses per intent)
├── nlp_utils.py      ← All NLP algorithms from scratch
│     ├── tokenize()           Lowercasing + contraction expansion + regex split
│     ├── PorterStemmer        Simplified Porter stemming algorithm
│     ├── preprocess()         Full pipeline: tokenize → stopwords → stem
│     ├── TFIDFVectorizer      TF-IDF from scratch (fit/transform/query)
│     ├── cosine_similarity()  Vector dot product / magnitude
│     ├── find_best_match()    Argmax over cosine scores
│     └── SimpleNER            Rule-based entity recognition (email, URL, date...)
│
├── chatbot.py        ← Core chatbot engine (intent classification + responses)
├── app.py            ← HTTP server (no Flask, uses built-in http.server)
├── index.html        ← Chat UI (dark theme, debug panel, intent scores)
├── demo.py           ← Interactive terminal chat
└── README.md         ← You're reading it
```

---

## 🚀 Quick Start

### Option 1 — Terminal Chat (fastest)
```bash
cd nlp_chatbot
python demo.py

# With NLP debug info on every message:
python demo.py --debug
```

### Option 2 — Web UI
```bash
cd nlp_chatbot
python app.py
# Open http://localhost:8080 in your browser
```

### Option 3 — Use as a Python module
```python
from chatbot import NLPChatbot

bot = NLPChatbot("intents.json")
result = bot.respond("What is NLP?")
print(result["response"])     # The chatbot's reply
print(result["tag"])          # Detected intent: "nlp"
print(result["confidence"])   # Cosine similarity score: 0.87
print(result["entities"])     # Any detected entities (email, URL, etc.)
```

---

## 🧠 How the NLP Pipeline Works

### 1. Tokenization
```python
tokenize("I'm learning NLP! It's great.")
# → ['i', 'am', 'learning', 'nlp', 'it', 'is', 'great']
```
Steps: lowercase → expand contractions → remove punctuation → split on whitespace

### 2. Stop Word Removal
Common words like "the", "is", "a", "and" are removed — they don't carry meaning.
```python
preprocess("what is natural language processing")
# Stop words removed: ['natural', 'language', 'process']
```

### 3. Stemming (Porter Algorithm)
Reduces words to their root form so variants match each other.
```python
stem("running")   → "run"
stem("studies")   → "studi"
stem("happiness") → "happi"
stem("tokenization") → "token"
```

### 4. TF-IDF Vectorization
Converts text into numerical vectors capturing word importance.

```
TF(word, doc)  = count(word in doc) / total_words(doc)
IDF(word)      = log(N / (1 + df(word))) + 1

TF-IDF = TF × IDF
```

Words rare across documents but frequent in one document get high scores.

### 5. Cosine Similarity
Measures the angle between two TF-IDF vectors:

```
similarity = (A · B) / (||A|| × ||B||)
```

Returns 0.0 (totally different) to 1.0 (identical meaning).

### 6. Intent Classification
```
query_vector = vectorize(user_input)
scores = [cosine_similarity(query_vector, pattern_vector) for each training pattern]
best_intent = intent_tag[argmax(scores)]
```
If the top score is below threshold (0.15), falls back gracefully.

### 7. Named Entity Recognition (Rule-based)
```python
ner.extract("Email me at alice@example.com on 01/15/2024")
# → {"EMAIL": ["alice@example.com"], "DATE": ["01/15/2024"]}
```

---

## ✏️ Adding New Intents

Edit `intents.json` — add a new intent block:

```json
{
  "tag": "my_new_intent",
  "patterns": [
    "example pattern one",
    "another way to say it",
    "third variation"
  ],
  "responses": [
    "First possible response.",
    "Second possible response (randomly chosen)."
  ]
}
```

The chatbot re-trains automatically on startup. More patterns = better accuracy!

---

## 🔌 REST API Reference (when running app.py)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/`      | Chat UI (index.html) |
| POST   | `/chat`  | Send a message |
| GET    | `/stats` | Session statistics |
| POST   | `/reset` | Reset conversation |

**POST /chat** — Request body:
```json
{ "message": "What is NLP?", "debug": true }
```

**Response:**
```json
{
  "response": "NLP (Natural Language Processing) is AI...",
  "tag": "nlp",
  "confidence": 0.8721,
  "entities": {},
  "time_ms": 1.2,
  "debug": {
    "tokens": ["what", "is", "nlp"],
    "stemmed_tokens": ["nlp"],
    "preprocessed": "nlp",
    "best_pattern": "what is nlp",
    "top_intents": [["nlp", 0.8721], ["help", 0.1203], ...]
  }
}
```

---

## 🧪 Test the NLP Utils Directly

```bash
python nlp_utils.py   # runs built-in tests
python chatbot.py     # runs a quick test conversation
```

---

## 📊 Supported Intents

| Tag | Example Patterns |
|-----|-----------------|
| `greeting` | "hello", "hi there", "good morning" |
| `goodbye` | "bye", "see you", "farewell" |
| `thanks` | "thank you", "much appreciated" |
| `name` | "what is your name?", "who are you?" |
| `help` | "what can you do?", "help me" |
| `joke` | "tell me a joke", "make me laugh" |
| `nlp` | "what is NLP?", "explain natural language processing" |
| `python` | "tell me about Python", "python for AI" |
| `machine_learning` | "what is machine learning?", "types of ML" |
| `sentiment` | "what is sentiment analysis?", "opinion mining" |
| `tokenization` | "what is tokenization?", "what are tokens?" |
| `tfidf` | "explain TF-IDF", "what does TF-IDF measure?" |
| `how_built` | "how do you work?", "what algorithm do you use?" |
| `weather` | "what's the weather?", "will it rain?" |
| `fallback` | *(catches anything unrecognized)* |

---

## 🔧 Extending This Project

**Level 1 — More intents:** Add to `intents.json`

**Level 2 — Better NLP:** Replace `TFIDFVectorizer` with scikit-learn's version:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

**Level 3 — Neural approach:** Replace cosine similarity with a neural classifier:
```python
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
```

**Level 4 — Transformers:** Use sentence embeddings for semantic similarity:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

**Level 5 — LLM-powered:** Use the Anthropic API or OpenAI for open-domain responses.

---

## 📚 Key Concepts Demonstrated

- ✅ Text tokenization and normalization  
- ✅ Stop word filtering  
- ✅ Rule-based stemming (Porter algorithm)  
- ✅ TF-IDF vectorization from first principles  
- ✅ Cosine similarity for text matching  
- ✅ Intent classification  
- ✅ Rule-based Named Entity Recognition  
- ✅ Context tracking across turns  
- ✅ REST API design (no framework needed)  
- ✅ Building a chat UI from scratch  
