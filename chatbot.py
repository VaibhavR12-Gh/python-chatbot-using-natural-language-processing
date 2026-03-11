"""
chatbot.py — Core NLP Chatbot Engine

Architecture:
  1. Load intents from intents.json
  2. Preprocess all training patterns
  3. Build TF-IDF vectors for each pattern
  4. At inference: vectorize user input → cosine similarity → pick intent → random response

Features:
  - Intent classification via TF-IDF + Cosine Similarity
  - Confidence thresholding (falls back gracefully)
  - Context tracking (remembers last N turns)
  - Named Entity Recognition on input
  - Conversation history
  - Detailed debug/explain mode
"""

import json
import random
import time
from pathlib import Path

from nlp_utils import (
    TFIDFVectorizer, preprocess_to_string, find_best_match,
    tokenize, preprocess, SimpleNER
)


# ─────────────────────────────────────────────
# CHATBOT ENGINE
# ─────────────────────────────────────────────

class NLPChatbot:
    """
    NLP-powered chatbot using TF-IDF + Cosine Similarity for intent matching.
    
    Usage:
        bot = NLPChatbot("intents.json")
        response = bot.respond("Hello there!")
        print(response)
    """

    CONFIDENCE_THRESHOLD = 0.15   # Minimum score to accept intent
    CONTEXT_WINDOW = 5            # Number of past turns to remember

    def __init__(self, intents_path: str = "intents.json"):
        self.intents_path = Path(intents_path)
        self.vectorizer = TFIDFVectorizer()
        self.ner = SimpleNER()

        # Loaded data
        self.intents: list[dict] = []
        self.patterns: list[str] = []          # raw patterns
        self.pattern_tags: list[str] = []      # corresponding intent tag
        self.pattern_vectors: list[list[float]] = []

        # Runtime state
        self.conversation_history: list[dict] = []
        self.context: list[str] = []           # last N user intents
        self.turn_count: int = 0

        self._load_and_train()

    # ── TRAINING ──────────────────────────────

    def _load_and_train(self):
        """Load intents and build TF-IDF model."""
        with open(self.intents_path, "r") as f:
            data = json.load(f)

        self.intents = data["intents"]

        # Flatten all patterns with their tags
        preprocessed_patterns = []
        for intent in self.intents:
            for pattern in intent["patterns"]:
                self.patterns.append(pattern)
                self.pattern_tags.append(intent["tag"])
                preprocessed_patterns.append(preprocess_to_string(pattern))

        # Fit TF-IDF on all training patterns
        if preprocessed_patterns:
            self.pattern_vectors = self.vectorizer.fit_transform(preprocessed_patterns)

        intent_count = len(self.intents)
        pattern_count = len(self.patterns)
        vocab_size = len(self.vectorizer.vocab)
        print(f"✅ Chatbot trained: {intent_count} intents | {pattern_count} patterns | vocab size: {vocab_size}")

    # ── INFERENCE ─────────────────────────────

    def classify_intent(self, user_input: str) -> tuple[str, float, dict]:
        """
        Classify user input into an intent tag.
        
        Returns:
            (tag, confidence_score, debug_info)
        """
        processed = preprocess_to_string(user_input)
        tokens = tokenize(user_input)
        stemmed = preprocess(user_input)

        # Vectorize query
        query_vec = self.vectorizer.transform_query(processed)

        # Find best matching pattern
        best_idx, best_score = find_best_match(query_vec, self.pattern_vectors)
        best_tag = self.pattern_tags[best_idx] if best_score >= self.CONFIDENCE_THRESHOLD else "fallback"

        # Compute all scores for debug
        all_scores = {}
        for i, tag in enumerate(self.pattern_tags):
            from nlp_utils import cosine_similarity
            score = cosine_similarity(query_vec, self.pattern_vectors[i])
            if tag not in all_scores or score > all_scores[tag]:
                all_scores[tag] = score

        # Sort by score
        top_intents = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:5]

        debug = {
            "raw_input": user_input,
            "tokens": tokens,
            "stemmed_tokens": stemmed,
            "preprocessed": processed,
            "best_pattern": self.patterns[best_idx] if best_idx < len(self.patterns) else "N/A",
            "confidence": best_score,
            "top_intents": top_intents,
            "entities": self.ner.extract(user_input)
        }

        return best_tag, best_score, debug

    def get_response(self, tag: str) -> str:
        """Pick a random response for the given intent tag."""
        for intent in self.intents:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
        # Fallback
        for intent in self.intents:
            if intent["tag"] == "fallback":
                return random.choice(intent["responses"])
        return "I'm not sure how to respond to that."

    def respond(self, user_input: str, debug: bool = False) -> dict:
        """
        Main entry point: process user input and return a response.
        
        Returns a dict with:
          - response: str
          - tag: str
          - confidence: float
          - entities: dict
          - debug: dict (if debug=True)
        """
        self.turn_count += 1
        start_time = time.time()

        # Classify intent
        tag, confidence, debug_info = self.classify_intent(user_input)

        # Generate response
        response_text = self.get_response(tag)

        # Update context and history
        self.context.append(tag)
        if len(self.context) > self.CONTEXT_WINDOW:
            self.context.pop(0)

        elapsed = (time.time() - start_time) * 1000  # ms

        self.conversation_history.append({
            "turn": self.turn_count,
            "user": user_input,
            "bot": response_text,
            "intent": tag,
            "confidence": confidence,
            "time_ms": round(elapsed, 2)
        })

        result = {
            "response": response_text,
            "tag": tag,
            "confidence": confidence,
            "entities": debug_info["entities"],
            "time_ms": round(elapsed, 2),
        }

        if debug:
            result["debug"] = debug_info

        return result

    # ── UTILITIES ─────────────────────────────

    def get_stats(self) -> dict:
        """Return chatbot statistics."""
        if not self.conversation_history:
            return {"turns": 0}

        avg_confidence = sum(h["confidence"] for h in self.conversation_history) / len(self.conversation_history)
        avg_time = sum(h["time_ms"] for h in self.conversation_history) / len(self.conversation_history)
        intent_counts = {}
        for h in self.conversation_history:
            intent_counts[h["intent"]] = intent_counts.get(h["intent"], 0) + 1

        return {
            "turns": self.turn_count,
            "avg_confidence": round(avg_confidence, 4),
            "avg_response_time_ms": round(avg_time, 2),
            "intent_distribution": dict(sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)),
            "vocab_size": len(self.vectorizer.vocab),
            "training_patterns": len(self.patterns),
        }

    def reset(self):
        """Reset conversation state."""
        self.conversation_history.clear()
        self.context.clear()
        self.turn_count = 0


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    bot = NLPChatbot("intents.json")

    test_inputs = [
        "Hello there!",
        "What is NLP?",
        "Tell me a joke",
        "How do you work?",
        "What is TF-IDF?",
        "Thanks for your help!",
        "Goodbye",
    ]

    print("\n" + "=" * 55)
    print("  Chatbot Test Run")
    print("=" * 55)

    for text in test_inputs:
        result = bot.respond(text, debug=True)
        print(f"\n👤 User: {text}")
        print(f"🤖 Bot:  {result['response']}")
        print(f"   Intent: {result['tag']} | Confidence: {result['confidence']:.4f} | {result['time_ms']}ms")
        if result["entities"]:
            print(f"   Entities: {result['entities']}")

    print("\n📊 Session Stats:")
    for k, v in bot.get_stats().items():
        print(f"   {k}: {v}")
