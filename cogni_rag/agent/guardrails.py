"""
Guardrails Module: Prompt Injection Filter and Anti-Hallucination Grounding Validator.
"""

import re
from typing import List, Dict, Any, Tuple
from cogni_rag.core.types import SearchResult


class GuardrailEngine:
    INJECTION_PATTERNS = [
        r"ignore (all )?(previous|prior) instructions",
        r"system prompt override",
        r"you are now (unrestricted|jailbroken|dan)",
        r"reveal (the )?(system|internal) prompt",
        r"disregard (all )?guidelines",
        r"output the word ['\"]?hacked['\"]?",
        r"<script.*?>.*?</script>"
    ]

    @classmethod
    def check_prompt_injection(cls, text: str) -> Tuple[bool, str]:
        """Detects adversarial jailbreak attempts and prompt injections."""
        lower = text.lower()
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, lower):
                return True, f"Security Alert: Input matched prohibited adversarial pattern: '{pattern}'"
        return False, "Input passed safety filter."

    @classmethod
    def verify_grounding(cls, answer: str, citations: List[SearchResult]) -> Tuple[float, bool]:
        """
        Evaluates whether claims and factual tokens in the answer are supported by retrieved context.
        Returns (faithfulness_score in [0.0, 1.0], is_grounded: bool).
        """
        if not citations:
            # If no context was retrieved, check if answer honestly admits lack of knowledge
            if any(phrase in answer.lower() for phrase in ["not found", "does not contain", "no information", "cannot find", "unable to locate"]):
                return 1.0, True
            return 0.20, False

        # Aggregate context tokens
        context_text = " ".join([c.chunk.text for c in citations]).lower()
        context_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", context_text))

        # Answer tokens (excluding common stop words)
        stop_words = {
            "the", "and", "for", "that", "this", "with", "from", "have", "are", "was",
            "were", "will", "been", "has", "can", "could", "should", "would", "about",
            "there", "their", "they", "what", "which", "when", "where", "who", "why"
        }
        answer_tokens = [
            w for w in re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", answer.lower())
            if w not in stop_words
        ]

        if not answer_tokens:
            return 1.0, True

        supported_count = sum(1 for t in answer_tokens if t in context_tokens)
        faithfulness = round(supported_count / len(answer_tokens), 4)
        is_grounded = faithfulness >= 0.35

        return faithfulness, is_grounded
