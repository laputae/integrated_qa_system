"""
HallucinationGuard: real-time NLI-based output-side verification.

Decomposes LLM output into atomic claims and checks each claim against
the retrieved context using a dedicated Chinese NLI model.

Design: SoftGate only — flags/logs/metrics, does NOT block responses.
Runs AFTER streaming completes so Time-to-First-Token is unaffected.
"""
import os
import sys
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Project-relative imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_rag_qa_path = os.path.dirname(_current_dir)
_project_root = os.path.dirname(_rag_qa_path)
sys.path.insert(0, _project_root)

from base import logger, Config
from base.metrics import qa_hallucination_guard_total, qa_hallucination_guard_latency_seconds

conf = Config()


# ============================================================
# Data classes
# ============================================================

@dataclass
class ClaimResult:
    claim: str
    entailment_score: float
    neutral_score: float
    contradiction_score: float
    is_contradicted: bool


@dataclass
class HallucinationResult:
    is_hallucinated: bool
    score: float
    claims: List[ClaimResult]
    details: str


# ============================================================
# HallucinationGuard
# ============================================================

class HallucinationGuard:
    """Real-time NLI-based hallucination detector.

    Uses a CrossEncoder-style NLI model to classify each atomic claim
    from the LLM output as entailed/neutral/contradicted by the context.
    """

    def __init__(self, model_name: str = None, device: str = "cpu"):
        if model_name is None:
            model_name = conf.HALLUCINATION_GUARD_MODEL

        self.device = device
        self.logger = logger

        # Resolve model path: local models/ dir first, then HuggingFace hub
        model_dir = os.path.join(_rag_qa_path, "models", os.path.basename(model_name))
        if os.path.isdir(model_dir):
            model_source = model_dir
            self.logger.info(f"从本地加载 HallucinationGuard 模型: {model_dir}")
        else:
            model_source = model_name
            self.logger.info(f"从 HuggingFace Hub 加载 HallucinationGuard 模型: {model_name}")

        start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_source)
        self.model.to(device)
        self.model.eval()

        load_time = time.time() - start
        self.logger.info(f"HallucinationGuard 模型加载完成 (设备: {device}, 耗时: {load_time:.2f}s)")

        # Configurable thresholds
        self.entailment_threshold = conf.HALLUCINATION_GUARD_ENTAILMENT_THRESHOLD
        self.contradiction_threshold = conf.HALLUCINATION_GUARD_CONTRADICTION_THRESHOLD

    # ============================================================
    # Public API
    # ============================================================

    def verify(self, answer: str, context: str) -> HallucinationResult:
        """Verify that the answer is grounded in the provided context.

        Returns a HallucinationResult without modifying the answer.
        """
        start_time = time.time()

        claims = self._decompose_claims(answer)
        if not claims:
            qa_hallucination_guard_total.labels(result="no_claims").inc()
            return HallucinationResult(
                is_hallucinated=False,
                score=1.0,
                claims=[],
                details="答案中没有可分解的陈述",
            )

        results = []
        for claim in claims:
            r = self._check_claim(claim, context)
            results.append(r)

        hallucinated_claims = [r for r in results if r.is_contradicted]
        all_entailed = len(hallucinated_claims) == 0

        overall_score = (
            sum(r.entailment_score for r in results) / len(results)
            if results else 1.0
        )

        latency = time.time() - start_time
        qa_hallucination_guard_latency_seconds.observe(latency)

        if all_entailed:
            qa_hallucination_guard_total.labels(result="passed").inc()
        else:
            qa_hallucination_guard_total.labels(result="flagged").inc()
            self.logger.warning(
                f"HallucinationGuard: {len(hallucinated_claims)}/{len(claims)} "
                f"个陈述可能缺乏文档依据"
            )

        return HallucinationResult(
            is_hallucinated=not all_entailed,
            score=overall_score,
            claims=results,
            details=(
                f"{len(hallucinated_claims)}/{len(claims)} claims potentially unsupported"
            ),
        )

    # ============================================================
    # Claim decomposition
    # ============================================================

    def _decompose_claims(self, text: str) -> List[str]:
        """Split answer text into atomic claims.

        Handles Chinese sentence boundaries, numbered lists, bullet points,
        and semicolons as clause boundaries.
        """
        text = text.replace("\n\n", "\n")

        # Handle numbered lists and bullet points
        lines = re.split(r"\n\s*(?:\d+[.、）\)]\s*|[-*•]\s*)", text)
        lines = [l.strip() for l in lines if l.strip()]

        claims = []
        for line in lines:
            # Split on Chinese sentence/clause boundaries
            sentences = re.split(r"(?<=[。！？；\n])\s*", line)
            for sent in sentences:
                sent = sent.strip()
                # Strip trailing punctuation for cleaner claims
                sent = sent.rstrip("。！？；，、\n")
                if sent and len(sent) >= 2:
                    claims.append(sent)

        return claims

    # ============================================================
    # NLI inference
    # ============================================================

    def _check_claim(self, claim: str, context: str) -> ClaimResult:
        """Check if a single claim is entailed by the context.

        NLI formulation: premise=context, hypothesis=claim.
        Labels: 0=entailment, 1=neutral, 2=contradiction.
        """
        max_len = self.model.config.max_position_embeddings - len(claim) - 10
        truncated_context = context[:max_len] if len(context) > max_len else context

        inputs = self.tokenizer(
            truncated_context,
            claim,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).squeeze(0)

        entailment_score = float(probs[0].item())
        neutral_score = float(probs[1].item())
        contradiction_score = float(probs[2].item())

        is_contradicted = (
            contradiction_score > self.contradiction_threshold
            and contradiction_score > entailment_score
        )

        return ClaimResult(
            claim=claim,
            entailment_score=entailment_score,
            neutral_score=neutral_score,
            contradiction_score=contradiction_score,
            is_contradicted=is_contradicted,
        )
