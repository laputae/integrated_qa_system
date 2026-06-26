"""Tests for HallucinationGuard: claim decomposition and NLI verification."""
import sys
import os
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestClaimDecomposition:
    """Unit tests for claim splitting logic (no model needed)."""

    @pytest.fixture
    def guard(self):
        from rag_qa.core.nli_guard import HallucinationGuard
        return HallucinationGuard

    def test_split_chinese_sentences(self, guard):
        text = "AI学科包含机器学习、深度学习和自然语言处理。深度学习包括CNN和RNN。"
        claims = guard._decompose_claims(guard, text)
        assert len(claims) == 2
        assert "机器学习" in claims[0]
        assert "CNN" in claims[1]

    def test_split_numbered_list(self, guard):
        text = "要点：\n1. 机器学习\n2. 深度学习\n3. 自然语言处理"
        claims = guard._decompose_claims(guard, text)
        assert len(claims) == 4  # "要点：" + 3 items
        # First item after "要点：" should contain "机器学习"
        assert any("机器学习" in c for c in claims)

    def test_split_semicolons(self, guard):
        text = "优点：速度快；精度高；成本低。"
        claims = guard._decompose_claims(guard, text)
        assert len(claims) >= 2

    def test_empty_answer(self, guard):
        claims = guard._decompose_claims(guard, "")
        assert claims == []

    def test_short_fragments_skipped(self, guard):
        text = "好的。嗯。对。AI学科包含机器学习。"
        claims = guard._decompose_claims(guard, text)
        # "嗯" (1 char) and "对" (1 char) skipped; "好的" (2 chars) and AI claim kept
        assert len(claims) == 2
        assert any("机器学习" in c for c in claims)


class TestGuardIntegration:
    """Integration tests that require the NLI model to be downloaded."""

    @pytest.fixture
    def guard_instance(self):
        try:
            from rag_qa.core.nli_guard import HallucinationGuard
            return HallucinationGuard(device="cpu")
        except Exception:
            pytest.skip("HallucinationGuard model not available")

    def test_verify_grounded_answer(self, guard_instance):
        context = "人工智能课程包括机器学习、深度学习和自然语言处理三个模块。"
        answer = "AI课程包含机器学习、深度学习和自然语言处理三个模块。"
        result = guard_instance.verify(answer, context)
        assert result.score > 0
        assert len(result.claims) > 0

    def test_verify_ungrounded_answer(self, guard_instance):
        context = "人工智能课程包括机器学习和深度学习。"
        answer = "AI课程包含机器学习、深度学习、计算机视觉和强化学习。"
        result = guard_instance.verify(answer, context)
        # At minimum, result should be computed without error
        assert result.score >= 0
        assert len(result.claims) > 0

    def test_verify_empty_context(self, guard_instance):
        context = ""
        answer = "AI课程包含机器学习。"
        result = guard_instance.verify(answer, context)
        # Empty context should not crash
        assert result.score >= 0


class TestResultDataclasses:
    def test_claim_result_fields(self):
        from rag_qa.core.nli_guard import ClaimResult
        r = ClaimResult(
            claim="test",
            entailment_score=0.8,
            neutral_score=0.1,
            contradiction_score=0.1,
            is_contradicted=False,
        )
        assert r.is_contradicted is False
        assert r.entailment_score == 0.8

    def test_hallucination_result_fields(self):
        from rag_qa.core.nli_guard import HallucinationResult
        r = HallucinationResult(
            is_hallucinated=True,
            score=0.4,
            claims=[],
            details="test",
        )
        assert r.is_hallucinated is True
        assert r.score == 0.4
