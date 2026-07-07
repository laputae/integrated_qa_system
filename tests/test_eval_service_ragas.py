"""评估自动化管道 — RAGAS integration tests"""
import json


class TestEvalServiceEnsureRagas:
    def test_ensure_ragas_importable(self):
        import sys

        from rag_qa.eval.ragas_runner import ensure_ragas_importable
        ensure_ragas_importable()
        assert "langchain_community.chat_models.vertexai" in sys.modules


class TestEvalServicePrepareRagasDataset:
    def test_prepare_with_valid_contexts(self):
        from db_models.eval_result import EvalResult
        from rag_qa.eval.ragas_runner import prepare_ragas_dataset

        results = [
            EvalResult(id=1, run_id=1, question="Q1", ground_truth="A1",
                       answer="Answer1", contexts=json.dumps(["ctx1", "ctx2"], ensure_ascii=False)),
            EvalResult(id=2, run_id=1, question="Q2", ground_truth="A2",
                       answer="Answer2", contexts=json.dumps(["ctx3"], ensure_ascii=False)),
        ]

        dataset = prepare_ragas_dataset(results)
        assert len(dataset["question"]) == 2
        assert dataset["question"][0] == "Q1"
        assert dataset["contexts"][0] == ["ctx1", "ctx2"]

    def test_prepare_with_none_contexts(self):
        from db_models.eval_result import EvalResult
        from rag_qa.eval.ragas_runner import prepare_ragas_dataset

        results = [
            EvalResult(id=1, run_id=1, question="Q1", ground_truth="A1",
                       answer="Answer1", contexts=None),
        ]
        dataset = prepare_ragas_dataset(results)
        assert dataset["contexts"][0] == []

    def test_prepare_with_none_answer(self):
        from db_models.eval_result import EvalResult
        from rag_qa.eval.ragas_runner import prepare_ragas_dataset

        results = [
            EvalResult(id=1, run_id=1, question="Q1", ground_truth="A1",
                       answer=None, contexts=None),
        ]
        dataset = prepare_ragas_dataset(results)
        assert dataset["answer"][0] == ""
