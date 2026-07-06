"""
RAGAS evaluation integration helpers.

Extracted from eval_service.py to keep each module under 300 lines.
Standalone functions that take dependencies as explicit parameters.
"""
import json
import sys
import types
from typing import Optional

from openai import OpenAI

from base import logger, Config


def ensure_ragas_importable():
    """Monkey-patch missing langchain_community modules that ragas requires."""
    import langchain_community.chat_models as chat_models_module

    if "langchain_community.chat_models.vertexai" not in sys.modules:
        vertexai_module = types.ModuleType("langchain_community.chat_models.vertexai")

        class ChatVertexAI:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "ChatVertexAI is not available. Install langchain-google-vertexai "
                    "or use a different LLM for RAGAS evaluation."
                )

        vertexai_module.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = vertexai_module
        setattr(chat_models_module, "vertexai", vertexai_module)


def prepare_ragas_dataset(results: list) -> "Dataset":
    """Convert eval results to a RAGAS-compatible Dataset."""
    from datasets import Dataset

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    for r in results:
        data["question"].append(r.question)
        data["answer"].append(r.answer or "")
        data["ground_truth"].append(r.ground_truth)
        if r.contexts:
            try:
                ctx = json.loads(r.contexts)
            except (json.JSONDecodeError, TypeError):
                ctx = [r.contexts]
        else:
            ctx = []
        data["contexts"].append(ctx)

    return Dataset.from_dict(data)


def create_langchain_llm(config: Config):
    """Create a LangChain-compatible LLM for RAGAS evaluation."""
    from ragas.llms import llm_factory

    model = config.EVAL_LLM_MODEL or config.LLM_MODEL
    base_url = config.EVAL_LLM_BASE_URL or config.DASHSCOPE_BASE_URL
    api_key = config.DASHSCOPE_API_KEY

    client = OpenAI(api_key=api_key, base_url=base_url)
    return llm_factory(model, client=client)


def create_langchain_embeddings(config: Config):
    """Create LangChain-compatible embeddings for RAGAS evaluation."""
    from ragas.embeddings.base import embedding_factory

    base_url = config.EVAL_EMBEDDING_BASE_URL
    model = config.EVAL_EMBEDDING_MODEL
    api_key = config.DASHSCOPE_API_KEY

    if "11434" in base_url or "ollama" in base_url.lower():
        client = OpenAI(api_key="ollama", base_url=base_url.rstrip("/") + "/v1")
    else:
        client = OpenAI(api_key=api_key, base_url=base_url)

    return embedding_factory("openai", model=model, client=client)


def run_ragas(dataset: "Dataset", config: Config, log=None) -> dict:
    """Run RAGAS evaluation metrics on a dataset."""
    if log is None:
        log = logger

    ensure_ragas_importable()

    from ragas import evaluate
    from ragas.metrics.collections import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )

    llm = create_langchain_llm(config)
    embeddings = create_langchain_embeddings(config)

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            ContextPrecision(llm=llm),
            ContextRecall(llm=llm),
        ],
        llm=llm,
        embeddings=embeddings,
    )

    return {
        "faithfulness": result.get("faithfulness", []),
        "answer_relevancy": result.get("answer_relevancy", []),
        "context_precision": result.get("context_precision", []),
        "context_recall": result.get("context_recall", []),
    }
