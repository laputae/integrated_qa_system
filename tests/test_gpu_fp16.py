"""
GPU FP16 加速测试脚本（不修改任何核心代码）

测试内容：
1. CUDA 可用性 + GPU 规格检测
2. 4 个本地模型分别加载 CPU/GPU+FP16 对比
3. 推理速度 benchmark
4. 显存占用报告

用法：uv run python tests/test_gpu_fp16.py
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# ---- Path setup ----
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import torch
import numpy as np

# ============================================================
# 第 0 步：环境探测
# ============================================================

def print_gpu_info():
    print("=" * 60)
    print("  GPU / CUDA 环境探测")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [FAIL] CUDA 不可用 — torch 未检测到 GPU")
        print("         请检查: pip install torch>=2.6.0 (CUDA 版本)")
        return False

    device_count = torch.cuda.device_count()
    print(f"  CUDA 可用: 是")
    print(f"  设备数量: {device_count}")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\n  GPU {i}: {props.name}")
        print(f"    总显存:    {props.total_memory / 1024**3:.2f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")
        print(f"    多处理器数: {props.multi_processor_count}")
        # FP16 硬件加速判断
        if props.major >= 7:
            print(f"    FP16 支持: 是 (Tensor Core: {'是' if props.major >= 8 else '否 — CC 7.x 仅部分支持'})")
        else:
            print(f"    FP16 支持: 有限 (CC < 7.0, 无 fp16 硬件加速)")

    # 显示当前空闲显存
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n  空闲显存: {(total - allocated):.2f} GB / {total:.2f} GB")

    return True


# ============================================================
# 工具函数
# ============================================================

def bench_inference(fn, warmup=3, repeats=10, desc=""):
    """运行推理 benchmark，返回平均耗时(ms)"""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / repeats * 1000
    return elapsed


def gpu_mem_used_mb():
    """返回当前已用显存 (MB)"""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.memory_allocated(0) / 1024**2


def gpu_mem_report():
    """打印显存使用情况"""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    reserved = torch.cuda.memory_reserved(0) / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"    显存: {allocated:.0f} MB 已分配 / {reserved:.0f} MB 保留 / {total:.0f} MB 总计")


# ============================================================
# 测试 1: BGE-Reranker-Large (CrossEncoder)
# ============================================================

def test_reranker():
    print("\n" + "=" * 60)
    print("  测试 1: BGE-Reranker-Large (CrossEncoder, ~300M)")
    print("=" * 60)

    from sentence_transformers import CrossEncoder

    model_path = os.path.join(_project_root, "rag_qa", "models", "bge-reranker-large")
    queries = ["人工智能课程的主要内容是什么"] * 5
    docs = ["本课程介绍人工智能的基本概念，包括机器学习、深度学习、自然语言处理等内容。"] * 5
    pairs = [[q, d] for q, d in zip(queries, docs)]

    # -- CPU --
    print("\n  [CPU] 加载模型...")
    mem_before = gpu_mem_used_mb()
    t0 = time.perf_counter()
    model_cpu = CrossEncoder(model_path, device="cpu")
    load_time_cpu = time.perf_counter() - t0
    print(f"    加载耗时: {load_time_cpu:.2f}s")
    gpu_mem_report()

    cpu_ms = bench_inference(lambda: model_cpu.predict(pairs), desc="CPU")
    print(f"    推理耗时: {cpu_ms:.1f} ms/次 (5对文档)")

    del model_cpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- GPU + FP16 --
    if torch.cuda.is_available():
        print(f"\n  [GPU+FP16] 加载模型...")
        mem_before = gpu_mem_used_mb()
        t0 = time.perf_counter()
        model_gpu = CrossEncoder(model_path, device="cuda")
        model_gpu.model.half()  # FP16
        load_time_gpu = time.perf_counter() - t0
        mem_after = gpu_mem_used_mb()
        print(f"    加载耗时: {load_time_gpu:.2f}s")
        print(f"    显存增量: {mem_after - mem_before:.0f} MB")
        gpu_mem_report()

        gpu_ms = bench_inference(lambda: model_gpu.predict(pairs), desc="GPU+FP16")
        print(f"    推理耗时: {gpu_ms:.1f} ms/次 (5对文档)")

        speedup = cpu_ms / gpu_ms if gpu_ms > 0 else 0
        print(f"    --> 加速比: {speedup:.1f}x")

        del model_gpu
        torch.cuda.empty_cache()
    else:
        print(f"\n  [GPU+FP16] 跳过 — CUDA 不可用")


# ============================================================
# 测试 2: BGE-M3 (Embedding, ~560M)
# ============================================================

def test_bge_m3():
    print("\n" + "=" * 60)
    print("  测试 2: BGE-M3 (Embedding, ~560M)")
    print("=" * 60)

    try:
        from milvus_model.hybrid import BGEM3EmbeddingFunction
    except ImportError:
        print("  [SKIP] milvus_model 未安装")
        return

    model_path = os.path.join(_project_root, "rag_qa", "models", "bge-m3")
    texts = ["人工智能课程的主要内容是什么"] * 5

    # -- CPU --
    print("\n  [CPU] 加载模型...")
    mem_before = gpu_mem_used_mb()
    t0 = time.perf_counter()
    model_cpu = BGEM3EmbeddingFunction(
        model_name_or_path=model_path,
        use_fp16=False,
        device="cpu",
    )
    load_time_cpu = time.perf_counter() - t0
    print(f"    加载耗时: {load_time_cpu:.2f}s")
    gpu_mem_report()

    cpu_ms = bench_inference(lambda: model_cpu.encode_documents(texts), desc="CPU")
    print(f"    推理耗时: {cpu_ms:.1f} ms/次 (5条文本)")

    del model_cpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- GPU + FP16 --
    if torch.cuda.is_available():
        print(f"\n  [GPU+FP16] 加载模型...")
        mem_before = gpu_mem_used_mb()
        t0 = time.perf_counter()
        model_gpu = BGEM3EmbeddingFunction(
            model_name_or_path=model_path,
            use_fp16=True,
            device="cuda",
        )
        load_time_gpu = time.perf_counter() - t0
        mem_after = gpu_mem_used_mb()
        print(f"    加载耗时: {load_time_gpu:.2f}s")
        print(f"    显存增量: {mem_after - mem_before:.0f} MB")
        gpu_mem_report()

        gpu_ms = bench_inference(lambda: model_gpu.encode_documents(texts), desc="GPU+FP16")
        print(f"    推理耗时: {gpu_ms:.1f} ms/次 (5条文本)")

        speedup = cpu_ms / gpu_ms if gpu_ms > 0 else 0
        print(f"    --> 加速比: {speedup:.1f}x")

        del model_gpu
        torch.cuda.empty_cache()
    else:
        print(f"\n  [GPU+FP16] 跳过 — CUDA 不可用")


# ============================================================
# 测试 3: NLI Hallucination Guard (mDeBERTa-v3, ~300M)
# ============================================================

def test_nli_guard():
    print("\n" + "=" * 60)
    print("  测试 3: NLI Hallucination Guard (mDeBERTa-v3, ~300M)")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_path = os.path.join(
        _project_root, "rag_qa", "models",
        "mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    premise = "人工智能是一门研究如何让计算机模拟人类智能行为的科学。"
    hypothesis = "人工智能是计算机科学的一个分支。"

    def infer(model, device):
        inputs = tokenizer(premise, hypothesis, truncation=True, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.logits

    # -- CPU --
    print("\n  [CPU] 加载模型...")
    mem_before = gpu_mem_used_mb()
    t0 = time.perf_counter()
    model_cpu = AutoModelForSequenceClassification.from_pretrained(model_path)
    model_cpu.to("cpu")
    model_cpu.eval()
    load_time_cpu = time.perf_counter() - t0
    print(f"    加载耗时: {load_time_cpu:.2f}s")
    gpu_mem_report()

    # 预热一次
    _ = infer(model_cpu, "cpu")
    cpu_ms = bench_inference(lambda: infer(model_cpu, "cpu"), warmup=2, desc="CPU")
    print(f"    推理耗时: {cpu_ms:.1f} ms/次")

    del model_cpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- GPU + FP16 --
    if torch.cuda.is_available():
        print(f"\n  [GPU+FP16] 加载模型...")
        mem_before = gpu_mem_used_mb()
        t0 = time.perf_counter()
        model_gpu = AutoModelForSequenceClassification.from_pretrained(model_path)
        model_gpu.half().to("cuda")  # FP16
        model_gpu.eval()
        load_time_gpu = time.perf_counter() - t0
        mem_after = gpu_mem_used_mb()
        print(f"    加载耗时: {load_time_gpu:.2f}s")
        print(f"    显存增量: {mem_after - mem_before:.0f} MB")
        gpu_mem_report()

        _ = infer(model_gpu, "cuda")
        gpu_ms = bench_inference(lambda: infer(model_gpu, "cuda"), warmup=2, desc="GPU+FP16")
        print(f"    推理耗时: {gpu_ms:.1f} ms/次")

        speedup = cpu_ms / gpu_ms if gpu_ms > 0 else 0
        print(f"    --> 加速比: {speedup:.1f}x")

        del model_gpu
        torch.cuda.empty_cache()
    else:
        print(f"\n  [GPU+FP16] 跳过 — CUDA 不可用")


# ============================================================
# 测试 4: BERT Query Classifier (bert-base-chinese, ~110M)
# ============================================================

def test_bert_classifier():
    print("\n" + "=" * 60)
    print("  测试 4: BERT Query Classifier (bert-base-chinese, ~110M)")
    print("=" * 60)

    from transformers import BertTokenizer, BertForSequenceClassification

    model_path = os.path.join(_project_root, "rag_qa", "models", "bert-base-chinese")
    tokenizer = BertTokenizer.from_pretrained(model_path)

    texts = ["人工智能课程的主要内容是什么", "什么是深度学习", "请介绍一下机器学习"] * 2

    def infer(model, device):
        inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.logits

    # -- CPU --
    print("\n  [CPU] 加载模型...")
    mem_before = gpu_mem_used_mb()
    t0 = time.perf_counter()
    model_cpu = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model_cpu.to("cpu")
    model_cpu.eval()
    load_time_cpu = time.perf_counter() - t0
    print(f"    加载耗时: {load_time_cpu:.2f}s")
    gpu_mem_report()

    _ = infer(model_cpu, "cpu")
    cpu_ms = bench_inference(lambda: infer(model_cpu, "cpu"), warmup=2, desc="CPU")
    print(f"    推理耗时: {cpu_ms:.1f} ms/次 (6条文本)")

    del model_cpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- GPU + FP16 --
    if torch.cuda.is_available():
        print(f"\n  [GPU+FP16] 加载模型...")
        mem_before = gpu_mem_used_mb()
        t0 = time.perf_counter()
        model_gpu = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
        model_gpu.half().to("cuda")
        model_gpu.eval()
        load_time_gpu = time.perf_counter() - t0
        mem_after = gpu_mem_used_mb()
        print(f"    加载耗时: {load_time_gpu:.2f}s")
        print(f"    显存增量: {mem_after - mem_before:.0f} MB")
        gpu_mem_report()

        _ = infer(model_gpu, "cuda")
        gpu_ms = bench_inference(lambda: infer(model_gpu, "cuda"), warmup=2, desc="GPU+FP16")
        print(f"    推理耗时: {gpu_ms:.1f} ms/次 (6条文本)")

        speedup = cpu_ms / gpu_ms if gpu_ms > 0 else 0
        print(f"    --> 加速比: {speedup:.1f}x")

        del model_gpu
        torch.cuda.empty_cache()
    else:
        print(f"\n  [GPU+FP16] 跳过 — CUDA 不可用")


# ============================================================
# 汇总
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("  GPU FP16 加速可行性测试")
    print("  项目: integrated_qa_system")
    print("=" * 60)

    cuda_ok = print_gpu_info()

    if not cuda_ok:
        print("\n  [结论] CUDA 不可用，无法进行 GPU FP16 加速")
        print("  请安装 CUDA 版本的 PyTorch:")
        print("    pip install torch>=2.6.0 --index-url https://download.pytorch.org/whl/cu126")
        return

    test_reranker()
    test_bge_m3()
    test_nli_guard()
    test_bert_classifier()

    print("\n" + "=" * 60)
    print("  测试完成")
    print("=" * 60)

    allocated = torch.cuda.memory_allocated(0) / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"  最终显存: {allocated:.0f} MB 已分配 / {total:.0f} MB 总计")


if __name__ == "__main__":
    main()
