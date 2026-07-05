from typing import TYPE_CHECKING
'''
OCR 引擎选择策略：
- GPU 可用 + 显式要求 CUDA → rapidocr_paddle（GPU 加速，需 PaddlePaddle）
- CPU 模式 → rapidocr_onnxruntime（ONNX Runtime，无 PaddlePaddle 依赖）

Windows 上 PaddlePaddle 的 OneDNN 缺少 fused_conv2d 等算子，会报：
  (NotFound) OneDnnContext does not have the input Filter
因此 CPU 模式默认使用 rapidocr_onnxruntime 彻底绕开此问题。
'''


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_ocr(use_cuda: bool = False) -> "RapidOCR":
    """获取 OCR 实例。

    CPU 模式（默认）：使用 rapidocr_onnxruntime，无需 PaddlePaddle。
    GPU 模式（use_cuda=True 且 CUDA 可用）：使用 rapidocr_paddle。
    """
    if use_cuda and _cuda_available():
        from rapidocr_paddle import RapidOCR
        ocr = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
        return ocr

    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()
