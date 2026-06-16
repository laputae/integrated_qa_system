# -*-coding:utf-8-*-
"""estimate_document_quality 独立冒烟测试（不触发 llamaindex_processor 的重型 ML 导入）"""
import re


# ---- 从 llamaindex_processor.py 复制的质量评估逻辑 ----

_CJK_START = 0x4E00        # CJK统一表意文字起始
_CJK_END = 0x9FFF          # CJK统一表意文字结尾
_CJK_EXT_A_START = 0x3400  # CJK扩展A起始
_CJK_EXT_A_END = 0x4DBF    # CJK扩展A结尾
_STANDARD_PUNCT = set(',.;:!?"\'()[]{}<>-+/\\| \t\n\r@#$%^&*~`=')

LOW_QUALITY_THRESHOLD = 0.3


def _is_content_char(c: str) -> bool:
    cp = ord(c)
    if _CJK_START <= cp <= _CJK_END:
        return True
    if _CJK_EXT_A_START <= cp <= _CJK_EXT_A_END:
        return True
    return c.isascii() and (c.isalpha() or c.isdigit())


def estimate_document_quality(doc) -> float:
    text = doc.page_content
    if not text or not text.strip():
        doc.metadata["quality_score"] = 0.0
        doc.metadata["is_low_quality"] = True
        return 0.0

    total = len(text)

    # 1. 文本长度分数
    if total < 50:
        length_score = 0.0
    elif total < 200:
        length_score = (total - 50) / 150 * 0.5
    elif total < 500:
        length_score = 0.5 + (total - 200) / 300 * 0.35
    else:
        length_score = 1.0

    # 2. 有效字符占比
    content_chars = sum(1 for c in text if _is_content_char(c))
    content_ratio = content_chars / total

    # 3. OCR 噪音分数
    repeat_count = len(re.findall(r'(.)\1{5,}', text))
    repeat_penalty = min(repeat_count * 0.1, 0.30)

    non_standard = sum(
        1 for c in text
        if not _is_content_char(c) and c not in _STANDARD_PUNCT and not c.isspace()
    )
    ns_ratio = non_standard / total
    ns_penalty = min(ns_ratio * 2.0, 0.40)

    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        avg_line_len = sum(len(l) for l in lines) / len(lines)
        if avg_line_len < 15:
            line_penalty = (15 - avg_line_len) / 15 * 0.30
        elif avg_line_len > 300:
            line_penalty = min((avg_line_len - 300) / 300, 1.0) * 0.30
        else:
            line_penalty = 0.0
    else:
        line_penalty = 0.30

    noise_penalty = min(repeat_penalty + ns_penalty + line_penalty, 0.80)
    noise_score = 1.0 - noise_penalty

    quality = (0.30 * length_score +
               0.40 * content_ratio +
               0.30 * noise_score)
    quality = max(0.0, min(1.0, quality))
    if content_ratio < 0.1:
        quality = min(quality, 0.15)

    doc.metadata["quality_score"] = round(quality, 4)
    doc.metadata["is_low_quality"] = quality < LOW_QUALITY_THRESHOLD

    return quality


# ---- 简易 Document 模拟 ----

class FakeDoc:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}


# ---- 测试用例 ----

def test_empty_text():
    doc = FakeDoc("")
    score = estimate_document_quality(doc)
    assert score == 0.0, f"空文本应得0分，实际: {score}"
    assert doc.metadata["is_low_quality"] is True
    print("  PASS test_empty_text")


def test_whitespace_only():
    doc = FakeDoc("   \n\n  ")
    score = estimate_document_quality(doc)
    assert score == 0.0, f"仅空白应得0分，实际: {score}"
    assert doc.metadata["is_low_quality"] is True
    print("  PASS test_whitespace_only")


def test_short_chinese():
    doc = FakeDoc("机器学习概述 监督学习 无监督学习")
    score = estimate_document_quality(doc)
    # 内容干净但长度不足，长度分为0，综合约0.65
    assert 0.55 < score < 0.75, f"短中文分数异常，实际: {score}"
    print(f"  PASS test_short_chinese (score={score:.4f})")


def test_long_clean_chinese():
    text = "人工智能" * 200  # 800 chars, all CJK
    doc = FakeDoc(text)
    score = estimate_document_quality(doc)
    assert score >= 0.85, f"长干净中文应得高分，实际: {score}"
    assert doc.metadata["is_low_quality"] is False
    print(f"  PASS test_long_clean_chinese (score={score:.4f})")


def test_garbled_punctuation():
    # 使用非标准字符模拟 OCR 乱码
    doc = FakeDoc("¤" * 300)
    score = estimate_document_quality(doc)
    assert score < 0.3, f"OCR乱码应得低分，实际: {score}"
    assert doc.metadata["is_low_quality"] is True
    print(f"  PASS test_garbled_punctuation (score={score:.4f})")


def test_short_english():
    doc = FakeDoc("Hello World! This is a test.")
    score = estimate_document_quality(doc)
    # 内容干净但太短，长度分0，约0.57
    assert 0.45 < score < 0.70, f"短英文分数异常，实际: {score}"
    print(f"  PASS test_short_english (score={score:.4f})")


def test_metadata_preserved():
    existing_meta = {"source": "test", "file_path": "/tmp/test.txt"}
    doc = FakeDoc("人工智能" * 200, metadata=existing_meta)
    estimate_document_quality(doc)
    assert doc.metadata["source"] == "test"
    assert doc.metadata["file_path"] == "/tmp/test.txt"
    assert "quality_score" in doc.metadata
    assert "is_low_quality" in doc.metadata
    print("  PASS test_metadata_preserved")


def test_repeat_penalty():
    """连续重复字符应触发惩罚"""
    doc = FakeDoc("aaaaaa" + "正常文本" * 100)
    score = estimate_document_quality(doc)
    # 有重复字符惩罚但仍有很多正常内容，分数应该还行但低于完美
    assert score < 1.0, f"重复字符应有惩罚，实际: {score}"
    print(f"  PASS test_repeat_penalty (score={score:.4f})")


def test_line_fragmentation():
    """碎片化行应触发惩罚"""
    doc = FakeDoc("a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\n")
    score = estimate_document_quality(doc)
    # 行极短触发 line_penalty，但内容字符有效，综合约0.42
    assert 0.30 < score < 0.55, f"碎片化行分数异常，实际: {score}"
    print(f"  PASS test_line_fragmentation (score={score:.4f})")


if __name__ == "__main__":
    print("estimate_document_quality 冒烟测试")
    print(f"LOW_QUALITY_THRESHOLD = {LOW_QUALITY_THRESHOLD}")
    print("=" * 50)

    test_empty_text()
    test_whitespace_only()
    test_short_chinese()
    test_long_clean_chinese()
    test_garbled_punctuation()
    test_short_english()
    test_metadata_preserved()
    test_repeat_penalty()
    test_line_fragmentation()

    print("=" * 50)
    print("所有测试通过")
