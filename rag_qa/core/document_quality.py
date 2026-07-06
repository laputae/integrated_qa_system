"""
Document text cleaning and quality assessment utilities.

Extracted from llamaindex_processor.py to keep each module under 300 lines.
"""
import re

# ---- CJK character ranges ----

_CJK_START = 0x4E00        # CJK统一表意文字起始
_CJK_END = 0x9FFF          # CJK统一表意文字结尾
_CJK_EXT_A_START = 0x3400  # CJK扩展A起始
_CJK_EXT_A_END = 0x4DBF    # CJK扩展A结尾
_STANDARD_PUNCT = set(',.;:!?"\'()[]{}<>-+/\\| \t\n\r@#$%^&*~`=')

LOW_QUALITY_THRESHOLD = 0.3


# ---- Text cleaning ----

def clean_document_text(text: str) -> str:
    """OCR文本预处理管道：去除噪音、规范化空白、统一标点"""
    if not text:
        return text

    # 1. 去除零宽字符
    text = re.sub(
        r'[­ -‏   ⁠-⁤　﻿￾￿]',
        '', text
    )

    # 2. 规范化换行 → 单 \n
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 3. 统一中英文标点
    text = text.replace('，', ',')
    text = text.replace('；', ';')
    text = text.replace('：', ':')
    text = text.replace('（', '(').replace('）', ')')
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('【', '[').replace('】', ']')
    text = text.replace('《', '<').replace('》', '>')
    text = text.replace('！', '!')
    text = text.replace('？', '?')
    text = text.replace('～', '~')

    # 4. 去除页码/页眉/页脚噪音
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[\(（]?\d{1,4}[\)）]?\s*$', '', text, flags=re.MULTILINE)

    # 5. 压缩多余空白
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'^[ \t]+|[ \t]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n +', '\n', text)
    text = re.sub(r' +\n', '\n', text)

    # 6. 清理连续空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text


# ---- Quality assessment ----

def _is_content_char(c: str) -> bool:
    """是否为内容字符（中文、拉丁字母、数字）"""
    cp = ord(c)
    if _CJK_START <= cp <= _CJK_END:
        return True
    if _CJK_EXT_A_START <= cp <= _CJK_EXT_A_END:
        return True
    return c.isascii() and (c.isalpha() or c.isdigit())


def estimate_document_quality(doc) -> float:
    """评估 OCR 文档质量，返回 0-1 分数。

    基于已清洗文本（clean_document_text 之后）评估：
      - 文本长度充足度（权重 0.30）
      - 有效字符占比（权重 0.40）
      - OCR 噪音伪影（权重 0.30）

    同时设置 doc.metadata["quality_score"] 和 doc.metadata["is_low_quality"]。
    """
    text = doc.page_content
    if not text or not text.strip():
        doc.metadata["quality_score"] = 0.0
        doc.metadata["is_low_quality"] = True
        return 0.0

    total = len(text)

    # 1. 文本长度分数（权重 0.30）
    if total < 50:
        length_score = 0.0
    elif total < 200:
        length_score = (total - 50) / 150 * 0.5
    elif total < 500:
        length_score = 0.5 + (total - 200) / 300 * 0.35
    else:
        length_score = 1.0

    # 2. 有效字符占比（权重 0.40）
    content_chars = sum(1 for c in text if _is_content_char(c))
    content_ratio = content_chars / total

    # 3. OCR 噪音分数（权重 0.30）
    # 3a. 连续重复字符
    repeat_count = len(re.findall(r'(.)\1{5,}', text))
    repeat_penalty = min(repeat_count * 0.1, 0.30)

    # 3b. 非标准字符惩罚
    non_standard = sum(
        1 for c in text
        if not _is_content_char(c) and c not in _STANDARD_PUNCT and not c.isspace()
    )
    ns_ratio = non_standard / total
    ns_penalty = min(ns_ratio * 2.0, 0.40)

    # 3c. 行结构一致性
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

    # 综合评分
    quality = (0.30 * length_score +
               0.40 * content_ratio +
               0.30 * noise_score)
    quality = max(0.0, min(1.0, quality))
    # 几乎没有有效内容时硬封顶
    if content_ratio < 0.1:
        quality = min(quality, 0.15)

    doc.metadata["quality_score"] = round(quality, 4)
    doc.metadata["is_low_quality"] = quality < LOW_QUALITY_THRESHOLD

    return quality
