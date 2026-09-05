"""阿里云百炼（MaaS）千问 VL 图片识别封装 — DashScope 原生 SDK 版。

通过 dashscope.MultiModalConversation 调用视觉大模型，把图片转换为结构化
JSON，再扁平化为可拼接进 RAG chunk 的文本。

配置来源：
- config.ini [vlm] 段：model、api_key（支持 ${DASHSCOPE_API_KEY} 引用环境变量/.env）、api_host（拼 DashScope 原生 /api/v1 地址）
- 环境变量 DASHSCOPE_API_KEY：业务空间 API Key（兜底，config.ini 未配置 api_key 时直接读取）
"""

import base64
import configparser
import json
import os
import time
from io import BytesIO
from pathlib import Path

import dashscope
from dashscope import MultiModalConversation
from PIL import Image

from base.config import expand_env_vars, load_project_dotenv


def _load_vlm_config() -> tuple[str, str, str, int, float]:
    """从 config.ini 读取 [vlm] base_url(或 api_host)/api_key/model 和 [retry] 重试参数。"""
    parser = configparser.ConfigParser(interpolation=None)
    config_path = Path(__file__).resolve().parents[2] / "config.ini"
    # .env 中可能定义 api_key 引用的环境变量，先加载（不覆盖已有环境变量）
    load_project_dotenv()
    if config_path.exists():
        parser.read(config_path, encoding="utf-8")
    # base_url 优先；否则用 api_host 拼 DashScope 原生 /api/v1 路径
    base_url = parser.get("vlm", "base_url", fallback="")
    if not base_url:
        api_host = parser.get("vlm", "api_host", fallback="").strip().removeprefix("https://").removeprefix("http://").rstrip("/")
        if api_host:
            base_url = f"https://{api_host}/api/v1"
    model = parser.get("vlm", "model", fallback="qwen3.7-flash")
    # api_key 支持 ${DASHSCOPE_API_KEY} 引用；为空时调用处兜底读系统环境变量
    vlm_api_key = expand_env_vars(parser.get("vlm", "api_key", fallback=""))
    if not base_url:
        raise RuntimeError(
            "config.ini 缺少 [vlm] api_host 配置，例如 "
            "{WorkspaceId}.cn-beijing.maas.aliyuncs.com"
        )
    max_retries = parser.getint("retry", "max_retries", fallback=3)
    base_delay = parser.getfloat("retry", "base_delay", fallback=1.0)
    return base_url, vlm_api_key, model, max_retries, base_delay


VLM_BASE_URL, VLM_API_KEY, VLM_MODEL, VLM_MAX_RETRIES, VLM_RETRY_BASE_DELAY = _load_vlm_config()
# DashScope 原生 SDK 的服务地址（参考: dashscope.base_http_api_url = "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1"）
dashscope.base_http_api_url = VLM_BASE_URL

# 图片理解的输出 schema：summary 用于向量检索，ocr_text 用于精确引用，
# tables/charts 保留结构化数据，source.bbox 用于溯源定位。
VLM_PROMPT = """请识别图片内容并只输出一个 JSON 对象（不要 markdown 代码块、不要多余文字），字段如下：
{
  "type": "chart | table | diagram | photo | screenshot | slide | mixed",
  "title": "图表标题，没有则为 null",
  "caption": "题注文字，没有则为 null",
  "summary": "用1~3句话描述图片整体内容，保留关键结论",
  "keywords": ["关键词"],
  "ocr_text": "图中出现的所有文字，按阅读顺序，包含表格单元格、坐标轴标签、图例",
  "tables": [{"headers": ["列名"], "rows": [["值"]]}],
  "charts": [{"chart_type": "bar", "axes": {"x": "x轴名称", "y": "y轴名称"}, "series": [{"name": "系列名", "points": [{"x": "横轴值", "y": 0}]}], "trend": "简要结论"}],
  "structure": "图中元素的空间布局或流程关系描述，没有则为 null"
}
要求：图中没有的信息填 null 或空数组；禁止编造图中不存在的数字。"""

def _get_text(resp_content) -> str:
    """从 dashscope 响应的 content 中提取文本（兼容 list[dict] 和 str 两种格式）。"""
    if isinstance(resp_content, str):
        return resp_content
    if isinstance(resp_content, list):
        return "".join(part.get("text", "") for part in resp_content if isinstance(part, dict))
    return str(resp_content or "")


def get_vlm():
    """返回一个函数：传入 PNG 字节流，返回解析后的 dict（失败返回 {}）。"""

    def describe_image(img_bytes: bytes) -> dict:
        last_err: str | None = None
        image_uri = "data:image/png;base64," + base64.b64encode(img_bytes).decode()
        for attempt in range(1, VLM_MAX_RETRIES + 1):
            try:
                response = MultiModalConversation.call(
                    api_key=VLM_API_KEY or os.getenv("DASHSCOPE_API_KEY"),
                    model=VLM_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"image": image_uri},
                                {"text": VLM_PROMPT},
                            ],
                        }
                    ],
                )
                if response.status_code != 200:
                    # 鉴权失败（401/InvalidApiKey）重试无意义，直接跳过
                    code = str(getattr(response, "code", "") or "")
                    if response.status_code == 401 or "apikey" in code.lower():
                        print(f"VLM 鉴权失败（API Key 无效）: code={response.code}, message={response.message}")
                        return {}
                    last_err = f"code={response.code}, message={response.message}"
                    print(f"VLM 调用失败（第 {attempt}/{VLM_MAX_RETRIES} 次）: {last_err}")
                else:
                    content = response.output.choices[0].message.content
                    return parse_vlm_json(_get_text(content))
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                print(f"VLM 调用失败（第 {attempt}/{VLM_MAX_RETRIES} 次）: {e}")
            if attempt < VLM_MAX_RETRIES:
                time.sleep(VLM_RETRY_BASE_DELAY * (2 ** (attempt - 1)))  # 指数退避
        print("VLM 调用最终失败，返回空结果")
        return {}

    return describe_image


def parse_vlm_json(content: str | None) -> dict:
    """解析模型返回的 JSON，容忍 ```json 包裹与解析失败。"""
    if not content:
        return {}
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {"summary": text}


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def vlm_json_to_text(data: dict) -> str:
    """把 VLM 输出的 JSON 扁平化为拼接进 chunk 的文本。"""
    if not data:
        return ""
    parts = []
    for key in ("title", "caption", "summary"):
        if data.get(key):
            parts.append(str(data[key]))
    keywords = data.get("keywords") or []
    if keywords:
        parts.append("关键词: " + "、".join(map(str, keywords)))
    if data.get("structure"):
        parts.append(f"结构: {data['structure']}")
    if data.get("ocr_text"):
        parts.append(f"图中文字: {data['ocr_text']}")
    for table in data.get("tables") or []:
        parts.append(_table_to_markdown(table))
    for chart in data.get("charts") or []:
        parts.append(_chart_to_markdown(chart))
    return "\n".join(p for p in parts if p)


def _table_to_markdown(table: dict) -> str:
    headers = table.get("headers") or []
    rows = table.get("rows") or []
    if not headers and not rows:
        return ""
    lines = []
    if headers:
        lines.append("| " + " | ".join(map(str, headers)) + " |")
        lines.append("|" + "---|" * len(headers))
    for row in rows:
        cells = [str(c) for c in row] if isinstance(row, list) else [str(row)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _chart_to_markdown(chart: dict) -> str:
    lines = []
    chart_type = chart.get("chart_type")
    if chart_type:
        lines.append(f"图表({chart_type})")
    axes = chart.get("axes") or {}
    if axes.get("x") or axes.get("y"):
        lines.append(f"坐标轴: x={axes.get('x')}, y={axes.get('y')}")
    for series in chart.get("series") or []:
        name = series.get("name", "")
        points = series.get("points") or []
        point_str = ", ".join(f"{p.get('x')}={p.get('y')}" for p in points)
        lines.append(f"{name}: {point_str}")
    if chart.get("trend"):
        lines.append(f"结论: {chart['trend']}")
    return "\n".join(lines)
