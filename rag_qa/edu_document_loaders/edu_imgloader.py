from collections.abc import Iterator

import numpy as np
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from PIL import Image

from .edu_ocr import get_ocr
from .edu_vlm import get_vlm, image_to_png_bytes, vlm_json_to_text


class OCRIMGLoader(BaseLoader):
    """独立图片加载器：优先用千问 VL 生成结构化 JSON，失败时回退本地 OCR。"""

    def __init__(self, img_path: str) -> None:
        """Initialize the loader with a file path.

        Args:
            img_path: The path to the img to load.
        """
        self.img_path = img_path

    def lazy_load(self) -> Iterator[Document]:
        # <-- Does not take any arguments
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """

        line = self.img2text()
        yield Document(page_content=line, metadata={"source": self.img_path})

    def img2text(self):
        # 千问 VL 识别，返回结构化 JSON 后扁平化为文本
        vlm = get_vlm()
        image = Image.open(self.img_path)
        data = vlm(image_to_png_bytes(image.convert("RGB")))
        if data:
            return vlm_json_to_text(data)

        # VLM 失败/返回空时，回退本地 OCR
        resp = ""
        ocr = get_ocr()
        result, _ = ocr(np.array(image.convert("RGB")))
        if result:
            ocr_result = [line[1] for line in result]
            resp += "\n".join(ocr_result)
        return resp


if __name__ == "__main__":
    img_loader = OCRIMGLoader(
        img_path="/Users/ligang/Desktop/EduRAG课堂资料/codes/integrated_qa_system/rag_qa/samples/ocr_04.png"
    )
    doc = img_loader.load()
    print(doc)
