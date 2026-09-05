from collections.abc import Iterator

import cv2
import fitz  # pyMuPDF里面的fitz包，不要与pip install fitz混淆
import numpy as np
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from PIL import Image
from tqdm import tqdm

from .edu_vlm import get_vlm, image_to_png_bytes, vlm_json_to_text

# PDF 图片控制：只对宽高超过页面一定比例（图片宽/页面宽，图片高/页面高）的图片送千问 VL 识别。
# 这样可以避免 PDF 中一些小图片的干扰，提高非扫描版 PDF 处理速度
PDF_OCR_THRESHOLD = (0.6, 0.6)
# 页面可提取文本低于该字符数且未处理过大图时，整页渲染送 VL（兜底纯扫描页）
PAGE_TEXT_MIN_CHARS = 30


class OCRPDFLoader(BaseLoader):
    """An example document loader that reads a file line by line."""

    def __init__(self, file_path: str) -> None:
        """Initialize the loader with a file path.

        Args:
            file_path: The path to the file to load.
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        # <-- Does not take any arguments
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """

        line = self.pdf2text()
        yield Document(page_content=line, metadata={"source": self.file_path})

    def pdf2text(self):
        vlm = get_vlm()
        # 打开pdf文件
        doc = fitz.open(self.file_path)
        ## 获取页数
        # print(f'len(doc)-->{len(doc)}')
        resp = ""
        b_unit = tqdm(total=doc.page_count, desc="OCRPDFLoader context page index: 0")
        for i, page in enumerate(doc):
            b_unit.set_description(f"OCRPDFLoader context page index: {i}")
            b_unit.refresh()
            # 提取文本：默认使用 "text" 模式提取文本。
            # text = page.get_text("")
            text = page.get_text("text")
            resp += text + "\n"
            # print(f'resp-->{resp}')
            # 获取图片：获得所有显示的图像的元信息列表。
            # 它适用于所有文档类型，不仅限于 PDF。
            img_list = page.get_image_info(xrefs=True)
            # print(f'img_list--》{img_list}')
            # print(f'img_list--》{len(img_list)}')
            page_vlm_done = False
            for img in img_list:
                # xref一种编号，指向该图像对象在PDF文件中的位置，程序可以通过这个编号快速定位和提取图像数据。
                if xref := img.get("xref"):
                    # 图像在页面上的位置和尺寸。
                    bbox = img["bbox"]
                    # 检查图片尺寸是否超过设定的阈值
                    if (bbox[2] - bbox[0]) / (page.rect.width) < PDF_OCR_THRESHOLD[0] or (bbox[3] - bbox[1]) / (
                        page.rect.height
                    ) < PDF_OCR_THRESHOLD[1]:
                        continue
                    pix = fitz.Pixmap(doc, xref)
                    if pix.colorspace and pix.colorspace.n > 3:  # CMYK 等色彩空间转 RGB
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    # print(f'page.rotation-->{page.rotation}')
                    if int(page.rotation) != 0:  # 如果Page有旋转角度，则旋转图片
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
                        tmp_img = Image.fromarray(img_array)
                        ori_img = cv2.cvtColor(np.array(tmp_img), cv2.COLOR_RGB2BGR)
                        rot_img = self.rotate_img(img=ori_img, angle=360 - page.rotation)
                        img_bytes = image_to_png_bytes(Image.fromarray(cv2.cvtColor(rot_img, cv2.COLOR_BGR2RGB)))
                    else:
                        img_bytes = pix.tobytes("png")

                    # 送千问 VL 识别，返回结构化 JSON 后扁平化为文本
                    page_vlm_done = True
                    data = vlm(img_bytes)
                    if data:
                        resp += "\n" + vlm_json_to_text(data)
            # 纯扫描页兜底：可提取文本极少且没有处理过大图时，整页渲染送 VL
            if len(text.strip()) < PAGE_TEXT_MIN_CHARS and not page_vlm_done:
                page_pix = page.get_pixmap(dpi=150)
                data = vlm(page_pix.tobytes("png"))
                if data:
                    resp += "\n" + vlm_json_to_text(data)
            # 更新进度
            b_unit.update(1)
        return resp

    def rotate_img(self, img, angle):
        """
        img   --image
        angle --rotation angle
        return--rotated img
        """

        h, w = img.shape[:2]
        rotate_center = (w / 2, h / 2)
        # 获取旋转矩阵
        # 参数1为旋转中心点;
        # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
        # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
        M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
        # 计算图像新边界
        new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
        new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
        # 调整旋转矩阵以考虑平移
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
        return rotated_img


if __name__ == "__main__":
    pdf_loader = OCRPDFLoader(
        file_path="/Users/ligang/Desktop/EduRAG课堂资料/codes/integrated_qa_system/rag_qa/samples/ocr_03.pdf"
    )
    doc = pdf_loader.load()

    print(type(doc))
    print(doc)
    # text_spliter = CharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    # result = text_spliter.split_documents(doc)
    # print(len(result))
    # print(result[0])
