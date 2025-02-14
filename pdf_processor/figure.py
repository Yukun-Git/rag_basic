from pdfminer.high_level import extract_text
from PIL import Image
from typing import Dict, List
import camelot
import tabula
import io, os, re
import pytesseract
import fitz  # PyMuPDF
import pandas as pd
import requests
import dashscope
import base64
import requests
from typing import List
import fitz
from PIL import Image
import io, cv2
import numpy as np
import pdfplumber
from langchain.text_splitter import TextSplitter


def extract_figures_page_ocr(file_path: str) -> List[str]:
    """把整个页面图片化，然后进行OCR处理。生成结果里包含了页面原有的文本信息。"""
    figure_texts = []
    with fitz.open(file_path) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            # 提高分辨率
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # 灰度化
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 二值化
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            # 转换回 PIL 图像
            image = Image.fromarray(binary)
            # 设置语言
            text = pytesseract.image_to_string(image, lang='chi_sim')  # 根据实际情况修改语言
            figure_texts.append(text)

    return figure_texts

def extract_figures_by_pytesseract(file_path: str) -> List[str]:
    """WARNING by 瑜琨: 图片的识别精度非常差，方法需要继续优化"""
    figure_texts = []
    with fitz.open(file_path) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                base_image = doc.extract_image(img[0])
                image = Image.open(io.BytesIO(base_image["image"]))
                text = pytesseract.image_to_string(image)
                figure_texts.append(text)
    return figure_texts
