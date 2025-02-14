from typing import Dict, List

import camelot
import fitz  # PyMuPDF
import pdfplumber
from pdfminer.high_level import extract_text

from .formula import FormulaDetector
from .figure import extract_figures_page_ocr
from .text import clean_text, split_paragraphs


class PDFProcessor:
    """PDF处理器"""

    def __init__(self):
        self.text_parser = "fitz"
        self.table_parser = "camelot"
        self.formula_detector = "baidu"

    def process_pdf(self, file_path: str) -> Dict:
        """返回结构化内容：文本/表格/公式/图表"""
        content = {
            "text_blocks": [],
            "tables": [],
            "formulas": [],
            "figures": []
        }
        FormulaDetector()

        # 文本提取（优化段落检测）
        if self.text_parser == "pdfminer":
            full_text = extract_text(file_path)
            content["text_blocks"] = split_paragraphs(full_text)
        elif self.text_parser == "fitz":
            doc = fitz.open(file_path)
            raw_text = ""
            for page in doc:
                raw_text += page.get_text("text")
            cleaned_text = clean_text(raw_text)
            # text_splitter = LegalTextSplitter()
            # final_chunks = text_splitter.split_text(cleaned_text)
            content["text_blocks"] = [cleaned_text]
        elif self.text_parser == "pymupdf":
            with fitz.open(file_path) as doc:
                text = []
                for page in doc:
                    text.extend(page.get_text("blocks"))
                content["text_blocks"] = self._process_blocks(text)

        # 表格提取
        if self.table_parser == "camelot":
            tables = camelot.read_pdf(file_path, pages='all')
            content["tables"] = [table.df.to_markdown() for table in tables]
        elif self.table_parser == "pdfplumber":
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    content["tables"].extend(page.extract_tables())

        # # 公式检测
        # if self.formula_detector == "baidu":
        #     content["formulas"] = formula._detect_formulas_baidu(file_path)

        # 图表处理
        content["figures"] = extract_figures_page_ocr(file_path)

        return content

    def _process_blocks(self, blocks: List) -> List[str]:
        """
        处理 PyMuPDF 提取的文本块
        经测试，对于法律文件，最长的paragraph长度为383，平均长度60
        """
        paragraphs = []
        current_para = []
        for block in blocks:
            block_text = block[4].strip()  # block[4] 包含文本内容
            if self._is_new_paragraph(block_text, current_para):
                if current_para:
                    paragraphs.append('\n'.join(current_para))
                    current_para = []
            current_para.append(block_text)
        if current_para:
            paragraphs.append('\n'.join(current_para))

        return paragraphs

if __name__ == "__main__":
    pdf = PDFProcessor()
    # result = pdf.process_pdf("../data/raw/legal/opinios_no_18.pdf")
    # print(result)

    result = pdf.process_pdf("../data/raw/fund/guotai.pdf")
    print(result)
