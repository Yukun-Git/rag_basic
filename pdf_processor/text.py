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


def legal_hierarchical_split(text: str) -> List[Dict]:
    patterns = {
        # 一级标题：匹配 "一、" 到 "十、"
        "level1": r'(\s*[一二三四五六七八九十]+、\s*[\u4e00-\u9fa5（]+)',
        
        # 二级标题：匹配 "（一）" 类
        "level2": r'(\s*（[一二三四五六七八九十]+）)',
        
        # 三级标题：匹配 "1." 类
        "level3": r'(\s*(?:\d+\.|•)\s*)'
    }

    # 使用复合分隔符切割文本
    split_regex = re.compile(
        f'({"|".join(patterns.values())})',
        flags=re.MULTILINE
    )

    segments = split_regex.split(text)
    # 过滤掉空字符串和无效段落
    segments = [segment.strip() for segment in segments if segment]

    chunks = []
    current_chunk = ""
    current_level = 0

    for seg in segments:
        if not seg:
            continue
            
        # 判断层级
        if re.fullmatch(patterns["level1"], seg):
            level = 1
        elif re.fullmatch(patterns["level2"], seg):
            level = 2
        elif re.fullmatch(patterns["level3"], seg):
            level = 3
        else:
            level = 0
            
        if level > 0:
            if current_chunk:
                chunks.append({"text": current_chunk, "level": current_level})
            current_chunk = seg
            current_level = level
        else:
            current_chunk += seg
            
    if current_chunk:
        chunks.append({"text": current_chunk, "level": current_level})
        
    return chunks

def visualize_chunks(chunks):
    """
    仅供debug使用，可视化输出chunk的层级信息
    """
    for idx, chunk in enumerate(chunks):
        print(f"Chunk {idx} [Level {chunk["level"]}]")
        print(chunk["text"][:500] + "...")
        print("-" * 80)

def post_process_chunks(chunks: List[Dict]) -> List[str]:
    """
    基于栈的层级合并算法，时间复杂度O(n)
    输入示例：
    [
        {"text": "一、关于第九条...", "level": 1},
        {"text": "（一）财务性投资包括...", "level": 2},
        {"text": "1. 投资类金融业务...", "level": 3},
        {"text": "（二）围绕产业链...", "level": 2},
        {"text": "二、关于第十条...", "level": 1}
    ]
    """

    merged = []
    stack = []  # 保存当前处理中的父级块
    
    for chunk in chunks:
        current_level = chunk["level"]
        current_text = chunk["text"]
        
        # 弹出已完成处理的祖先层级
        while stack and stack[-1]["level"] >= current_level:
            closed_chunk = stack.pop()
            merged.append(closed_chunk)
            
        # 将当前块推入栈或合并到父级
        if not stack or current_level == 1:
            # 新的一级块
            stack.append({"text": current_text, "level": current_level})
        else:
            # 合并到最近的父级
            parent = stack[-1]
            parent["text"] += "\n" + current_text
            
    # 处理栈中剩余块
    while stack:
        merged.append(stack.pop())
    
    visualize_chunks(merged)
    return merged

def split_paragraphs(text: str) -> List[str]:
    """优化段落分割算法"""
    # 基于标点+缩进+字体变化的段落检测
    paragraphs = []
    current_para = []
    lines = text.split('\n')

    for line in lines:
        if is_new_paragraph(line, current_para):
            if current_para:
                paragraphs.append('\n'.join(current_para))
                current_para = []
        current_para.append(line.strip())
    if current_para:
        paragraphs.append('\n'.join(current_para))

    return paragraphs

def is_new_paragraph(line: str, current_para: List[str]) -> bool:
    """判断是否为新段落"""
    # 简单示例：如果行是空的或者以缩进开始，则认为是新段落
    if not line:
        return True
    if line.startswith((' ', '\t')) and current_para:
        return True
    return False

def clean_text(text):
    # 定义正则表达式，匹配文本中的格式字符
    pattern = r'项目符号\+.*?设置格式\[c\]:?'
    # 使用 re.sub() 函数替换掉所有匹配的字符
    text = re.sub(pattern, '', text)

    # 合并被错误分割的换行（保留自然段落分隔）
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # 替换单换行为空格
    text = re.sub(r'\n{2,}', '\n\n', text)  # 标准化多换行为双换行
    # 移除制表符
    text = text.replace('\t', ' ')
    # 合并法律条款编号
    text = re.sub(r'（\s*(\d+)\s*）', r'（\1）', text)  # 标准化"（ 一 ）"为"（一）"

    # 修复被分割的条款引用
    text = re.sub(r'《\s*([^》]+)\s*》', r'《\1》', text)  # 合并《 XXX 》为《XXX》
    # 修复条款编号与内容的断裂
    text = re.sub(r'（(\d+)）\s+', r'（\1）', text)
    # 标准化金额表述
    text = re.sub(r'(\d+)\s*%\s*', r'\1%', text)

    # 合并被错误断行的标题（如"证 券"→"证券"）
    text = re.sub(r'(\S)\s*\n\s*(\S)', r'\1\2', text)
    
    # 标准化中文标点间距
    text = re.sub(r'([。；）])\s+', r'\1', text)
    
    # 修复跨行标题（如"——证\n券期货"→"——证券期货"）
    text = re.sub(r'([^\s-])\s*\n\s*([^\s-])', r'\1\2', text)

    return text

class LegalTextSplitter(TextSplitter):

    def split_text(self, text: str) -> List[str]:
        chunks = legal_hierarchical_split(text)
        processed = post_process_chunks(chunks)
        processed = [chunk for chunk in processed if len(chunk["text"])>30]
        return processed
