import base64
from typing import List

import fitz  # PyMuPDF
import requests

from config import config


class FormulaDetector:

    def __init__(self):
        self.baidu_key = config["baidu"]["key"]
        self.baidu_secret = config["baidu"]["secret"]

    def _get_baidu_token(self) -> str:
        """获取百度API访问令牌"""
        auth_url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.baidu_key,
            "client_secret": self.baidu_secret
        }
        response = requests.get(auth_url, params=params)
        return response.json().get("access_token")

    def _detect_formulas_baidu(self, file_path: str) -> List[str]:
        """使用百度公式识别接口"""
        formulas = []
        access_token = self._get_baidu_token()
        url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/formula?access_token={access_token}"
        
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        
        with fitz.open(file_path) as doc:
            for page_index in range(len(doc)):
                page = doc[page_index]
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    base_image = doc.extract_image(img[0])
                    image_bytes = base_image["image"]
                    
                    # 转换为base64编码
                    img_base64 = base64.b64encode(image_bytes).decode()
                    
                    # 构造请求参数
                    data = {"image": img_base64, "detect_direction": "true"}
                    
                    response = requests.post(url, headers=headers, data=data)
                    if response.status_code == 200:
                        result = response.json()
                        if 'words_result' in result:
                            # 提取所有识别结果
                            formulas.extend([item['words'] for item in result['words_result']])
        return formulas
