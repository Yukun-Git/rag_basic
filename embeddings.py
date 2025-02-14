import os
import yaml
import shutil
import dashscope
import numpy as np
from typing import List
import concurrent.futures
from config import config
from langchain.schema import Document
from pdf_processor.pdf import PDFProcessor
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
import time

class DashScopeEmbeddings(Embeddings):
    """
        支持并发调用的DashScope嵌入类
        https://help.aliyun.com/zh/model-studio/user-guide/embedding
    """
    def __init__(self, model: str):
        self.model = model
        # 从配置获取并发参数
        self.batch_size = config["embeddings"]["batch_size"]
        self.max_workers = config["embeddings"]["max_workers"]
        self.rpm_limit = config["embeddings"]["rpm"]

    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        """处理单个批次的嵌入请求"""
        try:
            response = dashscope.TextEmbedding.call(
                model=self.model,
                input=batch,
                dimension=1024,
                output_type="dense"
            )
            if response.status_code == 200:
                return [resp['embedding'] for resp in response.output['embeddings']]
            else:
                print(f"请求失败: {response.code} - {response.message}")
                return []
        except Exception as e:
            print(f"处理批次时发生异常: {str(e)}")
            return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """并发批量生成文本嵌入"""
        # 计算安全间隔避免触发限速
        delay = 60 / self.rpm_limit if self.rpm_limit > 0 else 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            results = []
            
            # 创建批次任务
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                futures.append(executor.submit(self._embed_batch, batch))
                
                # 添加限速延迟
                if self.rpm_limit > 0:
                    time.sleep(delay)
            
            # 收集结果
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_result = future.result()
                    results.extend(batch_result)
                    print(f"成功处理批次，累计完成 {len(results)} 项")
                except Exception as e:
                    print(f"批次处理异常: {str(e)}")
        
        return results

    def embed_query(self, text: str) -> List[float]:
        """生成查询的嵌入（保持同步调用）"""
        response = dashscope.TextEmbedding.call(
            model=self.model,
            input=text,
            dimension=1024,
            output_type="dense"
        )
        return response.output['embeddings'][0]['embedding']


def test_embedding_model():
    emb = DashScopeEmbeddings(config["embeddings"]["default"])

    # 示例文本（从你的数据库取出一段）
    example_text = "严重损害上市公司利益、投资者合法权益、社会公共利益的判断标准是什么？"
    example_embedding = emb.embed_query(example_text)

    # 查询文本
    query_text = "什么情况回呗判定为严重损害上市公司利益、投资者合法权益、社会公共利益？"
    query_embedding = emb.embed_query(query_text)

    # 计算余弦相似度
    cos_sim = np.dot(example_embedding, query_embedding) / (
        np.linalg.norm(example_embedding) * np.linalg.norm(query_embedding)
    )
    print("Cosine Similarity:", cos_sim)


if __name__ == '__main__':
    test_embedding_model()
