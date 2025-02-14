import os, yaml
import argparse
import shutil
import dashscope
import numpy as np
from typing import List
from config import config
from langchain.schema import Document
from pdf_processor.pdf import PDFProcessor
from embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings


class VectorIndexManager:

    def __init__(self):
        self.legal_embeddings = DashScopeEmbeddings(config["embeddings"]["legal"])
        self.fund_embeddings = DashScopeEmbeddings(config["embeddings"]["fund"])
    
    def load_indices(self):
        self.vector_legal = FAISS.load_local("data/embeddings/legal", self.legal_embeddings, allow_dangerous_deserialization=True)
        self.vector_fund = FAISS.load_local("data/embeddings/fund", self.fund_embeddings, allow_dangerous_deserialization=True)
        
    def build_indices(self):
        # 法律文档处理
        legal_docs = self._prepare_legal_docs("data/raw/legal")
        self._build_index(legal_docs, "legal")
        
        # 基金文档处理
        fund_docs = self._prepare_fund_docs("data/raw/fund")
        self._build_index(fund_docs, "fund")

    def _prepare_legal_docs(self, path: str) -> List[Document]:
        """处理法律文本型PDF"""
        processor = PDFProcessor()
        processor.text_parser = config["pdf"]["text"]
        
        docs = []
        for file in os.listdir(path):
            content = processor.process_pdf(os.path.join(path, file))
            # 按章节分块
            for para in content["text_blocks"]:
                docs.append(Document(
                    page_content=para,
                    metadata={"type": "text", "source": file}
                ))
        return docs

    def _prepare_fund_docs(self, path: str) -> List[Document]:
        processor = PDFProcessor()
        processor.table_parser = config["pdf"]["table"]
        
        docs = []
        for file in os.listdir(path):
            content = processor.process_pdf(os.path.join(path, file))
            
            # 文本内容（带上下文）
            for i, para in enumerate(content["text_blocks"]):
                docs.append(Document(
                    page_content=f"[Text Block {i}]: {para}",
                    metadata={"type": "text", "source": file}
                ))
            
            # 表格转描述
            for table in content["tables"]:
                docs.append(Document(
                    page_content=f"[Table]: {table}",
                    metadata={"type": "table", "source": file}
                ))
            
            # 公式处理
            for formula in content["formulas"]:
                docs.append(Document(
                    page_content=f"[Formula]: {formula}",
                    metadata={"type": "formula", "source": file}
                ))
            
            # 图表OCR结果
            for figure in content["figures"]:
                docs.append(Document(
                    page_content=f"[Figure]: {figure}",
                    metadata={"type": "figure", "source": file}
                ))
        return docs
    
    def _build_index(self, docs: List[Document], index_name:str):
        """将 List[Document] 转换成向量，并存入 FAISS 向量库"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["splitter"]["chunk_size"],
            chunk_overlap=config["splitter"]["chunk_overlap"],
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        texts = [doc.page_content for doc in splits]

        embeddings = self.fund_embeddings if index_name == "fund" else self.legal_embeddings
        import ipdb; ipdb.set_trace()
        # 这里需要传入 Embeddings 实例，而不是手动生成向量
        vector_store = FAISS.from_texts(texts, embeddings)
        vector_store.save_local(f"data/embeddings/{index_name}")
    
    def _clean_indices(self):
        path = "data/embeddings"
        if os.path.exists(path) and os.path.isdir(path):
            # 遍历目录中的所有内容
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                try:
                    # 如果是文件或符号链接，直接删除
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                        print(f"已删除文件: {item_path}")
                    # 如果是目录，递归删除
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        print(f"已删除目录: {item_path}")
                except Exception as e:
                    print(f"无法删除 {item_path}. 原因: {e}")
    
    def search(self, query: str, index_type: str) -> List[str]:
        """根据索引类型进行检索"""
        if index_type == "fund":
            query_vec = self.fund_embeddings.embed_query(query)
            return self.vector_fund.similarity_search_by_vector(
                query_vec, 
                k=config["retrieval"]["fund"]["k"],
                score_threshold=config["retrieval"]["fund"]["threshold"])
        elif index_type == "legal":    
            query_vec = self.legal_embeddings.embed_query(query)
            return self.vector_legal.similarity_search_by_vector(
                query_vec,k=config["retrieval"]["legal"]["k"],
                score_threshold=config["retrieval"]["legal"]["threshold"])
        else:
            raise ValueError(f"Unsupported index type: {index_type}")


def main_with_args():
    parser = argparse.ArgumentParser(description="执行索引操作和查询")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    # clean index 命令
    clean_parser = subparsers.add_parser('clean', help='清理索引')
    clean_parser.add_argument('index', choices=['index'], help='指定操作对象为索引')
    # build index 命令
    build_parser = subparsers.add_parser('build', help='构建索引')
    build_parser.add_argument('index', choices=['index'], help='指定操作对象为索引')
    # rebuild index 命令
    rebuild_parser = subparsers.add_parser('rebuild', help='重建索引')
    rebuild_parser.add_argument('index', choices=['index'], help='指定操作对象为索引')
    # query 命令
    query_parser = subparsers.add_parser('query', help='执行查询')
    query_parser.add_argument('data_type', choices=['fund', 'legal'],
                              help='要查询的数据类型，可选值为 fund 或 legal')
    query_parser.add_argument('query_text', type=str, help='具体的查询语句')

    args = parser.parse_args()

    index = VectorIndexManager()

    if args.command == 'clean':
        index._clean_indices()
    elif args.command == 'build':
        index.build_indices()
    elif args.command == 'rebuild':
        index._clean_indices()
        index.build_indices()
    elif args.command == 'query':
        index.load_indices()
        result = index.search(args.data_type, args.query_text)
        print(result)
    else:
        parser.print_help()


def main():
    index = VectorIndexManager()

    # index._clean_indices()
    # index.build_indices()

    index.load_indices()

    # 测试查询样例
    fund_queries = [
        "在2002年3月降息前，国泰基金当时旗下四只基金的债券仓位重不重？",
    ]
    legal_queries = [
        # "严重损害上市公司利益、投资者合法权益、社会公共利益的判断标准是什么？"
        "根据第九条 最近一期末不存在金额较大的财务性投资的理解与适用, 金额较大是指什么？"
    ]

    for query in fund_queries:
        print(f"查询: {query}")
        docs = index.search(query, "fund")
        for i, doc in enumerate(docs):
            print(f"结果{i+1}: {doc.page_content}")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
