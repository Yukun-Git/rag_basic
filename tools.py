import asyncio
from typing import Any, Dict

from langchain.agents import initialize_agent
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.tools import Tool
from langchain_community.chat_models import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

from qwen_llm import QwenClient
from vector_index import VectorIndexManager


# 工具类：法律检索
class LegalSearchTool(Tool): 

    vector_manager: VectorIndexManager

    def __init__(self, vector_manager: VectorIndexManager):
        name: str = "legal_search"
        description: str = "适用于法律相关问题的检索工具"
        super().__init__(name=name, func=self._run, description=description, vector_manager=vector_manager)

    def _run(self, query: str) -> str:
        """执行法律检索"""
        results = self.vector_manager.search(query, index_type="legal")
        return "\n".join([doc.page_content for doc in results])

    async def _arun(self, query: str) -> str:
        """异步版本的法律检索"""
        return await asyncio.to_thread(self._run, query)


# 工具类：基金检索
class FundSearchTool(Tool): 

    vector_manager: VectorIndexManager

    def __init__(self, vector_manager: VectorIndexManager):
        name: str = "fund_search"
        description: str = "适用于基金相关问题的检索工具"
        super().__init__(name=name, func=self._run, description=description, vector_manager=vector_manager)

    def _run(self, query: str) -> str:
        """执行基金检索"""
        results = self.vector_manager.search(query, index_type="fund")
        return "\n".join([doc.page_content for doc in results])

    async def _arun(self, query: str) -> str:
        """异步版本的基金检索"""
        return await asyncio.to_thread(self._run, query)


# 工具类：直接回答问题
class DirectAnswerTool(Tool):
    
    llm: QwenClient

    def __init__(self, llm):
        name: str = "direct_answer"
        description: str = "适用于非法律/基金问题的直接回答工具"
        super().__init__(name=name, func=self._run, description=description, llm=llm)
        self.llm = llm
        
    def _run(self, query: str) -> str:
        """直接回答问题"""
        prompt = PromptTemplate(
            template="请回答以下问题：{query}",
            input_variables=["query"],
        )
        formatted_query = prompt.format(query=query)
        return self.llm.invoke(formatted_query)

    async def _arun(self, query: str) -> str:
        """异步版本的直接回答"""
        return await asyncio.to_thread(self._run, query)