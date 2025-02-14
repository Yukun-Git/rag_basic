import pytest
import asyncio

import logging
import asyncio
from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.language_models import BaseLanguageModel
from qwen_llm import QwenClient
from agent import RouteAgent
from tools import LegalSearchTool, FundSearchTool, DirectAnswerTool
from vector_index import VectorIndexManager
from config import config

@pytest.fixture(scope="module")
def event_loop():
    """在 pytest 运行期间创建一个 event loop（避免嵌套 asyncio loops）"""
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def route_agent():
    # 初始化各组件（需要具体实现）
    llm = QwenClient(model_name=config["llm"]["model"])
    vector_manager = VectorIndexManager()
    vector_manager.load_indices()

    # 创建工具实例
    legal_tool = LegalSearchTool(vector_manager)
    fund_tool = FundSearchTool(vector_manager)
    direct_tool = DirectAnswerTool(llm)

    # 创建Agent
    agent = RouteAgent(
        llm=llm,
        legal_tool=legal_tool,
        fund_tool=fund_tool,
        direct_tool=direct_tool
    )
    return agent

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query, expected_matches",
    [
        ("金龙行业精选基金的独特优势是什么？", ["产品核心技术", "QFII"]),
        ("请简单介绍一下金龙债券基金的基金经理顾伟勇", ["CFA", "美国投资管理与研究协会"]),
        ("如何评价金龙行业精选基金的投资策略和选股标准？", ["行业选择标准", "个股选择标准"]),
        ("为什么要引入 QFII 选股策略？", ["国际化"])
    ],
)
async def test_fund_end_to_end(route_agent, query, expected_matches):
    response = await route_agent.a_invoke(query)
    assert response, f"Response should not be empty for query: {query}"
    answer = response["answer"]
    for match in expected_matches:
        assert match in answer

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query, expected_matches",
    [
        ("根据第九条 最近一期末不存在金额较大的财务性投资的理解与适用, 金额较大是指什么？", ["母公司净资产的百分之三十"]),
        ("严重损害上市公司利益、投资者合法权益、社会公共利益的判断标准是什么？", ["主观恶性程度", "国家安全"]),
        ("向特定对象发行股票的董事会决议公告后，如果本次证券发行方案出现哪些情况，应当视为本次发行方案发生重大变化？", ["增加募集资金数额", "增加新的募投项目"]),
        ("上市公司证券发行注册管理办法 所称的战略投资者，指的是什么？", ["长期持有上市公司较大比例的股份", "中国证监会的行政处罚"]),
        ("保荐机构和发行人律师应当勤勉尽责履行核查义务，对哪些事项发表明确意见？", ["投资者是否符合战略投资者的要求", "损害中小投资者合法权益"])
    ],
)
async def test_legal_end_to_end(route_agent, query, expected_matches):
    response = await route_agent.a_invoke(query)
    
    assert response, f"Response should not be empty for query: {query}"
    answer = response["answer"]
    for match in expected_matches:
        assert match in answer
