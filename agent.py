import asyncio
import logging
from typing import Literal, Optional, TypedDict

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

from config import config
from qwen_llm import QwenClient
from tools import DirectAnswerTool, FundSearchTool, LegalSearchTool
from vector_index import VectorIndexManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 状态类型定义
class AgentState(TypedDict):
    query: str
    tool_type: Optional[Literal["legal", "fund", "other"]]
    retrieved_docs: str
    retry_count: int
    answer: Optional[str]

class RouteAgent:
    def __init__(
        self,
        llm: BaseLanguageModel,
        legal_tool: LegalSearchTool,
        fund_tool: FundSearchTool,
        direct_tool: DirectAnswerTool,
        max_retries: int = 3
    ):
        self.llm = llm
        self.legal_tool = legal_tool
        self.fund_tool = fund_tool
        self.direct_tool = direct_tool
        self.max_retries = max_retries
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        # 初始化状态图
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("classify", self.classify_node())
        workflow.add_node("retrieve", self.retrieve_node())
        workflow.add_node("evaluate", self.evaluate_node())
        workflow.add_node("other", self.other_node())
        workflow.add_node("generate_answer", self.generate_answer_node())

        # 设置边和条件
        workflow.set_entry_point("classify")

        # 分类后的路由
        workflow.add_conditional_edges(
            "classify",
            self.route_by_type,
            {
                "legal": "retrieve",
                "fund": "retrieve",
                "other": "other"
            }
        )

        # 检索评估流程
        workflow.add_edge("retrieve", "evaluate")
        workflow.add_conditional_edges(
            "evaluate",
            self.should_retry,
            {
                True: "retrieve",  # 继续检索
                False: "generate_answer"  # 生成答案
            }
        )

        workflow.add_edge("other", "generate_answer")
        workflow.add_edge("generate_answer", END)
        return workflow.compile()

    def classify_node(self):
        prompt = ChatPromptTemplate.from_template(
            """判断问题类型并返回对应类型标识：
            法律问题 -> legal
            基金问题 -> fund
            其他问题 -> other
            
            问题：{query}
            只需返回单个类型标识，不要包含其他内容。"""
        )

        def parse_output(text: str) -> Literal["legal", "fund", "other"]:
            text = text.strip().lower()
            return text if text in ["legal", "fund"] else "other"

        def node_function(state: AgentState):
            logging.info(f"Entering node: classify")
            logging.debug(f"Classify Input state: {state}")
            output = {
                "tool_type": parse_output(
                    self.llm.invoke(prompt.format(query=state["query"])))
            }
            logging.info(f"Output state: {output}")
            return output

        return RunnableLambda(node_function)
    
    def other_node(self):
        async def node_function(state: AgentState):
            logging.info(f"Entering node: other")
            logging.debug(f"Other Input state: {state}")
            legal_docs = await self.legal_tool._arun(state["query"])
            if legal_docs:
                logging.info(f"Found legal items for other type query")
                output = {
                    "retrieved_docs": f"{state.get('retrieved_docs', '')}\n{legal_docs}".strip()
                }
                logging.info(f"Other Output state: {output}")
                return output
            fund_docs = await self.fund_tool._arun(state["query"])
            if fund_docs:
                logging.info(f"Found fund items for other type query")
                output = {
                    "retrieved_docs": f"{state.get('retrieved_docs', '')}\n{fund_docs}".strip()
                }
                logging.info(f"Other Output state: {output}")
                return output

            logging.info(f"Found nothing for other type query, will jump to llm direct answer")
            output = {
                "tool_type": "other"
            }
            logging.info(f"Other Output state: {output}")
            return output

        return RunnableLambda(node_function)

    def retrieve_node(self):
        async def _retrieve(state: AgentState):
            logging.info(f"Entering node: retrieve")
            logging.debug(f"Retrieve Input state: {state}")
            tool_type = state["tool_type"]
            query = state["query"]

            # 执行检索
            if tool_type == "legal":
                docs = await self.legal_tool._arun(query)
            else:
                docs = await self.fund_tool._arun(query)

            output = {
                "retrieved_docs": f"{state.get('retrieved_docs', '')}\n{docs}".strip(),
                "retry_count": state.get("retry_count", 0) + 1
            }
            logging.debug(f"Output state: {output}")
            return output

        return RunnableLambda(_retrieve)

    def evaluate_node(self):
        prompt = ChatPromptTemplate.from_template(
            """根据以下检索内容和原始问题，判断信息是否足够回答问题：
            
            原始问题：{query}
            检索内容：{docs}
            
            请严格按格式返回：是 或 否"""
        )

        async def _evaluate(state: AgentState):
            logging.info(f"Entering node: evaluate")
            logging.debug(f"Evaluate Input state: {state}")
            response = await self.llm.ainvoke(prompt.format(
                query=state["query"],
                docs=state.get("retrieved_docs", "")
            ))
            output = {"sufficient": "是" in response}
            logging.info(f"Output state: {output}")
            return output

        return RunnableLambda(_evaluate)

    def generate_answer_node(self):
        async def _generate(state: AgentState):
            logging.info(f"Entering node: generate_answer")
            logging.debug(f"GA Input state: {state}")
            if state["tool_type"] == "other":
                output = {"answer": await self.direct_tool._arun(state["query"])}
            else:
                # 使用检索内容生成答案
                prompt = ChatPromptTemplate.from_template(
                    """基于以下信息回答问题：
                    {docs}
                    
                    问题：{query}
                    请给出专业、完整的回答："""
                )
                result = await self.llm.ainvoke(prompt.format(
                    docs=state["retrieved_docs"],
                    query=state["query"]
                ))
                output = {"answer": result}
            logging.debug(f"Output state: {output}")
            return output

        return RunnableLambda(_generate)

    def route_by_type(self, state: AgentState):
        logging.info(f"Routing based on type. Current state: {state}")
        output = state["tool_type"]
        logging.info(f"Routing output: {output}")
        return output

    def should_retry(self, state: AgentState):
        logging.info(f"Checking if should retry.")
        logging.debug(f"Current state for retry checking: {state}")
        output = state.get("sufficient", False) is False and \
                 state.get("retry_count", 0) < self.max_retries
        logging.info(f"Retry decision: {output}")
        return output

    async def a_invoke(self, query: str):
        logging.info(f"Starting async workflow with query: {query}")
        return await self.workflow.ainvoke({
            "query": query,
            "retrieved_docs": "",
            "retry_count": 0,
            "tool_type": None,
            "answer": None
        })

# 初始化示例
if __name__ == "__main__":
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

    legal_query_list = [
        "向特定对象发行股票的董事会决议公告后，如果本次证券发行方案出现哪些情况，应当视为本次发行方案发生重大变化？"
    ]

    fund_query_list = [
        "为什么要引入 QFII 选股策略？",
    ]

    # 使用示例
    async def test():
        for query in fund_query_list:
            response = await agent.a_invoke(query)
            print(response["answer"])

    asyncio.run(test())
