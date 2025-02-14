import asyncio
import os

import dashscope
from langchain.schema.runnable import Runnable
from langchain.tools import Tool

# 初始化阿里云配置
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

class QwenClient(Runnable):
    """千问大模型客户端封装"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def _async_call(self, prompt: str, **kwargs) -> str:
        """异步调用千问 API"""
        def _sync_call():
            response = dashscope.Generation.call(
                model=self.model_name,
                prompt=prompt,
                **kwargs
            )
            return response.output['text']

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_call)

    def __call__(self, prompt: str, **kwargs) -> str:
        """同步调用（封装异步调用）"""
        return asyncio.run(self._async_call(prompt, **kwargs))
    
    def __ror__(self, other):
        """支持管道操作"""
        from langchain.schema.runnable import RunnableSequence
        return RunnableSequence([other, self])

    async def predict(self, prompt: str, **kwargs) -> str:
        """异步预测方法"""
        return await self._async_call(prompt, **kwargs)

    def invoke(self, input, config=None):
        """兼容 LangChain 的 invoke 方法"""
        if isinstance(input, str):
            return self(input)
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

    def build_tools(self):
        """提供 LangGraph 可注册的工具"""
        return [
            Tool(
                name="qwen_llm",
                func=self.__call__,
                description="调用千问大模型进行自然语言处理",
            )
        ]
    
    def bind_tools(self, tools):
        """绑定外部工具，支持 LangGraph"""
        self.tools = tools
        return self
