""
import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessageChunk, AIMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from app.common.logger import logger
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3




load_dotenv()

# 模型,系统提示词,工具, 记忆,会话管理, 图片识别

qwen = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3.6-plus",
    temperature=0.7,
)


SYSTEM_PROMPT = """
你是一名私人厨师。收到用户提供的食材照片或清单后，请按以下流程操作：

识别和评估食材：若用户提供照片，首先辨识所有可见食材。基于食材的外观状态，评估其新鲜度与可用量，整理出一份“当前可用食材清单”。

智能食谱检索：优先调用 web_search 工具，以“可用食材清单”为核心关键词，查找可行菜谱。

多维度评估与排序：从营养价值与制作难度两个维度对检索到的候选食谱进行量化打分，并根据得分排序，制作简单且营养丰富的排名单。

结构化方案输出：把排序后的食谱整理为一份结构清晰的建议报告，要包含食谱信息、得分、推荐理由、食谱的参考图片，帮助用户快速做出决策。
请严格按照流程，优先调用 web_search 工具搜索食谱，搜索不到的情况下才能自己发挥
"""

search_tool = TavilySearch(
    max_results=5,
    topic="general",  # general, news, finance
)

connection = sqlite3.connect("../db/checkpoint.db", check_same_thread=False)
# 初始化checkpointer
checkpointer = SqliteSaver(connection)
# 自动建表
checkpointer.setup()

@tool
def web_search(query : str):
    """进行web搜索"""
    return search_tool.invoke(input=query)




agent = create_agent(
    model=qwen,
    tools=[search_tool],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

# 流式对话
async def search_recipes(prompt: str, image: str, thread_id: str):
    """调用agent搜索食谱"""
    logger.info(f"[用户]: {prompt}, image: {image}, thread_id: {thread_id}")
    try:
        # 判断是否有图片，封装不同格式的消息
        if not image or image.strip() == "":
            message = HumanMessage(content=prompt)
        else:
            message = HumanMessage(content=[
                {"type": "image", "url": image},
                {"type": "text", "text": prompt}
            ])

        # 流式调用Agent
        for chunk, metadata in agent.stream(
            {"messages": [message]},
            {"configurable": {"thread_id": thread_id}},
            stream_mode="messages"
        ):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                # 这个就像java中的迭代器一样, 可以暂停,等待处理,适用就是调用
                yield chunk.content

    except Exception as e:
        logger.error(f"\n[错误]: {str(e)}")
        yield "信息检索失败，试试看手动输入食物列表？"

# 清空会话
def clear_messages(thread_id: str):
    """清空会话"""
    logger.info(f"清空历史消息，thread_id: {thread_id}")
    checkpointer.delete_thread(thread_id)

# 查询会话历史
def get_messages(thread_id: str) -> list[dict[str, str]]:
    """获取会话历史"""
    logger.info(f"获取历史消息，thread_id: {thread_id}")

    # 根据 thread_id 查询 checkpoint
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})

    # 如果不存在，返回空列表
    if not checkpoint:
        return []

    # 安全获取 messages
    channel_values = checkpoint.get("channel_values")
    if not channel_values:
        return []

    messages = channel_values.get("messages", [])
    if not messages:
        return []

    # 转换消息格式
    result = []
    for msg in messages:
        if not msg.content:
            continue

        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})

    return result
