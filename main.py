import os

from langchain_openai import ChatOpenAI


def main() -> None:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY environment variable.")

    llm = ChatOpenAI(
        model="qwen3.6-plus",
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    response = llm.invoke("你是谁？")
    print(response.content)


if __name__ == "__main__":
    main()
