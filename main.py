import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载 .env 中的密钥
load_dotenv()

# 初始化 DeepSeek 模型
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.7,
    max_tokens=1024
)

# 发送请求并打印回答
response = llm.invoke("你好，DeepSeek！请做一个简短的自我介绍。")
print(response.content)