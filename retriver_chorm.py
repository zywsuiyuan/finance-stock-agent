from dotenv import load_dotenv
import os
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# ===================== 1. 初始化百炼向量模型 =====================
embeddings = OpenAIEmbeddings(
    model="text-embedding-v4",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    check_embedding_ctx_length=False,
    chunk_size=10
)

# ===================== 2. 创建/加载 Chroma 向量库 =====================
vector_store = Chroma(
    collection_name="runoob_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"  # 数据存在本地 chroma_db 文件夹
)

# ===================== 3. 添加文档（首次运行执行，后面可以注释掉） =====================
texts = [
    "菜鸟教程（RUNOOB）是一个免费的编程学习网站，提供 HTML、CSS、JavaScript、Python 等教程。",
    "Python3 基础教程共 30 章，适合零基础入门，包含环境搭建、语法基础、面向对象等内容。",
    "HTML 基础教程共 25 章，覆盖 HTML 标签、表单、多媒体等基础知识。",
]
vector_store.add_texts(texts)
print(f"✅ 已添加 {len(texts)} 个文档到向量存储\n")

# ===================== 4. 方式一：用 similarity_search 原生方法检索 =====================
print("="*50)
print("方式一：similarity_search 原生检索")
print("="*50)
results = vector_store.similarity_search(
    "我想学 Python，有什么教程推荐？",
    k=2  # 返回最相关的2条
)

for i, doc in enumerate(results):
    print(f"\n结果 {i+1}:")
    print(f"  内容: {doc.page_content}")
    print(f"  元数据: {doc.metadata}")

# ===================== 5. 方式二：用 Retriever 标准接口检索 =====================
print("\n" + "="*50)
print("方式二：Retriever 标准接口检索")
print("="*50)

# 把向量库包装成标准检索器
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # 默认返回3条
)

# 标准调用方式
docs = retriever.invoke("HTML 学什么内容")

for doc in docs:
    print(f"- {doc.page_content[:50]}...")
