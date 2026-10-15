# kb.py —— 知识库构建：加载 → 切分 → 向量化 → 检索器
# 职责单一：只负责"知识库"这一件事，对外只暴露 retriever
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from config import (
    KB_FILES, CHUNK_SIZE, CHUNK_OVERLAP,
    CHROMA_DIR, COLLECTION_NAME, RETRIEVER_K,
    EMBEDDING_MODEL, EMBEDDING_BASE_URL,
)

# ===== 1. 加载知识文件 =====
documents = []
for _file in KB_FILES:
    if not os.path.exists(_file) or os.path.getsize(_file) == 0:
        print(f"[提示] 知识文件 {_file} 不存在或为空，已跳过")
        continue
    documents.extend(TextLoader(_file, encoding="utf-8").load())

if not documents:
    raise RuntimeError("没有任何可用的知识文件，请先填写 stock_knowledge.txt 或 myself_knowledge.txt")

# ===== 2. 文本切分 =====
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", "；", " "]
)
chunks = text_splitter.split_documents(documents)

# ===== 3. 初始化向量模型 =====
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=EMBEDDING_BASE_URL,
    check_embedding_ctx_length=False,
    chunk_size=10
)

# ===== 4. 加载/创建向量库（已存在就连接，首次运行才写入）=====
if os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
else:
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

# ===== 5. 检索器（公共出口，别的模块都从这里拿）=====
retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})