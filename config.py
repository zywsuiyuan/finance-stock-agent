# config.py —— 所有配置集中管理，改配置只改这里，不碰业务代码
import os
from dotenv import load_dotenv
load_dotenv()

# ===== 模型配置 =====
MODEL_NAME = "deepseek:deepseek-v4-flash"
MODEL_TEMPERATURE = 0.2

# ===== 知识库配置 =====
KB_FILES = ["stock_knowledge.txt", "myself_knowledge.txt"]
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
CHROMA_DIR = "./finance_chroma_db"
COLLECTION_NAME = "finance_knowledge"
RETRIEVER_K = 3

# ===== 向量模型（Embedding）配置 =====
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ===== 对话记忆配置 =====
DB_FILE = "finance_chat.db"
DEFAULT_THREAD_ID = "user_001"