from dotenv import load_dotenv
import os
load_dotenv()

# ========== 1. 导入所有需要的工具 ==========
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# ========== 2. 初始化百炼向量模型（和你之前跑通的完全一致） ==========
embeddings = OpenAIEmbeddings(
    model="text-embedding-v4",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    check_embedding_ctx_length=False,
    chunk_size=10
)

# ========== 3. 创建文本切分器 ==========
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每块最多500字
    chunk_overlap=50,      # 块之间重叠50字，防止语义断裂
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],  # 中文优先切分顺序
    length_function=len    # 按字符数计算长度
)

# ========== 4. 加载本地 TXT 文档 ==========
# 项目根目录新建一个 knowledge.txt，里面放你的知识库内容
loader = TextLoader("knowledge.txt", encoding="utf-8")
documents = loader.load()  # 返回 Document 对象列表，1个txt对应1个Document

print(f"✅ 加载完成，原始文档共 {len(documents)} 个")
print(f"   原文总长度：{len(documents[0].page_content)} 字符\n")

# ========== 5. 执行文档切分 ==========
# split_documents：直接切分Document对象，自动保留原文件的元数据（来源、文件名等）
chunks = text_splitter.split_documents(documents)

print(f"✅ 切分完成，共切成 {len(chunks)} 个文本块")
print(f"   第一块长度：{len(chunks[0].page_content)} 字符")
print(f"   第一块预览：{chunks[0].page_content[:80]}...\n")

# ========== 6. 创建/加载 Chroma 向量库 ==========
vector_store = Chroma(
    collection_name="knowledge_base",  # 知识库集合名
    embedding_function=embeddings,     # 绑定百炼向量模型
    persist_directory="./chroma_db"    # 数据持久化到本地文件夹
)

# ========== 7. 把切分好的文档批量存入向量库 ==========
vector_store.add_documents(chunks)
print(f"✅ 已将 {len(chunks)} 个文本块存入 Chroma 向量库\n")

# ========== 8. 测试：语义检索验证 ==========
query = "Python 有什么内容？"
results = vector_store.similarity_search(query, k=2)

print("="*50)
print(f"检索问题：{query}")
print("="*50)
for i, doc in enumerate(results):
    print(f"\n结果 {i+1}：")
    print(f"  内容：{doc.page_content}")
    print(f"  来源：{doc.metadata.get('source', '未知')}")
