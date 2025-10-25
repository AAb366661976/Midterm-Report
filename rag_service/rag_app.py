import os
import sys
# 解決 Docker 環境中文輸入問題（如果需要）
try:
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

from llama_index.core import VectorStoreIndex
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient

# --------------------------------------------------
# 1. API 金鑰和 Pinecone 設定
# --------------------------------------------------
# 從環境變數讀取 Pinecone 資訊，移除 YOUR_FALLBACK_KEY 避免誤用
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY") 
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
INDEX_NAME = "midterm-rag-index"

# 確保所有必要的金鑰都有設定
if "GEMINI_API_KEY" not in os.environ:
    print("❌ 錯誤：請先設定 GEMINI_API_KEY 環境變數。")
    sys.exit(1)
if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    print("❌ 錯誤：請務必透過環境變數設定 PINECONE_API_KEY 和 PINECONE_ENVIRONMENT。")
    sys.exit(1)



# A. 初始化 Pinecone 客戶端 (解決 NameError)
try:
    pinecone = PineconeClient(api_key=PINECONE_API_KEY)
except Exception as e:
    print(f"❌ Pinecone 客戶端初始化失敗: {e}")
    sys.exit(1)

# B. 從 Pinecone 讀取索引
vector_store = PineconeVectorStore(
    pinecone_index=pinecone.Index(INDEX_NAME)
)
embed_model = GeminiEmbedding(model_name="text-embedding-004") 
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model
)

# C. 設定 LLM 模型與查詢引擎 (移出迴圈，只初始化一次)
llm = Gemini(model="gemini-2.5-flash")
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3 
)


print("--------------------------------------------------")
print(" RAG 系統 (Pinecone 模式) 啟動成功！請輸入您的問題 (輸入 'exit' 退出)")
print("--------------------------------------------------")
print("提示：請先確認 pinecone_indexer.py 已成功運行並上傳資料。")

while True:
    # 接收使用者輸入
    user_input = input("\n 您的問題: ")

    # 檢查退出指令
    if user_input.lower() in ["exit", "quit", "離開"]:
        print(" 感謝使用 RAG 系統，再見！")
        break
    
    # 執行基本的空白檢查
    cleaned_input = user_input.strip()
    if not cleaned_input:
        continue 

    # 執行 RAG 查詢 (直接使用已建立的 query_engine)
    response = query_engine.query(cleaned_input) 

    # 輸出結果
    print("\n RAG 系統回答:")
    print(response)
    print("\n 參考來源 (檢索到的證據):")
    for node in response.source_nodes:
        # 使用 strip() 確保文本片段乾淨
        print(f"文本片段: {node.text.strip()[:100]}...")