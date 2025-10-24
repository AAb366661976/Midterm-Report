import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# 確保 GEMINI_API_KEY 已透過環境變數傳入
if "GEMINI_API_KEY" not in os.environ:
    print(" 錯誤：請先設定 GEMINI_API_KEY 環境變數。")
    exit()

# 1. 載入文件資料
documents = SimpleDirectoryReader(input_dir="Data").load_data()

# 2. 向量化與建立索引 (只執行一次，效率高)
embed_model = GeminiEmbedding(model_name="text-embedding-004") 
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

print("--------------------------------------------------")
print(" RAG 系統啟動成功！請輸入您的問題 (輸入 'exit' 退出)")
print("--------------------------------------------------")
print("提示：這是一個 RAG 系統，可以查詢關於這個專案的基本介紹")

while True:
    # 接收使用者輸入
    user_input = input("\n 您的問題: ")

    # 檢查退出指令
    if user_input.lower() in ["exit", "quit", "離開"]:
        print(" 感謝使用 RAG 系統，再見！")
        break
    
    # 執行基本的空白檢查
    if not user_input.strip():
        continue 

    # 3. 每次查詢時重新建立 Query Engine (確保穩定性)
    llm = Gemini(model="gemini-2.5-flash") 

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=3 # 告訴系統檢索最相關的 3 個文件片段
    )

    # 4. 執行 RAG 查詢
    response = query_engine.query(user_input) 

    # 5. 輸出結果
    print("\n RAG 系統回答:")
    print(response)
    print("\n 參考來源 (檢索到的證據):")
    for node in response.source_nodes:
        print(f"文本片段: {node.text[:100]}...")