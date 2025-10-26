import os
import sys
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from pinecone import Pinecone as PineconeClient

# --- 配置 ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", None) 
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", None)
INDEX_NAME = "midterm-rag-index" 

# 檢查必要的環境變數
if not os.environ.get("GEMINI_API_KEY"):
    print("錯誤：請先設定 GEMINI_API_KEY 環境變數。")
    sys.exit(1)

def setup_rag_system():
    """初始化 LlamaIndex 設定並連線到 Pinecone/本地向量資料庫。"""
    
    # 步驟 1: 設定 LLM 和 Embedding Model
    llm = Gemini(model="gemini-2.5-flash")
    embed_model = GeminiEmbedding(model_name="text-embedding-004") 

    # 將設定應用到 LlamaIndex 的全域環境
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512

    print("嘗試連線到 Pinecone...")
    
    # 嘗試 Pinecone 模式
    if PINECONE_API_KEY and PINECONE_ENVIRONMENT:
        try:
            pinecone = PineconeClient(api_key=PINECONE_API_KEY)
            
            if INDEX_NAME not in pinecone.list_indexes():
                print(f"警告：Pinecone 索引 '{INDEX_NAME}' 不存在或無法訪問，切換到本地文件模式。")
                return setup_local_rag()

            vector_store = PineconeVectorStore(pinecone_index=pinecone.Index(INDEX_NAME))
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            
            print("INFO: 已成功連線 Pinecone 索引並建立 RAG 引擎。")
            
            # 建立查詢引擎
            query_engine = index.as_query_engine(
                llm=llm,
                similarity_top_k=3 # 檢索最相似的 3 個文檔塊
            )
            return query_engine

        except Exception as e:
            print(f"錯誤：初始化 Pinecone RAG 系統失敗 ({e})，切換到本地文件模式。")
            return setup_local_rag()

    # 如果沒有 Pinecone 參數，則直接使用本地模式
    else:
        print("警告：缺少 Pinecone 環境變數，直接使用本地文件模式。")
        return setup_local_rag()

def setup_local_rag():
    """初始化基於本地文件（Data/）的 RAG 系統。"""
    try:
        documents = SimpleDirectoryReader(input_dir="Data").load_data()
        index = VectorStoreIndex.from_documents(documents, show_progress=False)
        
        print("INFO: 已成功從本地文件建立 RAG 引擎。")
        
        query_engine = index.as_query_engine(
            llm=Settings.llm,
            similarity_top_k=3
        )
        return query_engine
    except Exception as e:
        print(f"致命錯誤：本地文件 RAG 系統初始化失敗: {e}")
        print("請確認 'Data' 資料夾是否存在且包含文件。")
        sys.exit(1)


def main():
    """主函數，啟動 RAG 互動問答循環。"""
    query_engine = setup_rag_system()

    print("-" * 50)
    print(" RAG 系統啟動成功！請輸入您的問題 (輸入 'exit' 退出)")
    print("-" * 50)
    
    # 判斷當前模式
    mode = "Pinecone 模式" if PINECONE_API_KEY else "本地模式"
    print(f"當前模式: {mode}")
    print("提示：如果 Pinecone 模式失敗，請嘗試使用本地模式運行 (移除 Pinecone 參數)。")

    while True:
        try:
            user_input = input("\n 您的問題: ")
        except UnicodeDecodeError:
            print("\n錯誤：中文輸入編碼失敗。請確保在運行 docker 時加上 -e PYTHONIOENCODING=utf-8 參數。")
            continue

        if user_input.lower() == 'exit':
            break

        if not user_input.strip():
            continue

        try:
            # 執行查詢
            response = query_engine.query(user_input)

            print("\n RAG 系統回答:")
            # 增加除錯資訊
            if not response.response or not response.source_nodes:
                print("Empty Response (檢索失敗)")
                print("=== 除錯資訊 ===")
                print(f"模型回覆內容是否為空: {not bool(response.response)}")
                print(f"檢索到的文件數: {len(response.source_nodes)}")
                print("如果檢索到的文件數為 0，請檢查 Pinecone 中的 Vector Count。")
            else:
                print(response.response)

            # 顯示參考來源
            print("\n 參考來源 (檢索到的證據):")
            if response.source_nodes:
                for i, node in enumerate(response.source_nodes):
                    source_file = node.metadata.get('file_name', '未知文件')
                    print(f"  [{i+1}] 來源: {source_file} (相似度: {node.score:.4f})")
                    print(f"      內容: {node.text[:150]}...")
            else:
                print("  未檢索到任何相關證據。")

        except Exception as e:
            print(f"\n查詢時發生錯誤: {e}")

if __name__ == "__main__":
    main()
