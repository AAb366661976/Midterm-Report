import os
import sys
import logging
# 引入 Pinecone 相關的異常類，用於捕獲索引已存在的錯誤
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from pinecone.exceptions import PineconeApiException # <--- 這是關鍵的匯入，用於捕捉 API 錯誤
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.node_parser import SentenceSplitter 

# 啟用日誌記錄，便於觀察 Pinecone 處理過程
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "YOUR_FALLBACK_KEY") 
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "YOUR_FALLBACK_ENV")
INDEX_NAME = "midterm-rag-index" 


# 確保 API 金鑰已設定
if "GEMINI_API_KEY" not in os.environ:
    print(" 錯誤：請先設定 GEMINI_API_KEY 環境變數。")
    sys.exit(1)

def index_data_to_pinecone():
    """載入文件，分割，並將向量儲存到 Pinecone。"""
    
    # 步驟 1/5: 載入文件
    documents = SimpleDirectoryReader(input_dir="Data").load_data()
    logger.info(f"載入了 {len(documents)} 個文件。")
    
    # 步驟 2/5: 初始化 Pinecone 客戶端
    try:
        pinecone = PineconeClient(api_key=PINECONE_API_KEY)
    except Exception as e:
        logger.error(f" Pinecone 客戶端初始化失敗: {e}")
        sys.exit(1)

    # 步驟 3/5: 檢查並建立 Pinecone 索引 (如果已存在，捕獲 409 錯誤)
    # 為了應對 list_indexes() 可能的延遲，我們用 try-except 來確保索引建立不會造成程式中斷
    if INDEX_NAME not in pinecone.list_indexes():
        logger.info(f"正在嘗試建立 Pinecone 索引: {INDEX_NAME}...")
        try:
            pinecone.create_index(
                name=INDEX_NAME, 
                dimension=768, 
                metric="cosine", 
                # 採用免費層支援的 aws-us-east-1 地區
                spec=ServerlessSpec(cloud='aws', region='us-east-1'), 
            )
            logger.info("索引建立完成。")
        except PineconeApiException as e:
            # 捕獲 409 Conflict (資源已存在) 錯誤
            if e.status == 409 and "ALREADY_EXISTS" in str(e):
                logger.info(f"Pinecone 索引 '{INDEX_NAME}' 已存在，將直接使用（已捕獲到 409 衝突錯誤）。")
            else:
                logger.error(f"建立索引時發生未預期的 API 錯誤: {e}")
                sys.exit(1)
    else:
        logger.info(f"Pinecone 索引 '{INDEX_NAME}' 已存在，將直接使用。")

    # 步驟 4/5: 設定 LlamaIndex 的組件
    embed_model = GeminiEmbedding(model_name="text-embedding-004") 
    vector_store = PineconeVectorStore(pinecone_index=pinecone.Index(INDEX_NAME))
    
    # 文章分割器 (Chunking)
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20) 

    # 步驟 5/5: 建立索引 (將數據上傳到 Pinecone)
    logger.info("步驟 5/5: 開始建立 LlamaIndex 索引並上傳資料至 Pinecone...")
    # 由於索引已存在或剛剛建立，這裡將會執行數據上傳操作
    VectorStoreIndex.from_documents(
        documents, 
        embed_model=embed_model, 
        vector_store=vector_store, 
        text_splitter=text_splitter,
        show_progress=True
    )
    logger.info(" 數據上傳至 Pinecone 完成！")


if __name__ == "__main__":
    index_data_to_pinecone()
