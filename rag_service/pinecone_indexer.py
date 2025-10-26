import os
import sys
import logging
# 引入 Pinecone 相關的異常類，用於捕獲索引已存在的錯誤
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from pinecone.exceptions import PineconeApiException 
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
    
    # 載入文件 
    try:
        documents = SimpleDirectoryReader(input_dir="Data").load_data()
    except Exception as e:
        logger.error(f"載入文件失敗，請確認 'Data' 資料夾是否存在且包含文件: {e}")
        sys.exit(1)
        
    logger.info(f"載入了 {len(documents)} 個文件。")
    
    # 初始化 Pinecone 客戶端
    try:
        pinecone = PineconeClient(api_key=PINECONE_API_KEY)
    except Exception as e:
        logger.error(f" Pinecone 客戶端初始化失敗: {e}")
        sys.exit(1)

    # 強制刪除舊索引並重建
    logger.info(f"嘗試刪除舊索引 '{INDEX_NAME}' (如果存在)...")
    try:
        # 直接嘗試刪除，如果索引不存在會拋出錯誤，由 except 區塊捕獲
        pinecone.delete_index(INDEX_NAME)
        logger.info("舊索引刪除完成。")
    except PineconeApiException as e:
        # 捕獲索引不存在時的錯誤 (通常是 404/NOT_FOUND)，忽略並繼續創建
        if "NOT_FOUND" in str(e) or e.status == 404:
            logger.info("舊索引不存在，跳過刪除。")
        else:
            logger.error(f"刪除舊索引時發生 API 錯誤: {e}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"刪除舊索引時發生一般錯誤: {e}")
        sys.exit(1)

    logger.info(f"正在建立 Pinecone 索引: {INDEX_NAME}...")
    try:
        # 建立新索引
        pinecone.create_index(
            name=INDEX_NAME, 
            dimension=768, 
            metric="cosine", 
            # 採用免費層支援的 aws-us-east-1 地區
            spec=ServerlessSpec(cloud='aws', region='us-east-1'), 
        )
        logger.info("索引建立完成。")
    except PineconeApiException as e:
        # 再次處理 409 衝突，如果發生，則表示索引在刪除後到創建前已經被其他進程重建
        if e.status == 409 and "ALREADY_EXISTS" in str(e):
            logger.info(f"Pinecone 索引 '{INDEX_NAME}' 存在，直接使用。")
        else:
            logger.error(f"建立索引時發生 API 錯誤: {e}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"建立索引時發生一般錯誤: {e}")
        sys.exit(1)


    #  設定 LlamaIndex 的組件
    embed_model = GeminiEmbedding(model_name="text-embedding-004") 
    vector_store = PineconeVectorStore(pinecone_index=pinecone.Index(INDEX_NAME))
    
    # 文章分割器 (Chunking)
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20) 

    # 建立索引 (將數據上傳到 Pinecone)
    logger.info(" 開始建立 LlamaIndex 索引並上傳資料至 Pinecone...")
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
