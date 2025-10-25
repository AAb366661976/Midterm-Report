import os
import sys
import logging
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.node_parser import SentenceSplitter 
from pinecone import Pinecone as PineconeClient, PodSpec

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
    
    
    # 假設知識文件在 Data/ 資料夾內
    documents = SimpleDirectoryReader(input_dir="Data").load_data()
    logger.info(f"載入了 {len(documents)} 個文件。")
    try:
        pinecone = PineconeClient(api_key=PINECONE_API_KEY)
    except Exception as e:
        logger.error(f" Pinecone 客戶端初始化失敗: {e}")
        sys.exit(1)

    # 步驟 3/5: 檢查並建立 Pinecone 索引
    if INDEX_NAME not in pinecone.list_indexes().names:
        logger.info(f"正在建立 Pinecone 索引: {INDEX_NAME}...")
        # dimension 768 匹配 text-embedding-004
        pinecone.create_index(
            name=INDEX_NAME, 
            dimension=768, 
            metric="cosine", 
            spec=PodSpec(environment=PINECONE_ENVIRONMENT)
        )
        logger.info("索引建立完成。")
    else:
        logger.info(f"Pinecone 索引 '{INDEX_NAME}' 已存在，將直接使用。")

    # 步驟 4/5: 設定 LlamaIndex 的組件
    embed_model = GeminiEmbedding(model_name="text-embedding-004") 
    vector_store = PineconeVectorStore(pinecone_index=pinecone.Index(INDEX_NAME))
    
    # 文章分割器 (Chunking)
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20) 

    # 步驟 5/5: 建立索引 (將數據上傳到 Pinecone)
    logger.info("步驟 5/5: 開始建立 LlamaIndex 索引並上傳資料至 Pinecone...")
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