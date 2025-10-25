專案系統開發簡介

本專案展示了一個結合 檢索增強生成 (RAG) 系統與 語音轉錄服務 的容器化應用程式。所有服務均透過 Docker 容器化部署，以確保程式碼在任何環境下都能一致且穩定地運行。

核心服務

RAG 互動問答： 使用 LlamaIndex 和 Google Gemini 服務，透過 Pinecone 雲端向量資料庫檢索知識並回答問題。

語音轉錄： 使用 OpenAI Whisper API（通過 Gemini Key 模擬）將音訊檔（story.wav）轉換為文字。

**前置準備 (必備)**

取得 API 金鑰： 需準備您的 Gemini API Key，它同時用於 RAG 服務和語音轉錄服務。

Docker 服務： 確保您的本機 Docker 服務正在運行，且虛擬化支援已啟用。

部署與運行指南

專案提供兩種運行模式：本地文件模式（基礎）和 Pinecone 雲端模式（進階）。

模式一：使用 Docker Hub 預建映像檔 (基礎/最快速)

此模式使用 Docker Hub 上的最新映像檔，且知識庫不使用 Pinecone（僅使用映像檔內建的本地文件）。

1. 拉取映像檔

在終端機輸入指令，拉取最新的映像檔：

docker pull bella12345694/my-midterm-report:latest


2. 啟動 RAG 互動問答服務

啟用預設的 rag_app.py，請將 [YOUR_API_KEY_HERE] 替換為您的 Gemini API KEY。

docker run -it -e GEMINI_API_KEY="YOUR_API_KEY_HERE" bella12345694/my-midterm-report


3. 啟動語音轉錄服務 (audio_transcriber/openaiapi.py)

啟用 openaiapi.py 檔，會覆寫預設命令。

docker run \-e GEMINI_API_KEY="YOUR_API_KEY_HERE" \bella12345694/my-midterm-report \python audio_transcriber/openaiapi.py


模式二：使用原始碼與 Pinecone 雲端模式 (進階/專業升級)

此模式將 RAG 知識庫升級到 Pinecone 雲端向量資料庫，以提供更好的擴展性和持久性。

額外準備

需準備 Pinecone API Key 和 Environment Name。

步驟 A：建構映像檔

進入專案根目錄 (包含 Dockerfile 的資料夾)，在終端機輸入：

# 建構映像檔，並命名為 my-midterm-report
docker build -t my-midterm-report .


步驟 B：Pinecone 雲端索引器 (僅在知識庫更新時執行一次)

此步驟會執行 rag_service/pinecone_indexer.py，將知識文件分割、嵌入、並上傳到您的 Pinecone 雲端資料庫。

# 執行 pinecone_indexer.py 檔案，上傳資料到 Pinecone
docker run \-e GEMINI_API_KEY="YOUR_GEMINI_KEY" \-e PINECONE_API_KEY="YOUR_PINECONE_KEY" \-e PINECONE_ENVIRONMENT="YOUR_PINECONE_ENV" \my-midterm-report \python rag_service/pinecone_indexer.py


步驟 C：運行 RAG 互動問答服務 (Pinecone 模式)

一旦資料上傳完成，您即可運行 RAG 服務，它會連線到 Pinecone 進行問答。

# 運行容器，啟動 rag_app.py，並連線到 Pinecone
docker run -it \-e GEMINI_API_KEY="YOUR_GEMINI_KEY" \-e PINECONE_API_KEY="YOUR_PINECONE_KEY" \-e PINECONE_ENVIRONMENT="YOUR_PINECONE_ENV" \my-midterm-report


步驟 D：運行語音轉錄服務

# 執行 openaiapi.py 檔
docker run \-e GEMINI_API_KEY="YOUR_API_KEY_HERE" \my-midterm-report \python audio_transcriber/openaiapi.py
