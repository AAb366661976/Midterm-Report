#專案系統開發實務_期中專案
//不提供apikey，需自行準備

使用專案需準備:

**取得 Gemini API 金鑰**

**確保 Docker 服務正在運行，且虛擬化支援已啟用**

可選擇

1.使用DOCKERHUB 

在終端機輸入 docker pull bella12345694/my-midterm-report:latest

#啟用預設的rag_server.py， []請輸入你的Gemini API KEY

''' docker run -e GEMINI_API_KEY="[YOUR_API_KEY_HERE]" bella12345694/my-midterm-report

#啟用openaiapi.py檔，會複寫預設命令，[]請輸入你的Gemini API KEY

''' docker run \-e GEMINI_API_KEY="[YOUR_API_KEY_HERE]" \bella12345694/my-midterm-report \python audio_transcriber/openaiapi.py


2.使用原始碼

#如何使用 rag_server.py檔:

進入專案根目錄

在終端機使用

建構映像檔

''' docker build -t my-midterm-report .

運行容器，[]請輸入你的Gemini API KEY

'''  docker run -it -e GEMINI_API_KEY="[YOUR_API_KEY_HERE]" my-midterm-report

#如何使用 openaiapi.py檔:

進入專案根目錄

在終端機使用，會複寫預設命令，[]請輸入你的Gemini API KEY 

''' .docker run \-e GEMINI_API_KEY="[YOUR_API_KEY_HERE]" \ my-rag-system \python audio_transcriber/openaiapi.py
