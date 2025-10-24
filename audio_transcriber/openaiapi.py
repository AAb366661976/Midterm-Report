import os
import sys
from google import genai
from google.genai.errors import APIError

if "GEMINI_API_KEY" not in os.environ:
    # 如果找不到環境變數，提供提示並終止程式
    print(" 錯誤：找不到 GEMINI_API_KEY 環境變數。請使用 docker run -e 參數傳遞金鑰。")
    sys.exit(1)

#放入要轉錄的音訊檔案路徑(要注意檔案格式與大小)
AUDIO_FILE_PATH = "story.wav"  

# 3. 定義轉錄指令
transcribe_prompt = "請將此音訊檔案的內容轉錄為繁體中文文字稿，並檢查語法和標點符號。"

# 4. 選擇模型 (必須是支援多模態的模型)
MODEL_NAME = "gemini-2.5-flash"

def run_gemini_audio_transcription():
    """
    呼叫 Google Gemini API 進行語音轉文字
    """
    if not os.path.exists(AUDIO_FILE_PATH):
        print(f"錯誤：找不到檔案 {AUDIO_FILE_PATH}。請確認檔案名稱和路徑是否正確。")
        return

    try:
        # 初始化客戶端
        client = genai.Client()
        
        print(f"--- 模型：{MODEL_NAME} ---")
        print(f"正在上傳並處理檔案: {AUDIO_FILE_PATH}...")

        
        # 這會返回一個 File 物件
        audio_file = client.files.upload(file=AUDIO_FILE_PATH)

        # 2. 組合內容：將文字指令和上傳的檔案一起傳入
        contents = [
            audio_file,        # 音訊檔案
            transcribe_prompt  # 轉錄指令 (Prompt)
        ]

        # 3. 呼叫 API 進行內容生成
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents
        )

        # 4. 取得 AI 的回覆並印出
        ai_response = response.text
        
        print("\n--- 轉錄指令 ---")
        print(transcribe_prompt)
        print("\n--- AI 轉錄結果 ---")
        print(ai_response)
        
    except APIError as e:
        print(f"\n[API 錯誤] 發生錯誤：{e}")
    except Exception as e:
        print(f"\n[一般錯誤] 發生錯誤：{e}")

# --- 執行主要功能 --
run_gemini_audio_transcription()