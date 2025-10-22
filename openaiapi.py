import os
from google import genai
from google.genai.errors import APIError

# --- 設定參數 ---
# 1. 替換成您的 Gemini API Key
GEMINI_API_KEY = "AIzaSyDOKGxN-daR8k51hrcU1rty0Ea643SHzVs"

# 2. 替換成您要轉錄的音訊檔案路徑
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
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        print(f"--- 模型：{MODEL_NAME} ---")
        print(f"正在上傳並處理檔案: {AUDIO_FILE_PATH}...")

        # 1. 將音訊檔案上傳到 Gemini 服務
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
        
        # 5. 確保刪除上傳的檔案，避免佔用儲存空間
        client.files.delete(name=audio_file.name)
        print(f"\n[狀態] 檔案 {audio_file.name} 已刪除。")


    except APIError as e:
        print(f"\n[API 錯誤] 發生錯誤：{e}")
        print("請確認您的 API Key 是否正確，以及檔案大小是否超出免費層級的處理限制。")
    except Exception as e:
        print(f"\n[一般錯誤] 發生錯誤：{e}")

# --- 執行主要功能 ---
run_gemini_audio_transcription()