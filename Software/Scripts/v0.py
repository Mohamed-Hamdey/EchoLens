import sys
import json
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import os

# Use absolute paths
ARABIC_MODEL_PATH = r"E:\Desktop\EchoLens\Software\Models\vosk-model-ar-0.22-linto-1.1.0"
ENGLISH_MODEL_PATH = r"E:\Desktop\EchoLens\Software\Models\vosk-model-small-en-us-0.15"

# Enhanced path checking
def check_model_path(path):
    if not os.path.exists(path):
        print(f"Error: Model path does not exist: {path}")
        return False
    
    # Check for essential model files
    expected_files = ['am', 'conf', 'ivector']
    missing_files = [f for f in expected_files if not os.path.exists(os.path.join(path, f))]
    
    if missing_files:
        print(f"Error: Model at {path} is missing essential files: {missing_files}")
        return False
    
    print(f"Model path validated: {path}")
    return True

# Check models before loading
arabic_valid = check_model_path(ARABIC_MODEL_PATH)
english_valid = check_model_path(ENGLISH_MODEL_PATH)

# Try loading models with error handling
try:
    if arabic_valid:
        print("Loading Arabic model...")
        arabic_model = Model(ARABIC_MODEL_PATH)
        print("Arabic model loaded successfully")
    else:
        print("Skipping Arabic model load due to path validation failure")
except Exception as e:
    print(f"Failed to load Arabic model: {str(e)}")
    arabic_model = None

try:
    if english_valid:
        print("Loading English model...")
        english_model = Model(ENGLISH_MODEL_PATH)
        print("English model loaded successfully")
    else:
        print("Skipping English model load due to path validation failure")
except Exception as e:
    print(f"Failed to load English model: {str(e)}")
    english_model = None

def recognize_speech(language):
    if language == "arabic" and arabic_model is None:
        print("Arabic model not available.")
        return ""
    if language == "english" and english_model is None:
        print("English model not available.")
        return ""
    
    q = queue.Queue()
    
    # Select the model based on language
    if language == "arabic":
        rec = KaldiRecognizer(arabic_model, 16000)
        print("Listening for Arabic...")
    else:
        rec = KaldiRecognizer(english_model, 16000)
        print("Listening for English...")
    
    rec.SetWords(True)

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(bytes(indata))

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                print("You said:", result.get("text", ""))
                return result.get("text", "")

if __name__ == "__main__":
    if arabic_model is None and english_model is None:
        print("No models available. Please check the model paths and try again.")
        sys.exit(1)
        
    available_languages = []
    if english_model is not None:
        available_languages.append("[1] English")
    if arabic_model is not None:
        available_languages.append("[2] Arabic")
        
    print(f"Choose language: {' '.join(available_languages)}")
    choice = input("Enter your choice: ")
    
    if choice == "1" and english_model is not None:
        language = "english"
    elif choice == "2" and arabic_model is not None:
        language = "arabic"
    else:
        print("Invalid choice or selected model is not available.")
        sys.exit(1)
    
    while True:
        recognize_speech(language)