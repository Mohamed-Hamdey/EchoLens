import sys
import json
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# Load Models for Both English and Arabic
ARABIC_MODEL_PATH = "vosk-model-arabic-0.22"
ENGLISH_MODEL_PATH = "vosk-model-en-us-0.22"

arabic_model = Model(ARABIC_MODEL_PATH)
english_model = Model(ENGLISH_MODEL_PATH)


def recognize_speech(language):
    q = queue.Queue()
    
    # Select the model based on language
    if language == "arabic":
        rec = KaldiRecognizer(arabic_model, 16000)
        print("Listening for Egyptian Arabic...")
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
    print("Choose language: [1] English [2] Arabic")
    choice = input("Enter 1 or 2: ")
    language = "english" if choice == "1" else "arabic"
    
    while True:
        recognize_speech(language)
