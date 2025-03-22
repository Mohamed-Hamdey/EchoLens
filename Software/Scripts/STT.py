import sys
import threading
import json
import queue
import os
import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer
from langdetect import detect

# 🔹 Model Paths
ARABIC_MODEL_PATH = r"E:\Desktop\EchoLens\Software\Models\vosk-model-ar-0.22-linto-1.1.0"
ENGLISH_MODEL_PATH = r"E:\Desktop\EchoLens\Software\Models\vosk-model-en-us-0.22"

# 🔹 Load Models
def load_model(path):
    if not os.path.exists(path):
        print(f"❌ Model not found: {path}")
        return None
    return Model(path)

arabic_model = load_model(ARABIC_MODEL_PATH)
english_model = load_model(ENGLISH_MODEL_PATH)

# 🎤 Global Variables
recording = False
q = queue.Queue()
selected_language = "Auto"  # Default mode
current_recognizer = None  # Add global variable for current recognizer

# 🎤 Audio Processing
def preprocess_audio(audio_data):
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    max_val = np.max(np.abs(audio_array))
    if max_val > 0:
        audio_array = (audio_array / max_val) * 32767
    return audio_array.astype(np.int16).tobytes()

# 🔍 Detect Language
def detect_language(text):
    try:
        if not text.strip():  # Check if text is empty or whitespace
            return "unknown"
        return detect(text)
    except:
        return "unknown"

# 🎤 Speech Recognition Thread
def recognize_speech():
    global recording, selected_language, current_recognizer

    # Choose initial model based on selected language
    if selected_language == "English":
        model = english_model
    elif selected_language == "Arabic":
        model = arabic_model
    else:  # Auto mode - start with English by default
        model = english_model
    
    if model is None:
        print("❌ Error: Selected language model is missing!")
        recording = False  # Make sure to set recording to False
        return
    
    # Initialize recognizer
    current_recognizer = KaldiRecognizer(model, 16000)
    current_recognizer.SetWords(True)
    
    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        if recording:  # Only process audio when recording is active
            processed_audio = preprocess_audio(bytes(indata))
            q.put(processed_audio)

    # Start the audio stream
    stream = sd.RawInputStream(
        samplerate=16000, 
        blocksize=8000, 
        dtype='int16',
        channels=1, 
        callback=callback
    )
    
    with stream:
        while recording:
            try:
                data = q.get(timeout=1.0)  # Add timeout to prevent blocking forever
                if current_recognizer.AcceptWaveform(data):
                    result = json.loads(current_recognizer.Result())
                    text = result.get("text", "")
                    
                    if text:  # Only process non-empty text
                        print(f"Transcribed Text: {text}")  # Output text to console
                        
                        # In Auto mode, switch models if needed
                        if selected_language == "Auto":
                            detected_lang = detect_language(text)
                            if detected_lang == "ar" and isinstance(current_recognizer, KaldiRecognizer) and current_recognizer._model == english_model:
                                current_recognizer = KaldiRecognizer(arabic_model, 16000)
                                current_recognizer.SetWords(True)
                            elif detected_lang == "en" and isinstance(current_recognizer, KaldiRecognizer) and current_recognizer._model == arabic_model:
                                current_recognizer = KaldiRecognizer(english_model, 16000)
                                current_recognizer.SetWords(True)
            except queue.Empty:
                continue  # Handle timeout gracefully
            except Exception as e:
                print(f"Error in recognition thread: {e}")
                recording = False
                break

# ✅ Start Recording
def start_recording():
    global recording
    
    # Don't start if already recording
    if recording:
        return
        
    # Clear the queue to prevent processing old audio
    while not q.empty():
        q.get()
        
    recording = True
    print("🎤 Recording started...")
    threading.Thread(target=recognize_speech, daemon=True).start()

# ❌ Stop Recording
def stop_recording():
    global recording
    recording = False
    print("🛑 Recording stopped.")

# 🔄 Change Language
def change_language(new_lang):
    global selected_language, recording
    
    # If we're recording, stop first
    was_recording = recording
    if recording:
        stop_recording()
        
    selected_language = new_lang
    print(f"🌍 Language changed to: {new_lang}")
    
    # Restart recording if it was active
    if was_recording:
        start_recording()

# 🌍 Main Function
if __name__ == "__main__":
    print("🎤 Speech-to-Text Converter (SW Only)")
    print("Commands:")
    print("1. Type 'start' to begin recording.")
    print("2. Type 'stop' to stop recording.")
    print("3. Type 'lang <language>' to change language (e.g., 'lang English').")
    print("4. Type 'exit' to quit.")

    while True:
        command = input("> ").strip().lower()
        
        if command == "start":
            start_recording()
        elif command == "stop":
            stop_recording()
        elif command.startswith("lang "):
            new_lang = command.split(" ", 1)[1].capitalize()
            if new_lang in ["English", "Arabic", "Auto"]:
                change_language(new_lang)
            else:
                print("❌ Invalid language. Choose 'English', 'Arabic', or 'Auto'.")
        elif command == "exit":
            stop_recording()
            print("👋 Exiting...")
            break
        else:
            print("❌ Invalid command. Try 'start', 'stop', 'lang <language>', or 'exit'.")