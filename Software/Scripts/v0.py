import sys
import tkinter as tk
import customtkinter as ctk  # Modern UI
import threading
import json
import queue
import os
import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer
from langdetect import detect
from tkinter import messagebox

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
        messagebox.showerror("Error", "Selected language model is missing!")
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
                        update_text(text)
                        
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

# 🏆 Update Text in UI
def update_text(text):
    if not text.strip():  # Skip empty text
        return
        
    # Use after method to update UI from a different thread
    root.after(0, lambda: _update_text_safe(text))

def _update_text_safe(text):
    transcription_text.configure(state="normal")
    transcription_text.insert(tk.END, text + "\n")
    transcription_text.see(tk.END)  # Auto-scroll to the latest text
    transcription_text.configure(state="disabled")

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
    start_button.configure(state="disabled")
    stop_button.configure(state="normal")
    threading.Thread(target=recognize_speech, daemon=True).start()

# ❌ Stop Recording
def stop_recording():
    global recording
    recording = False
    start_button.configure(state="normal")
    stop_button.configure(state="disabled")

# 📋 Copy to Clipboard
def copy_text():
    text = transcription_text.get("1.0", tk.END).strip()
    if text:
        root.clipboard_clear()
        root.clipboard_append(text)
        messagebox.showinfo("Copied", "Text copied to clipboard!")
    else:
        messagebox.showinfo("Info", "No text to copy!")

# 🧹 Clear Text
def clear_text():
    transcription_text.configure(state="normal")
    transcription_text.delete("1.0", tk.END)
    transcription_text.configure(state="disabled")

# 🔄 Change Language
def change_language(new_lang):
    global selected_language, recording
    
    # If we're recording, stop first
    was_recording = recording
    if recording:
        stop_recording()
        
    selected_language = new_lang
    
    # Restart recording if it was active
    if was_recording:
        start_recording()

# 🎨 GUI Design
ctk.set_appearance_mode("dark")
root = ctk.CTk()
root.title("Speech-to-Text Converter")
root.geometry("600x500")
root.protocol("WM_DELETE_WINDOW", lambda: (stop_recording(), root.destroy()))  # Clean shutdown

# 🏆 Title Label
title_label = ctk.CTkLabel(root, text="🎤 Speech-to-Text Converter", font=("Arial", 20))
title_label.pack(pady=10)

# 🌍 Language Selection
lang_frame = ctk.CTkFrame(root)
lang_frame.pack(pady=10)

lang_label = ctk.CTkLabel(lang_frame, text="Select Language:")
lang_label.pack(side="left", padx=5)

lang_options = ["English", "Arabic", "Auto"]
lang_menu = ctk.CTkOptionMenu(lang_frame, values=lang_options, command=change_language)
lang_menu.set("Auto")
lang_menu.pack(side="right", padx=5)

# 📜 Transcription Textbox
transcription_frame = ctk.CTkFrame(root)
transcription_frame.pack(pady=10, fill="both", expand=True, padx=20)

transcription_text = ctk.CTkTextbox(transcription_frame, wrap="word")
transcription_text.pack(fill="both", expand=True, padx=5, pady=5)
transcription_text.configure(state="disabled")

# 🎤 Buttons
button_frame = ctk.CTkFrame(root)
button_frame.pack(pady=10)

start_button = ctk.CTkButton(button_frame, text="Start Recording 🎤", command=start_recording)
start_button.grid(row=0, column=0, padx=5)

stop_button = ctk.CTkButton(button_frame, text="Stop Recording 🛑", command=stop_recording)
stop_button.grid(row=0, column=1, padx=5)
stop_button.configure(state="disabled")  # Initially disabled

clear_button = ctk.CTkButton(button_frame, text="Clear Text 🧹", command=clear_text)
clear_button.grid(row=0, column=2, padx=5)

copy_button = ctk.CTkButton(button_frame, text="Copy to Clipboard 📋", command=copy_text)
copy_button.grid(row=0, column=3, padx=5)

# Status indicator
status_var = tk.StringVar(value="Ready")
status_label = ctk.CTkLabel(root, textvariable=status_var)
status_label.pack(pady=5)

# 🌍 Run App
if __name__ == "__main__":
    root.mainloop()