import sounddevice as sd
import queue
import sys
import json
import threading
import numpy as np
from collections import deque
from vosk import Model, KaldiRecognizer
from langdetect import detect, DetectorFactory

# Set seed for consistent language detection
DetectorFactory.seed = 0

# === Load Vosk Models ===
print("🔁 Loading Vosk models...")
model_en = Model("models/vosk-model-en-us-0.22")
model_ar = Model("models/vosk-model-ar-0.22-linto-1.1.0")
print("✅ Models loaded.")

# === Initialize Recognizers ===
recognizer_en = KaldiRecognizer(model_en, 16000)
recognizer_ar = KaldiRecognizer(model_ar, 16000)

# === Audio Stream Queue ===
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(f"⚠️ {status}", file=sys.stderr)
    q.put(bytes(indata))

# === Smoothing and State Management ===
class LanguageDetectionState:
    def __init__(self, history_size=10, switch_threshold=0.6):
        self.history_size = history_size
        self.switch_threshold = switch_threshold
        self.lang_history = deque(maxlen=history_size)
        self.current_lang = None
        self.confidence_history = {'en': deque(maxlen=history_size), 
                                  'ar': deque(maxlen=history_size)}
        self.silence_count = 0
        self.consecutive_same_lang = 0

    def update(self, detected_lang, en_conf, ar_conf):
        # Reset silence counter if we detected speech
        if detected_lang:
            self.silence_count = 0
        else:
            self.silence_count += 1
            
        # Add language to history
        if detected_lang:
            self.lang_history.append(detected_lang)
            self.confidence_history['en'].append(en_conf)
            self.confidence_history['ar'].append(ar_conf)
            
        # Determine most likely language based on history
        if len(self.lang_history) >= 3:
            en_count = sum(1 for lang in self.lang_history if lang == 'en')
            ar_count = len(self.lang_history) - en_count
            
            # Calculate average confidence
            avg_en_conf = sum(self.confidence_history['en']) / max(1, len(self.confidence_history['en']))
            avg_ar_conf = sum(self.confidence_history['ar']) / max(1, len(self.confidence_history['ar']))
            
            lang_ratio = en_count / len(self.lang_history)
            
            # Determine language with hysteresis to prevent rapid switching
            prev_lang = self.current_lang
            
            if lang_ratio > self.switch_threshold and avg_en_conf > 0.4:
                new_lang = 'en'
            elif lang_ratio < (1 - self.switch_threshold) and avg_ar_conf > 0.4:
                new_lang = 'ar'
            elif self.current_lang:
                new_lang = self.current_lang  # Maintain current language if uncertain
            else:
                # Initial language selection based on confidence
                new_lang = 'en' if avg_en_conf > avg_ar_conf else 'ar'
                
            # Track consecutive same language detections
            if new_lang == prev_lang:
                self.consecutive_same_lang += 1
            else:
                self.consecutive_same_lang = 0
                
            self.current_lang = new_lang
            
            return self.current_lang
        elif detected_lang:
            self.current_lang = detected_lang
            return detected_lang
        return None

    def reset_on_silence(self, silence_threshold=15):
        """Reset state after prolonged silence"""
        if self.silence_count > silence_threshold:
            self.lang_history.clear()
            self.confidence_history['en'].clear()
            self.confidence_history['ar'].clear()
            self.consecutive_same_lang = 0
            return True
        return False

# Initialize language detection state
lang_state = LanguageDetectionState(history_size=10)

# === Audio Energy Detection ===
class AudioEnergyDetector:
    def __init__(self, threshold=300, frame_size=1600, history_size=10):
        self.threshold = threshold
        self.frame_size = frame_size
        self.energy_history = deque(maxlen=history_size)
        self.is_speech = False
        
    def process(self, audio_data):
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Calculate energy
        energy = np.sqrt(np.mean(audio_array**2))
        self.energy_history.append(energy)
        
        # Determine if speech based on average energy
        avg_energy = sum(self.energy_history) / len(self.energy_history)
        was_speech = self.is_speech
        self.is_speech = avg_energy > self.threshold
        
        # Detect transitions
        if not was_speech and self.is_speech:
            return "start"
        elif was_speech and not self.is_speech:
            return "end"
        else:
            return "continue"

# === Improved Confidence Calculation ===
def calculate_confidence(result):
    """Calculate confidence score with more weight on recent words"""
    words = result.get("result", [])
    if not words:
        return 0
    
    # More weight to longer words and recent words
    total_weight = 0
    weighted_conf = 0
    
    for i, word in enumerate(words):
        length_weight = min(len(word.get("word", "")), 8) / 4  # Max length factor of 2
        recency_weight = 1 + (i / len(words))  # More recent words get higher weight
        combined_weight = length_weight * recency_weight
        
        weighted_conf += word.get("conf", 0) * combined_weight
        total_weight += combined_weight
    
    return weighted_conf / total_weight if total_weight > 0 else 0

# === Language-specific text processing ===
def process_text(text, lang):
    """Apply language-specific post-processing"""
    if not text:
        return text
        
    if lang == 'en':
        # English-specific processing (capitalization, etc.)
        processed = text.strip()
        if processed and len(processed) > 1:
            processed = processed[0].upper() + processed[1:]
        return processed
    elif lang == 'ar':
        # Arabic-specific processing
        return text.strip()
    return text.strip()

# === Context-aware language detection ===
def detect_language_with_context(text_en, text_ar, conf_en, conf_ar):
    """Detect language with contextual awareness"""
    # Strong confidence difference - clear winner
    if conf_en > conf_ar + 0.2:
        return 'en', conf_en, conf_ar
    elif conf_ar > conf_en + 0.2:
        return 'ar', conf_en, conf_ar
    
    # Similar confidence - try additional methods
    if text_en and text_ar:
        try:
            # Try langdetect as backup
            combined_text = text_en if len(text_en) > len(text_ar) else text_ar
            detected = detect(combined_text)
            
            # Map langdetect codes to our language identifiers
            lang_map = {'en': 'en', 'ar': 'ar'}
            detected = lang_map.get(detected[:2], None)
            
            if detected:
                return detected, conf_en, conf_ar
        except Exception as e:
            pass
    
    # If one has text and other doesn't
    if text_en and not text_ar:
        return 'en', conf_en, conf_ar
    elif text_ar and not text_en:
        return 'ar', conf_en, conf_ar
    
    # Default to the one with higher confidence
    return 'en' if conf_en >= conf_ar else 'ar', conf_en, conf_ar

# === Main Recognition Function ===
def recognize_stream():
    print("🎙️ Listening with enhanced language detection...")
    
    # Initialize energy detector
    energy_detector = AudioEnergyDetector(threshold=500)
    
    # Keep track of partial results
    partial_results = {'en': '', 'ar': ''}
    utterance_buffer = {'en': [], 'ar': []}
    
    while True:
        try:
            data = q.get()
            
            # Check audio energy
            energy_state = energy_detector.process(data)
            
            # Process with both recognizers
            has_en = recognizer_en.AcceptWaveform(data)
            has_ar = recognizer_ar.AcceptWaveform(data)
            
            # Get results
            if has_en:
                res_en = json.loads(recognizer_en.Result())
                text_en = res_en.get("text", "").strip()
                if text_en:
                    utterance_buffer['en'].append(text_en)
                partial_results['en'] = ''
            else:
                res_en = json.loads(recognizer_en.PartialResult())
                partial_results['en'] = res_en.get("partial", "").strip()
                
            if has_ar:
                res_ar = json.loads(recognizer_ar.Result())
                text_ar = res_ar.get("text", "").strip()
                if text_ar:
                    utterance_buffer['ar'].append(text_ar)
                partial_results['ar'] = ''
            else:
                res_ar = json.loads(recognizer_ar.PartialResult())
                partial_results['ar'] = res_ar.get("partial", "").strip()
            
            # Calculate confidence scores
            conf_en = calculate_confidence(res_en) if has_en else 0
            conf_ar = calculate_confidence(res_ar) if has_ar else 0
            
            # Check if we have final results
            if has_en or has_ar:
                text_en = res_en.get("text", "").strip()
                text_ar = res_ar.get("text", "").strip()
                
                if text_en or text_ar:
                    # Detect language for this segment
                    detected_lang, en_conf, ar_conf = detect_language_with_context(text_en, text_ar, conf_en, conf_ar)
                    
                    # Update language state
                    final_lang = lang_state.update(detected_lang, conf_en, conf_ar)
                    
                    # Get appropriate text based on detected language
                    final_text = text_en if final_lang == 'en' else text_ar
                    final_text = process_text(final_text, final_lang)
                    
                    if final_text:
                        confidence = conf_en if final_lang == 'en' else conf_ar
                        print(f"🗣️ [{final_lang.upper()}] ({confidence:.2f}) → {final_text}")
            
            # Check for silence reset
            if lang_state.reset_on_silence(silence_threshold=20):
                if any(utterance_buffer['en']) or any(utterance_buffer['ar']):
                    print("📌 End of utterance detected")
                    utterance_buffer = {'en': [], 'ar': []}
            
            # Show partial results based on current language preference
            current_lang = lang_state.current_lang
            if current_lang and partial_results[current_lang]:
                # Print partial results with a different indicator
                print(f"🔄 [{current_lang.upper()}] → {partial_results[current_lang]}", end='\r')
                
        except Exception as e:
            print(f"Error in recognition: {e}")

def start_audio_listener():
    stream = sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                               channels=1, callback=callback)
    with stream:
        recognize_stream()

if __name__ == "__main__":
    try:
        print("Starting multilingual speech recognition...")
        start_audio_listener()
    except KeyboardInterrupt:
        print("\nStopping recognition.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)