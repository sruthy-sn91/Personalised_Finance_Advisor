import speech_recognition as sr
import pyttsx3
import threading
import queue
import time

class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.speech_queue = queue.Queue()
        self.speaker_thread = threading.Thread(target=self._speak_loop, daemon=True)
        self.speaker_thread.start()

    def listen(self):
        with sr.Microphone() as source:
            print("Listening...")
            audio = self.recognizer.listen(source, phrase_time_limit=10)  # ⬅️ up to 10 seconds
        try:
            query = self.recognizer.recognize_google(audio)
            print(f"User said: {query}")
            return query
        except Exception:
            print("Sorry, I could not understand that.")
            return ""

    def speak(self, text):
        print(f"[VoiceAssistant] Response: {text[:200]}")
        self.speech_queue.put(text)

    def _speak_loop(self):
        while True:
            text = self.speech_queue.get()
            try:
                print(f"[VoiceAssistant] Speaking: {text[:100]}")
                self.engine.say(text)
                self.engine.runAndWait()
                time.sleep(0.5)  # slight delay to ensure engine resets cleanly
            except RuntimeError as e:
                print(f"[VoiceAssistant Error]: {e}")
            finally:
                self.speech_queue.task_done()
