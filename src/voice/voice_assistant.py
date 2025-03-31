import speech_recognition as sr
import pyttsx3

class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()

    def listen(self):
        with sr.Microphone() as source:
            print("Listening...")
            audio = self.recognizer.listen(source)
        try:
            query = self.recognizer.recognize_google(audio)
            print(f"User said: {query}")
            return query
        except Exception as e:
            print("Sorry, I could not understand that.")
            return ""

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()
