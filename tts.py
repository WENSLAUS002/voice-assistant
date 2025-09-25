from gtts import gTTS
import os
import pyttsx3
import logging

def text_to_speech_gtts(text, filename="output.mp3"):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
        os.system(f"start {filename}")  # Play the generated audio file
        print("TTS (gTTS) output saved and played.")
        return filename
    except Exception as e:
        logging.error(f"Error in gTTS TTS: {e}")
        return None

def text_to_speech_pyttsx3(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        print("TTS (pyttsx3) output spoken.")
    except Exception as e:
        logging.error(f"Error in pyttsx3 TTS: {e}")
        return None

if __name__ == "__main__":
    sample_text = "Welcome to the NLP Banking support. How can I help you today?"
    text_to_speech_gtts(sample_text)
    text_to_speech_pyttsx3(sample_text)
