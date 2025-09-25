import speech_recognition as sr
import logging

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)  
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            print("Recognized Text:", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError:
            print("Error: Could not request results. Check your internet connection.")
            return None
        except Exception as e:
            print("Error:", e)
            return None

if __name__ == "__main__":
    text_output = speech_to_text()
    if text_output:
        print("Converted Speech to Text:", text_output)
        logging.info(f"Processed Text: {text_output}")
       