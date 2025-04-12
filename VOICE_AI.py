import google.generativeai as genai
import pyttsx3
import os
genai.configure(api_key="AIzaSyCUgs3Ta_M2kHOrTXr3p8p0S20z7-9Ez-8")

engine=pyttsx3.init()


def content_model(text):
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(text)
    responses=response.candidates[0].content.parts[0].text
    return responses

def text_to_speech(response_text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    voices=engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    full_path = r"C:\python\RIT_HACKATHON\mictest - Copy\static\audio\response.wav"
    # Ensure directory exists
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    engine.save_to_file(response_text, full_path)
    engine.runAndWait()

def Voice_Generator(text):
    Generated=content_model(text)
    text_to_speech(Generated)
