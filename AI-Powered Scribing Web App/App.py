from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
from openai import OpenAI
from dotenv import load_dotenv
import os

# load variables from .env
load_dotenv()
client = OpenAI(
    #this is the defacult can be edited
    #api_key = os.environ.get("OPENAI_API_KEY"),
)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("I'm listening...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language='en-EN')
        print("This is what I have: ", text)

        summary = summarize_hpi(text)
        exam_template = insert_physical_exam(text)
        return jsonify({"transcript": text, "summary": summary, "template":exam_template})
    except Exception as e:
        return jsonify({"error": str(e)})

def summarize_hpi(transcript):
    prompt = f"Summarize the following clinical conversation as a history of illness (HPI):\n{transcript}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            # the code in lin e 46 is just optional, can be deleted if you want to, just for cosmetic purposes
            {"role" : "system", "content": "You are a physician in an emergency department writing a clinical history of illness (CHI) in a professional, clear, and concise manner."},
            {"role": "user", "content" : prompt},
        ],
    )
    return response.choices[0].message.content.strip()

def insert_physical_exam(text):
    if "abdominal pain" in text.lower():
        return "Physical examination: Soft, depressible abdomen, tenderness in the right lower quadrant..."
    elif "cough" in text.lower():
        return "Physical examination: Vesicular murmur present, without added sounds..."
    else:
        return "General physical examination: Vital signs within normal ranges."

if __name__ == '__main__':
    app.run(debug=True)
