from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import os
from dotenv import load_dotenv
from together import Together

#load env variables
load_dotenv()

client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))

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
        return jsonify({"transcript": text, "summary": summary, "template": exam_template})
    except Exception as e:
        return jsonify({"error": str(e)})

def summarize_hpi(transcript):
    prompt = f"Summarize the following clinical conversation as a history of illnes (HPI):\n{transcript}"
    response = client.chat.completions.create(
        model="Qwen/Qwen3-235B-A22B-fp8-tput",
        messages=[
            {
                "role": "system",
                "content": "You are a physician in an emergency department writing a clinical history of illness (CHI) in a professional, clear, and concise manner."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream=False
    )
    return response.choices[0].message.content.strip()

def insert_physical_exam(text):
    if "abdominal pain" in text.lower():
        return "Physical examination: Soft, depressible abdomen, tenderness in the right lower quadrant..."
    elif "cough" in text.lower():
        return "Physical examination: Vesicular murmur present, without added sounds..."
    else:
        return ("General: Appears well and non-toxic."
                "\nNeuro: Alert and oriented x3, normal phonation. moving extremities x4."
                "\nHEENT: Normal, moist oropharynx."
                "\nCardio: Normal heart sounds without murmur."
                "\nResp: Lungs clear bilaterally without adventitious sounds."
                "\nNo accessory muscle use."
                "\nAbdo: Soft, non-tender. No rebound or guarding. No CVA tenderness."
                "\nExtremities: Cap refill <2 seconds, pulses intact."
                "No pedal edema."
                )


if __name__ == '__main__':
    app.run(debug=True)