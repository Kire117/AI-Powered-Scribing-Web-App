from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import os
from dotenv import load_dotenv
from together import Together

# Load environment variables
load_dotenv()

client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index2.html')

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
        print("Transcript:", text)

        summary = summarize_hpi(text)
        physical_exam = generate_physical_exam(summary)

        return jsonify({
            "transcript": text,
            "summary": summary,
            "physical_exam": physical_exam
        })

    except Exception as e:
        return jsonify({"error": str(e)})

def summarize_hpi(transcript):
    prompt = f"Summarize the following clinical conversation as a history of present illness (HPI):\n{transcript}"
    response = client.chat.completions.create(
        model="Qwen/Qwen3-235B-A22B-fp8-tput",  # You can replace with DeepSeek or another model
        messages=[
            {
                "role": "system",
                "content": "You are a physician in an emergency department writing a clinical history of present illness (HPI) in a clear, concise, and professional manner."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream=False
    )
    return response.choices[0].message.content.strip()

def generate_physical_exam(hpi_summary):
    # load the physical exam template examples
    try:
        with open("ed_exam_templates.txt", "r") as file:
            template_examples = file.read()
    except FileNotFoundError:
        template_examples = "Template file not found. Please check the file path."

    prompt = f"""
    You are an emergency physician. Based on the following history of present illness (HPI), write a complete physical exam using the following examples as your reference.

    HPI:
    {hpi_summary}

    Use the structure and language from these real ED physical exam templates:

    {template_examples}

    Now generate the most appropriate physical exam for this patient.
    """

    response = client.chat.completions.create(
        model="Qwen/Qwen3-235B-A22B-fp8-tput",  # Use DeepSeek or your preferred Together model
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream=False
    )
    return response.choices[0].message.content.strip()

if __name__ == '__main__':
    app.run(debug=True)