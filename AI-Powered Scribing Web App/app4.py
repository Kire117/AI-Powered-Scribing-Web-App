from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import os
import threading
import time
from dotenv import load_dotenv
from together import Together
import wave
import io

# Load env variables
load_dotenv()

client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))

app = Flask(__name__)

# Global variables for recording state
recording_thread = None
is_recording = False
recorded_audio = None
recognizer = sr.Recognizer()

@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording_thread, is_recording, recorded_audio
    
    if is_recording:
        return jsonify({"error": "Already recording"})
    
    try:
        is_recording = True
        recorded_audio = None
        
        # Start recording in a separate thread
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.start()
        
        return jsonify({"message": "Recording started"})
    
    except Exception as e:
        is_recording = False
        return jsonify({"error": str(e)})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global is_recording, recorded_audio, recording_thread
    
    if not is_recording:
        return jsonify({"error": "Not currently recording"})
    
    # Stop recording
    is_recording = False
    
    # Wait for recording thread to finish
    if recording_thread:
        recording_thread.join(timeout=5)  # Wait up to 5 seconds
    
    # Process the recorded audio
    if recorded_audio is None:
        return jsonify({"error": "No audio was recorded"})
    
    try:
        # Transcribe the audio
        text = recognizer.recognize_google(recorded_audio, language='en-US')
        print("Transcribed text:", text)
        
        # Generate summary and template
        summary = summarize_hpi(text)
        exam_template = insert_physical_exam(text)
        
        return jsonify({
            "transcript": text, 
            "summary": summary, 
            "template": exam_template
        })
    
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio"})
    except sr.RequestError as e:
        return jsonify({"error": f"Error with speech recognition service: {e}"})
    except Exception as e:
        return jsonify({"error": str(e)})

def record_audio():
    """Function to record audio continuously until stopped"""
    global is_recording, recorded_audio, recognizer
    
    try:
        mic = sr.Microphone()
        
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Recording started...")
        
        # Record continuously
        audio_data = []
        
        with mic as source:
            while is_recording:
                try:
                    # Record in short chunks
                    chunk = recognizer.listen(source, timeout=1, phrase_time_limit=1)
                    audio_data.append(chunk.get_wav_data())
                except sr.WaitTimeoutError:
                    # No audio in this chunk, continue
                    continue
        
        # Combine all audio chunks
        if audio_data:
            # Combine WAV data
            combined_wav = combine_wav_data(audio_data)
            recorded_audio = sr.AudioData(combined_wav, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
            print("Recording stopped and saved")
        else:
            print("No audio data recorded")
            
    except Exception as e:
        print(f"Error during recording: {e}")
        is_recording = False

def combine_wav_data(wav_data_list):
    """Combine multiple WAV data chunks into a single WAV"""
    if not wav_data_list:
        return b''
    
    # For simplicity, just concatenate the WAV data
    # This is a basic approach - in production you might want more sophisticated audio processing
    combined = b''
    for i, wav_data in enumerate(wav_data_list):
        if i == 0:
            # Keep the full WAV header from the first chunk
            combined += wav_data
        else:
            # Skip the header (first 44 bytes) for subsequent chunks
            combined += wav_data[44:]
    
    return combined

def summarize_hpi(transcript):
    """Generate HPI summary using AI"""
    prompt = f"Summarize the following clinical conversation as a history of present illness (HPI):\n{transcript}"
    
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen3-235B-A22B-fp8-tput",  
            messages=[
                {
                    "role": "system",
                    "content": "You are a physician in an emergency department writing a clinical history of present illness (HPI) in a professional, clear, and concise manner."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def insert_physical_exam(text):
    """Generate physical examination template based on transcript"""
    text_lower = text.lower()
    
    if "abdominal pain" in text_lower or "stomach" in text_lower:
        return ("Physical examination:\n"
                "General: Appears uncomfortable due to abdominal pain.\n"
                "Abdomen: Soft, depressible, tenderness in the right lower quadrant, "
                "no rebound or guarding. Bowel sounds present.\n"
                "Extremities: Cap refill <2 seconds, pulses intact.")
    
    elif "cough" in text_lower or "breathing" in text_lower or "chest" in text_lower:
        return ("Physical examination:\n"
                "General: Appears well, no acute distress.\n"
                "Respiratory: Vesicular murmur present, without added sounds. "
                "No accessory muscle use. Chest expansion symmetric.\n"
                "Cardiovascular: Regular rate and rhythm, no murmurs.")
    
    elif "headache" in text_lower or "head" in text_lower:
        return ("Physical examination:\n"
                "General: Appears uncomfortable.\n"
                "Neurological: Alert and oriented x3, normal phonation, moving extremities x4. "
                "No focal neurological deficits.\n"
                "HEENT: Normocephalic, atraumatic. Pupils equal, round, reactive to light.")
    
    else:
        return ("Physical examination:\n"
                "General: Appears well and non-toxic.\n"
                "Neurological: Alert and oriented x3, normal phonation, moving extremities x4.\n"
                "HEENT: Normal, moist oropharynx.\n"
                "Cardiovascular: Normal heart sounds without murmur.\n"
                "Respiratory: Lungs clear bilaterally without adventitious sounds. "
                "No accessory muscle use.\n"
                "Abdomen: Soft, non-tender. No rebound or guarding. No CVA tenderness.\n"
                "Extremities: Cap refill <2 seconds, pulses intact. No pedal edema.")

if __name__ == '__main__':
    app.run(debug=True)