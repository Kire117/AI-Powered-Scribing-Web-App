# app.py - Enhanced version with proper template integration
from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import os
import re
import threading
from dotenv import load_dotenv
from together import Together

# import custom template mapper
from template_mapper import TemplateMapper

# Load env variables
load_dotenv()

# client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))
client = Together()

app = Flask(__name__)

# Global variables for recording state
recording_thread = None
is_recording = False
recorded_audio = None
recognizer = sr.Recognizer()

# initialize the template mapper
template_mapper = TemplateMapper()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/')
def health():
    return {'status': 'healthy'}

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording_thread, is_recording, recorded_audio, stop_requested
    
    if is_recording:
        return jsonify({"error": "Already recording"})
    
    try:
        is_recording = True
        stop_requested = False
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
    global is_recording, stop_requested, recorded_audio, recording_thread
    
    if not is_recording:
        return jsonify({"error": "Not currently recording"})
    
    # Stop recording
    stop_requested = True
    
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
        
        # First, analyze transcript to determine the correct template
        template_analysis = template_mapper.analyze_transcript(text)
        print(f"Template Analysis: {template_analysis}")
        
        # Generate summary with the determined template
        clinical_report = generate_clinical_report(text, template_analysis)
        
        return jsonify({
            "transcript": text, 
            "summary": clinical_report, 
            "template_info": {
                "selected_template": template_analysis['best_template'],
                "confidence": template_analysis['confidence']
            }
        })
    
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio"})
    except sr.RequestError as e:
        return jsonify({"error": f"Error with speech recognition service: {e}"})
    except Exception as e:
        return jsonify({"error": str(e)})

def record_audio():
    """Function to record audio continuously until stopped"""
    global is_recording, recorded_audio, recognizer, stop_requested
    
    try:
        mic = sr.Microphone()
        
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Recording started...")
        
        # Record continuously
        audio_data = []
        
        with mic as source:
            while is_recording and not stop_requested:
                try:
                    # Record in short chunks
                    chunk = recognizer.listen(source, timeout=1, phrase_time_limit=1)
                    audio_data.append(chunk.get_wav_data())
                except sr.WaitTimeoutError:
                    # No audio in this chunk, continue
                    continue
        
        # Combine all audio chunks
        if stop_requested and audio_data:
            # Combine WAV data
            combined_wav = combine_wav_data(audio_data)
            recorded_audio = sr.AudioData(combined_wav, mic.SAMPLE_RATE, mic.SAMPLE_WIDTH)
            print("Recording stopped and saved")
        else:
            print("No audio data recorded")
            recorded_audio = None
            
        is_recording = False
            
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

def clean_ai_response(text):
    """
    Cleans LLM responses by removing <think> sections and non-clinical commentary.

    Args:
        text (str): Raw AI response
        
    Returns:
        str: Cleaned clinical report
    """
    
    # Remove <think> tags and their content (case insensitive, multiline)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove other common thinking patterns and metadata
    thinking_patterns = [
        r'<thinking>.*?</thinking>',
        r'\*\*.*?\*\*',  # Remove any bold markdown
        r'Medical Documentation.*?:',
        r'Let me.*?\.{2,}',
        r'I need to.*?\.{2,}',
        r'Okay,.*?\.{2,}',
        r'The patient.*?transcript.*?\.',
        r'Based on.*?analysis.*?\.',
        r'The user wants.*?\.',
        r'The instructions.*?\.',
        r'First,.*?\.',
        r'Now,.*?\.',
        r'Check for.*?\.',
        r'Avoid.*?\.',
        r'So stick.*?\.',
        r'Let me.*?\.',
        r'The analysis.*?\.',
        r'The template.*?\.',
        r'I\'ll.*?\.',
        r'Also,.*?negative\.',
        r'Review of systems.*?mentioned symptoms\.',
    ]

    for pattern in thinking_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove multiple consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading whitespace from each line
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
    
    # Remove any trailing whitespace
    text = text.strip()
    
    # Ensure proper formatting starts with HISTORY OF PRESENT ILLNESS
    if not text.startswith('HISTORY OF PRESENT ILLNESS'):
        # Find the start of the actual clinical content
        hpi_match = re.search(r'HISTORY OF PRESENT ILLNESS:', text, re.IGNORECASE)
        if hpi_match:
            text = text[hpi_match.start():]
    
    # Remove any remaining standalone sentences that seem like thinking
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip lines that look like thinking process
        if any(thinking_phrase in line.lower() for thinking_phrase in [
            'let me', 'i need to', 'okay,', 'the user wants', 'the instructions',
            'first,', 'now,', 'check for', 'avoid', 'so stick', 'the analysis',
            'the template', 'i\'ll', 'also,', 'putting it all together'
        ]):
            continue
        cleaned_lines.append(line)
    
    # Rejoin the cleaned lines
    text = '\n'.join(cleaned_lines)
    
    # Final cleanup - remove extra spaces and ensure proper spacing
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Ensure double newlines between sections
    
    return text.strip()

def advanced_clean_ai_response(text):
    """
    More aggressive cleaning function that uses multiple strategies.
    
    Args:
        text (str): Raw AI response
        
    Returns:
        str: Cleaned clinical report
    """
    
    # First, try to extract only the clinical sections
    clinical_sections = []
    
    # Look for HPI section
    hpi_match = re.search(r'HISTORY OF PRESENT ILLNESS:(.*?)(?=PHYSICAL EXAMINATION:|$)', text, re.DOTALL | re.IGNORECASE)
    if hpi_match:
        hpi_content = hpi_match.group(1).strip()
        clinical_sections.append(f"HISTORY OF PRESENT ILLNESS:\n{hpi_content}")
    
    # Look for Physical Examination section
    pe_match = re.search(r'PHYSICAL EXAMINATION:(.*?)(?=NOTE:|$)', text, re.DOTALL | re.IGNORECASE)
    if pe_match:
        pe_content = pe_match.group(1).strip()
        clinical_sections.append(f"PHYSICAL EXAMINATION:\n{pe_content}")
    
    # Look for Notes section if present
    note_match = re.search(r'NOTE:(.*?)$', text, re.DOTALL | re.IGNORECASE)
    if note_match:
        note_content = note_match.group(1).strip()
        clinical_sections.append(f"NOTE:\n{note_content}")
    
    # If we successfully extracted sections, use them
    if clinical_sections:
        return '\n\n'.join(clinical_sections)
    
    # Fallback to regular cleaning
    return clean_ai_response(text)

def validate_cleaned_response(cleaned_text):
    """
    Validate that the cleaned response contains the expected clinical sections.
    
    Args:
        cleaned_text (str): Cleaned response text
        
    Returns:
        bool: True if response contains required sections
    """
    required_sections = ['HISTORY OF PRESENT ILLNESS', 'PHYSICAL EXAMINATION']
    
    for section in required_sections:
        if section not in cleaned_text.upper():
            return False
    
    # Check that it doesn't contain obvious thinking patterns
    thinking_indicators = ['<think>', 'let me', 'i need to', 'okay,', 'the user wants']
    for indicator in thinking_indicators:
        if indicator.lower() in cleaned_text.lower():
            return False
    
    return True


def generate_clinical_report(transcript, template_analysis):
    """
    Generate HPI summary using AI with the correct template enforced.
    
    Args:
        transcript (str): The patient conversation transcript
        template_analysis (dict): Analysis results from template mapper
        
    Returns:
        str: Generated HPI summary with appropriate examination template
    """
    selected_template = template_analysis['best_template']
    template_text = template_analysis['template_text']
    confidence = template_analysis['confidence']
    
    # prompt that enforces template usage
    prompt = f"""
    You are an emergency department physician writing a clinical history of present illness (HPI) and physical examination report on a patient conversation.
    
    TRANSCRIPT: {transcript}
    
    ANALYSIS RESULTS:
    - Selected Template: {selected_template}
    - Confidence Score: {confidence:.4f}
    - Template Rationale: Based on keywords and clinical presentation
    
    REQUIRED TEMPLATE TO USE:
    {template_text}
    
    INSTRUCTIONS:
    1. Write a professional HPI summary focusing on:
       - Chief complaint
       - Onset, duration, quality, severity
       - Associated symptoms
       - Relevant pertinent positives and negatives
    
    2. For the physical examination section, you MUST use EXACTLY the template provided above for "{selected_template}".
       - Include all sections as written
       - This template was specifically chosen based on the patient's presentation
    
    3. Format your response as:
       HISTORY OF PRESENT ILLNESS:
       [Your HPI summary here]
       
       PHYSICAL EXAMINATION:
       [Use the exact template provided above]
    
    CRITICAL: Provide ONLY the clinical report. Do not include any thinking process, analysis, explanatory text, or commentary. Start directly with "HISTORY OF PRESENT ILLNESS:" and end with the physical examination. No additional text before or after.
    """

    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen3-235B-A22B-fp8-tput",  
            messages=[
                {
                    "role": "system",
                    "content": """You are an experienced emergency department physician. You must provide ONLY the clinical documentation without any thinking process, analysis, or explanatory text. Start directly with "HISTORY OF PRESENT ILLNESS:" and use the exact physical examination template provided. No additional commentary or thinking process should be included."""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=False,
            temperature=0.2, 
            max_tokens=1500
        )
        
        generated_text = response.choices[0].message.content.strip()

        cleaned_output = advanced_clean_ai_response(generated_text)

        if not validate_cleaned_response(cleaned_output):
            cleaned_output = clean_ai_response(generated_text)
        
        if validate_cleaned_response(cleaned_output):
            print(f"✓ Successfully cleaned AI response for {selected_template} template")
            return cleaned_output
        else:
            print(f"⚠ Cleaning may be incomplete, using fallback approach")
            return create_fallback_report(transcript, template_analysis)
            
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        # Fallback: create a basic summary with the template
        return create_fallback_report(transcript, template_analysis)
    

def validate_template_usage(generated_text, expected_template):
    """
    Validate that the AI used the correct template in its response.
    
    Args:
        generated_text (str): The AI-generated response
        expected_template (str): The template that should have been used
        template_name (str): Name of the template
        
    Returns:
        bool: True if template was used correctly
    """
    # Check if key phrases from the template appear in the generated text
    template_lines = expected_template.strip().split('\n')
    key_phrases = []
    
    for line in template_lines:
        line = line.strip()
        if line and not line.startswith('#') and ':' in line:
            key_phrases.append(line.split(':')[0].strip())
    
    # Check if at least 70% of key phrases are present
    found_phrases = sum(1 for phrase in key_phrases if phrase.lower() in generated_text.lower())
    
    if len(key_phrases) == 0:
        return True 
    
    usage_percentage = found_phrases / len(key_phrases)
    print(f"Template usage validation: {found_phrases}/{len(key_phrases)} phrases found ({usage_percentage:.2%})")
    
    return usage_percentage >= 0.7

def enforce_template_usage(transcript, template_analysis, generated_text):
    """
    Enforce correct template usage by reconstructing the response.
    
    Args:
        transcript (str): Original transcript
        template_analysis (dict): Template analysis results
        generated_text (str): AI-generated text that may not follow template
        
    Returns:
        str: Corrected response with proper template usage
    """

    generated_text = clean_ai_response(generated_text)
    # Extract HPI from generated text if possible
    hpi_section = ""
    if "HISTORY OF PRESENT ILLNESS:" in generated_text:
        try:
            hpi_start = generated_text.index("HISTORY OF PRESENT ILLNESS:") + len("HISTORY OF PRESENT ILLNESS:")
            if "PHYSICAL EXAMINATION:" in generated_text:
                hpi_end = generated_text.index("PHYSICAL EXAMINATION:")
                hpi_section = generated_text[hpi_start:hpi_end].strip()
            else:
                hpi_section = generated_text[hpi_start:].strip()
        except:
            pass
    
    # If no HPI extracted, create a basic one
    if not hpi_section:
        hpi_section = f"Patient presents with chief complaint as described in the clinical interview. {transcript[:200]}..."
    
    # Combine with the correct template
    template_text = template_analysis['template_text']
    selected_template = template_analysis['best_template']
    
    return f"""HISTORY OF PRESENT ILLNESS:
{hpi_section}

PHYSICAL EXAMINATION:
{template_text}

NOTE: Physical examination template ({selected_template}) was automatically selected based on clinical presentation analysis."""

def create_fallback_report(transcript, template_analysis):
    """
    Create a basic fallback summary when AI generation fails.
    
    Args:
        transcript (str): Original transcript
        template_analysis (dict): Template analysis results
        
    Returns:
        str: Basic fallback summary
    """
    template_text = template_analysis['template_text']
    selected_template = template_analysis['best_template']
    
    return f"""HISTORY OF PRESENT ILLNESS:
Patient presents with clinical concerns as documented in the interview transcript. Further details available in the recorded conversation.

Transcript summary: {transcript[:300]}{'...' if len(transcript) > 300 else ''}

PHYSICAL EXAMINATION:
{template_text}

NOTE: Physical examination template ({selected_template}) selected automatically based on keyword analysis. Confidence: {template_analysis['confidence']:.3f}"""

# if __name__ == '__main__':
#     print("=== Enhanced AI Scribe App Starting ===")
#     print(f"Available templates: {template_mapper.get_available_templates()}")
#     print("Template integration: ENABLED")
#     print("Server starting on http://localhost:5000")
#     app.run(debug=True)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Fly.io uses 8080
    app.run(host='0.0.0.0', port=port)