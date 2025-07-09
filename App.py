# app.py - Updated for client-side audio recording
from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import os
import re
import tempfile
from dotenv import load_dotenv
from together import Together

# import custom template mapper
from template_mapper import TemplateMapper

# Load env variables
load_dotenv()

client = Together()
app = Flask(__name__)

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# initialize the template mapper
template_mapper = TemplateMapper()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return {'status': 'healthy'}

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Process uploaded audio file and return transcript and summary"""
    
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({"error": "No audio file selected"}), 400
    
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Process the audio file
        with sr.AudioFile(temp_file_path) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Record the audio
            audio_data = recognizer.record(source)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        # Transcribe the audio
        try:
            text = recognizer.recognize_google(audio_data, language='en-US')
            print("Transcribed text:", text)
        except sr.UnknownValueError:
            return jsonify({"error": "Could not understand audio"}), 400
        except sr.RequestError as e:
            return jsonify({"error": f"Error with speech recognition service: {e}"}), 500
        
        # Analyze transcript to determine the correct template
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
        
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return jsonify({"error": f"Error processing audio: {str(e)}"}), 500

def clean_ai_response(text):
    """
    Cleans LLM responses by removing <think> sections and non-clinical commentary.
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
            print(f"Successfully cleaned AI response for {selected_template} template")
            return cleaned_output
        else:
            print(f"Cleaning may be incomplete, using fallback approach")
            return create_fallback_report(transcript, template_analysis)
            
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return create_fallback_report(transcript, template_analysis)

def create_fallback_report(transcript, template_analysis):
    """
    Create a basic fallback summary when AI generation fails.
    """
    template_text = template_analysis['template_text']
    selected_template = template_analysis['best_template']
    
    return f"""HISTORY OF PRESENT ILLNESS:
Patient presents with clinical concerns as documented in the interview transcript. Further details available in the recorded conversation.

Transcript summary: {transcript[:300]}{'...' if len(transcript) > 300 else ''}

PHYSICAL EXAMINATION:
{template_text}

NOTE: Physical examination template ({selected_template}) selected automatically based on keyword analysis. Confidence: {template_analysis['confidence']:.3f}"""

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False)  # Set debug=False for production