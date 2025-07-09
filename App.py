# app.py - Fixed version with proper audio handling and error recovery
from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import os
import re
import tempfile
import logging
from dotenv import load_dotenv
from together import Together
import wave
import audioop
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import custom template mapper
try:
    from template_mapper import TemplateMapper
except ImportError:
    logger.error("Could not import template_mapper. Make sure the file exists.")
    # Create a fallback template mapper
    class TemplateMapper:
        def analyze_transcript(self, text):
            return {
                'best_template': 'general',
                'confidence': 0.5,
                'template_text': 'GENERAL:\nVital signs stable.\nExamination findings documented.'
            }

# Load env variables
load_dotenv()

# Initialize Together client
try:
    client = Together()
    logger.info("Together client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Together client: {str(e)}")
    client = None

app = Flask(__name__)

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize the template mapper
template_mapper = TemplateMapper()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return {'status': 'healthy'}

def validate_audio_file(file_path):
    """
    Validate that the audio file is readable and not corrupted.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    try:
        # Check file size
        if os.path.getsize(file_path) == 0:
            logger.error("Audio file is empty (0 bytes)")
            return False
        
        # Try to open as WAV file
        with wave.open(file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            if frames == 0:
                logger.error("Audio file contains no frames")
                return False
        
        logger.info(f"Audio file validation passed: {frames} frames")
        return True
        
    except wave.Error as e:
        logger.error(f"WAV file error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Audio validation error: {str(e)}")
        return False

def convert_audio_format(input_path, output_path):
    """
    Convert audio to proper WAV format for speech recognition.
    
    Args:
        input_path (str): Path to input audio file
        output_path (str): Path to output WAV file
        
    Returns:
        bool: True if conversion successful
    """
    try:
        # Try to read the audio file and convert to proper format
        with wave.open(input_path, 'rb') as input_wav:
            # Get audio parameters
            params = input_wav.getparams()
            frames = input_wav.readframes(params.nframes)
            
            # Convert to 16-bit mono if needed
            if params.sampwidth != 2:
                frames = audioop.lin2lin(frames, params.sampwidth, 2)
            
            if params.nchannels != 1:
                frames = audioop.tomono(frames, 2, 1, 0)
            
            # Write converted audio
            with wave.open(output_path, 'wb') as output_wav:
                output_wav.setnchannels(1)  # Mono
                output_wav.setsampwidth(2)  # 16-bit
                output_wav.setframerate(params.framerate)
                output_wav.writeframes(frames)
        
        logger.info(f"Audio converted successfully: {input_path} -> {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Audio conversion error: {str(e)}")
        return False

def create_valid_wav_file(audio_data, output_path):
    """
    Create a valid WAV file from raw audio data.
    
    Args:
        audio_data (bytes): Raw audio data
        output_path (str): Path to output WAV file
        
    Returns:
        bool: True if file created successfully
    """
    try:
        # Create a basic WAV file with standard parameters
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(44100)  # Standard sample rate
            wav_file.writeframes(audio_data)
        
        logger.info(f"WAV file created: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating WAV file: {str(e)}")
        return False

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Process uploaded audio file and return transcript and summary"""
    
    logger.info("Processing audio request received")
    
    try:
        # Check if audio file is present
        if 'audio' not in request.files:
            logger.error("No audio file provided in request")
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            logger.error("No audio file selected")
            return jsonify({"error": "No audio file selected"}), 400
        
        # Get file size
        audio_file.seek(0, 2)  # Seek to end
        file_size = audio_file.tell()
        audio_file.seek(0)  # Seek back to beginning
        
        logger.info(f"Audio file received: {audio_file.filename}, size: {file_size} bytes")
        
        # Check if file is empty
        if file_size == 0:
            logger.error("Received empty audio file")
            return jsonify({"error": "Audio file is empty. Please record some audio first."}), 400
        
        # Save the uploaded file temporarily
        temp_file_path = None
        converted_file_path = None
        
        try:
            # Save original file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                audio_file.save(temp_file.name)
                temp_file_path = temp_file.name
                logger.info(f"Audio file saved to: {temp_file_path}")
            
            # Validate the audio file
            if not validate_audio_file(temp_file_path):
                logger.warning("Audio file validation failed, attempting to fix...")
                
                # Try to read raw audio data and create a proper WAV file
                with open(temp_file_path, 'rb') as f:
                    raw_data = f.read()
                
                # Create a new temporary file for converted audio
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as converted_file:
                    converted_file_path = converted_file.name
                
                # Try to create a valid WAV file
                if not create_valid_wav_file(raw_data, converted_file_path):
                    return jsonify({"error": "Could not process audio file. Please ensure it's a valid audio recording."}), 400
                
                # Use the converted file
                temp_file_path = converted_file_path
                
                # Validate again
                if not validate_audio_file(temp_file_path):
                    return jsonify({"error": "Audio file appears to be corrupted or in an unsupported format."}), 400
            
        except Exception as e:
            logger.error(f"Error saving/processing audio file: {str(e)}")
            return jsonify({"error": f"Error processing audio file: {str(e)}"}), 500
        
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Process the audio file
        text = ""
        try:
            with sr.AudioFile(temp_file_path) as source:
                logger.info("Processing audio file with speech recognition")
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Record the audio
                audio_data = recognizer.record(source)
                logger.info("Audio data recorded successfully")
                
        except Exception as e:
            logger.error(f"Error reading audio file: {str(e)}")
            return jsonify({"error": f"Error reading audio file: {str(e)}"}), 500
        
        finally:
            # Clean up temporary files
            for file_path in [temp_file_path, converted_file_path]:
                if file_path and os.path.exists(file_path):
                    try:
                        os.unlink(file_path)
                        logger.info(f"Temporary file cleaned up: {file_path}")
                    except:
                        pass
        
        # Transcribe the audio
        try:
            text = recognizer.recognize_google(audio_data, language='en-US')
            logger.info(f"Transcribed text: {text[:100]}...")
            
            # Check if transcription is too short
            if len(text.strip()) < 5:
                return jsonify({"error": "Transcription too short. Please speak more clearly or record longer audio."}), 400
            
        except sr.UnknownValueError:
            logger.error("Could not understand audio")
            return jsonify({"error": "Could not understand audio. Please speak more clearly and try again."}), 400
        except sr.RequestError as e:
            logger.error(f"Error with speech recognition service: {e}")
            return jsonify({"error": f"Speech recognition service error: {e}"}), 500
        except Exception as e:
            logger.error(f"Unexpected error during transcription: {str(e)}")
            return jsonify({"error": f"Unexpected error during transcription: {str(e)}"}), 500
        
        # Analyze transcript to determine the correct template
        try:
            template_analysis = template_mapper.analyze_transcript(text)
            logger.info(f"Template Analysis: {template_analysis}")
        except Exception as e:
            logger.error(f"Error in template analysis: {str(e)}")
            template_analysis = {
                'best_template': 'general',
                'confidence': 0.5,
                'template_text': 'GENERAL:\nVital signs stable.\nExamination findings documented.'
            }
        
        # Generate summary with the determined template
        try:
            clinical_report = generate_clinical_report(text, template_analysis)
            logger.info("Clinical report generated successfully")
        except Exception as e:
            logger.error(f"Error generating clinical report: {str(e)}")
            clinical_report = create_fallback_report(text, template_analysis)
        
        return jsonify({
            "transcript": text, 
            "summary": clinical_report, 
            "template_info": {
                "selected_template": template_analysis['best_template'],
                "confidence": template_analysis['confidence']
            }
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in process_audio: {str(e)}")
        return jsonify({"error": f"Unexpected error processing audio: {str(e)}"}), 500

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
    if not client:
        logger.error("Together client not available, using fallback")
        return create_fallback_report(transcript, template_analysis)
    
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
            logger.info(f"Successfully cleaned AI response for {selected_template} template")
            return cleaned_output
        else:
            logger.warning(f"Cleaning may be incomplete, using fallback approach")
            return create_fallback_report(transcript, template_analysis)
            
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
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