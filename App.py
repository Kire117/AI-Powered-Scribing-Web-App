# app.py - Complete version with enhanced audio processing and error recovery
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
    Enhanced audio file validation with more detailed checks.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    try:
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logger.error("Audio file is empty (0 bytes)")
            return False
        
        if file_size < 1000:  # Less than 1KB is probably too small
            logger.warning(f"Audio file very small: {file_size} bytes")
        
        # Try to open as WAV file
        with wave.open(file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            frame_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            duration = frames / frame_rate if frame_rate > 0 else 0
            
            logger.info(f"Audio details: {frames} frames, {duration:.2f}s duration, {frame_rate}Hz sample rate, {channels} channels, {sample_width} bytes sample width")
            
            if frames == 0:
                logger.error("Audio file contains no frames")
                return False
            
            if duration < 0.1:  # Less than 0.1 seconds
                logger.warning(f"Audio duration very short: {duration:.2f} seconds")
                return False
            
            logger.info(f"Audio validation passed: {frames} frames, {duration:.2f}s duration, {frame_rate}Hz sample rate")
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
            
            # Use 16kHz sample rate for better speech recognition
            target_rate = 16000
            if params.framerate != target_rate:
                frames = audioop.ratecv(frames, 2, 1, params.framerate, target_rate, None)[0]
            
            # Write converted audio
            with wave.open(output_path, 'wb') as output_wav:
                output_wav.setnchannels(1)  # Mono
                output_wav.setsampwidth(2)  # 16-bit
                output_wav.setframerate(target_rate)
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
            wav_file.setframerate(16000)  # 16kHz sample rate for speech
            wav_file.writeframes(audio_data)
        
        logger.info(f"WAV file created: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating WAV file: {str(e)}")
        return False

def improved_speech_recognition(temp_file_path):
    """
    Enhanced speech recognition with multiple attempts and better error handling.
    
    Args:
        temp_file_path (str): Path to the audio file
        
    Returns:
        str: Transcribed text
        
    Raises:
        sr.UnknownValueError: When speech cannot be understood
        sr.RequestError: When there's a service error
        Exception: For other errors
    """
    recognizer = sr.Recognizer()
    
    # Adjust recognizer settings for better performance
    recognizer.energy_threshold = 200  # Lower threshold for quieter audio
    recognizer.dynamic_energy_threshold = True
    recognizer.dynamic_energy_adjustment_damping = 0.15
    recognizer.dynamic_energy_ratio = 1.5
    recognizer.pause_threshold = 0.8  # Seconds of non-speaking audio before a phrase is considered complete
    recognizer.operation_timeout = None  # No timeout for recognition
    recognizer.phrase_threshold = 0.3  # Minimum length of a phrase
    recognizer.non_speaking_duration = 0.5  # Seconds of non-speaking audio to keep on both sides
    
    try:
        with sr.AudioFile(temp_file_path) as source:
            logger.info("Processing audio with improved recognition settings")
            
            # Adjust for ambient noise with longer duration
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            
            # Record the audio
            audio_data = recognizer.record(source)
            logger.info("Audio data recorded successfully")
            
            # Try multiple recognition attempts with different settings
            recognition_attempts = [
                # Attempt 1: Standard Google recognition
                {
                    'language': 'en-US',
                    'show_all': False,
                    'description': 'Standard US English'
                },
                # Attempt 2: Google with show_all=True for confidence scores
                {
                    'language': 'en-US',
                    'show_all': True,
                    'description': 'US English with confidence scores'
                },
                # Attempt 3: Try with different language hint
                {
                    'language': 'en',
                    'show_all': False,
                    'description': 'Generic English'
                },
                # Attempt 4: Try with Canadian English
                {
                    'language': 'en-CA',
                    'show_all': False,
                    'description': 'Canadian English'
                }
            ]
            
            for i, attempt in enumerate(recognition_attempts, 1):
                try:
                    logger.info(f"Recognition attempt {i}: {attempt['description']}")
                    
                    result = recognizer.recognize_google(
                        audio_data, 
                        language=attempt['language'],
                        show_all=attempt['show_all']
                    )
                    
                    # Handle different result formats
                    if attempt['show_all'] and isinstance(result, dict):
                        if 'alternative' in result and result['alternative']:
                            text = result['alternative'][0]['transcript']
                            confidence = result['alternative'][0].get('confidence', 0)
                            logger.info(f"Recognition successful with confidence: {confidence}")
                        else:
                            logger.warning(f"Attempt {i}: No alternatives found")
                            continue
                    else:
                        text = result
                        logger.info(f"Recognition successful: {text[:100]}...")
                    
                    # Validate result
                    if text and len(text.strip()) >= 3:
                        logger.info(f"Final transcription: {text}")
                        return text.strip()
                    else:
                        logger.warning(f"Attempt {i}: Text too short or empty: '{text}'")
                        continue
                        
                except sr.UnknownValueError:
                    logger.warning(f"Attempt {i}: Could not understand audio")
                    continue
                except sr.RequestError as e:
                    logger.error(f"Attempt {i}: Recognition service error: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Attempt {i}: Unexpected error: {e}")
                    continue
            
            # If all attempts failed
            raise sr.UnknownValueError("Could not understand audio after multiple attempts")
            
    except Exception as e:
        logger.error(f"Error in speech recognition: {str(e)}")
        raise

def process_audio_with_fallback(audio_file_path):
    """
    Process audio with multiple fallback strategies.
    
    Args:
        audio_file_path (str): Path to the original audio file
        
    Returns:
        tuple: (success: bool, transcript: str, error_message: str)
    """
    
    # Strategy 1: Try with original file
    logger.info("Strategy 1: Processing original audio file")
    try:
        transcript = improved_speech_recognition(audio_file_path)
        return True, transcript, None
    except sr.UnknownValueError as e:
        logger.warning(f"Strategy 1 failed: {str(e)}")
    except sr.RequestError as e:
        logger.error(f"Strategy 1 service error: {str(e)}")
        return False, "", f"Speech recognition service error: {str(e)}"
    except Exception as e:
        logger.error(f"Strategy 1 unexpected error: {str(e)}")
    
    # Strategy 2: Try with converted audio
    logger.info("Strategy 2: Converting audio format and retrying")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as converted_file:
            converted_path = converted_file.name
        
        if convert_audio_format(audio_file_path, converted_path):
            if validate_audio_file(converted_path):
                try:
                    transcript = improved_speech_recognition(converted_path)
                    return True, transcript, None
                except sr.UnknownValueError as e:
                    logger.warning(f"Strategy 2 failed: {str(e)}")
                except Exception as e:
                    logger.error(f"Strategy 2 error: {str(e)}")
        
        # Clean up converted file
        try:
            os.unlink(converted_path)
        except:
            pass
        
    except Exception as e:
        logger.error(f"Audio conversion failed: {str(e)}")
    
    # Strategy 3: Try with basic audio recreation
    logger.info("Strategy 3: Recreating audio file")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as recreated_file:
            recreated_path = recreated_file.name
        
        # Read raw audio data and recreate WAV file
        with open(audio_file_path, 'rb') as f:
            raw_data = f.read()
        
        # Skip WAV header if present and use raw audio data
        if raw_data.startswith(b'RIFF'):
            # Find the data chunk
            data_start = raw_data.find(b'data') + 8
            if data_start > 8:
                raw_data = raw_data[data_start:]
        
        if create_valid_wav_file(raw_data, recreated_path):
            if validate_audio_file(recreated_path):
                try:
                    transcript = improved_speech_recognition(recreated_path)
                    return True, transcript, None
                except sr.UnknownValueError as e:
                    logger.warning(f"Strategy 3 failed: {str(e)}")
                except Exception as e:
                    logger.error(f"Strategy 3 error: {str(e)}")
        
        # Clean up recreated file
        try:
            os.unlink(recreated_path)
        except:
            pass
        
    except Exception as e:
        logger.error(f"Audio recreation failed: {str(e)}")
    
    # All strategies failed
    return False, "", "Could not understand audio. Please ensure the recording contains clear speech, minimal background noise, and sufficient volume."

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
        
        # Check file size limits
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            logger.error(f"File too large: {file_size} bytes")
            return jsonify({"error": "Audio file too large. Please record a shorter message."}), 400
        
        # Save the uploaded file temporarily
        temp_file_path = None
        
        try:
            # Save original file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                audio_file.save(temp_file.name)
                temp_file_path = temp_file.name
                logger.info(f"Audio file saved to: {temp_file_path}")
            
            # Log audio file details
            logger.info(f"Audio file size: {os.path.getsize(temp_file_path)} bytes")
            
            # Try to get audio details for debugging
            try:
                with wave.open(temp_file_path, 'rb') as wav:
                    logger.info(f"Audio format - Sample rate: {wav.getframerate()}, Channels: {wav.getnchannels()}, Duration: {wav.getnframes()/wav.getframerate():.2f}s")
            except Exception as e:
                logger.warning(f"Could not read audio details: {str(e)}")
            
            # Validate the audio file
            if not validate_audio_file(temp_file_path):
                logger.warning("Audio file validation failed, will attempt processing anyway")
            
        except Exception as e:
            logger.error(f"Error saving/processing audio file: {str(e)}")
            return jsonify({"error": f"Error processing audio file: {str(e)}"}), 500
        
        # Process the audio file with enhanced recognition
        try:
            success, transcript, error_message = process_audio_with_fallback(temp_file_path)
            
            if not success:
                logger.error(f"Speech recognition failed: {error_message}")
                return jsonify({"error": error_message}), 400
            
            logger.info(f"Transcribed text: {transcript[:100]}...")
            
            # Check if transcription is too short
            if len(transcript.strip()) < 5:
                return jsonify({"error": "Transcription too short. Please speak more clearly or record longer audio."}), 400
            
        except Exception as e:
            logger.error(f"Unexpected error during transcription: {str(e)}")
            return jsonify({"error": f"Unexpected error during transcription: {str(e)}"}), 500
        
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.info(f"Temporary file cleaned up: {temp_file_path}")
                except:
                    pass
        
        # Analyze transcript to determine the correct template
        try:
            template_analysis = template_mapper.analyze_transcript(transcript)
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
            clinical_report = generate_clinical_report(transcript, template_analysis)
            logger.info("Clinical report generated successfully")
        except Exception as e:
            logger.error(f"Error generating clinical report: {str(e)}")
            clinical_report = create_fallback_report(transcript, template_analysis)
        
        return jsonify({
            "transcript": transcript, 
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