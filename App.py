# app.py - Optimized version for long audio processing
from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import os
import re
import tempfile
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from together import Together
import wave
import audioop
import io
import subprocess
import math
from threading import Thread
import queue
import time


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import custom template mapper
try:
    from template_mapper import TemplateMapper
except ImportError:
    logger.error("Could not import template_mapper. Make sure the file exists.")
    class TemplateMapper:
        def analyze_transcript(self, text):
            return {
                'best_template': 'general',
                'confidence': 0.5,
                'template_text': 'GENERAL:\nVital signs stable.\nExamination findings documented.'
            }

# Load env variables
load_dotenv()

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

app = Flask(__name__)

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

# Initialize the template mapper
template_mapper = TemplateMapper()

# Thread pool for processing
executor = ThreadPoolExecutor(max_workers=2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return {'status': 'healthy'}

def split_audio_into_chunks(audio_file_path, chunk_duration_seconds=30):
    """
    Split audio file into smaller chunks for processing.
    """
    try:
        with wave.open(audio_file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            total_duration = frames / sample_rate
            chunk_frames = int(chunk_duration_seconds * sample_rate)
            
            logger.info(f"Audio duration: {total_duration:.2f}s, will create {math.ceil(total_duration/chunk_duration_seconds)} chunks")
            
            chunk_files = []
            
            for i in range(0, frames, chunk_frames):
                # Create chunk file
                chunk_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'_chunk_{i//chunk_frames}.wav')
                chunk_path = chunk_file.name
                chunk_file.close()
                
                # Read chunk data
                wav_file.setpos(i)
                chunk_data = wav_file.readframes(min(chunk_frames, frames - i))
                
                # Write chunk to file
                with wave.open(chunk_path, 'wb') as chunk_wav:
                    chunk_wav.setnchannels(channels)
                    chunk_wav.setsampwidth(sample_width)
                    chunk_wav.setframerate(sample_rate)
                    chunk_wav.writeframes(chunk_data)
                
                chunk_files.append(chunk_path)
                logger.info(f"Created chunk {len(chunk_files)}: {chunk_path}")
        
        return chunk_files
        
    except Exception as e:
        logger.error(f"Error splitting audio: {str(e)}")
        return []

def process_audio_chunk(chunk_path, chunk_index):
    """
    Process a single audio chunk.
    """
    try:
        logger.info(f"Processing chunk {chunk_index}: {chunk_path}")
        
        recognizer = sr.Recognizer()
        # Optimized settings for faster processing
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = False  # Disable for faster processing
        recognizer.pause_threshold = 0.5
        recognizer.operation_timeout = 30  
        
        with sr.AudioFile(chunk_path) as source:
            # Shorter ambient noise adjustment
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
            
            # Single recognition attempt for speed
            try:
                text = recognizer.recognize_google(audio_data, language='en-US')
                logger.info(f"Chunk {chunk_index} processed successfully: {len(text)} chars")
                return True, text, None
            except sr.UnknownValueError:
                logger.warning(f"Chunk {chunk_index}: No speech detected")
                return True, "", None  # Empty chunk is OK
            except sr.RequestError as e:
                logger.error(f"Chunk {chunk_index}: Service error: {e}")
                return False, "", str(e)
                
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_index}: {str(e)}")
        return False, "", str(e)

def process_long_audio_parallel(audio_file_path, max_duration=300):
    """
    Process long audio files by splitting into chunks and processing in parallel.
    """
    try:
        # Check audio duration first
        with wave.open(audio_file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            duration = frames / sample_rate
            
            logger.info(f"Audio duration: {duration:.2f} seconds")
            
            if duration > max_duration:
                logger.warning(f"Audio too long ({duration:.2f}s), truncating to {max_duration}s")
                # Truncate the audio file
                truncated_file = tempfile.NamedTemporaryFile(delete=False, suffix='_truncated.wav')
                truncated_path = truncated_file.name
                truncated_file.close()
                
                truncated_frames = int(max_duration * sample_rate)
                truncated_data = wav_file.readframes(truncated_frames)
                
                with wave.open(truncated_path, 'wb') as truncated_wav:
                    truncated_wav.setnchannels(wav_file.getnchannels())
                    truncated_wav.setsampwidth(wav_file.getsampwidth())
                    truncated_wav.setframerate(sample_rate)
                    truncated_wav.writeframes(truncated_data)
                
                audio_file_path = truncated_path
                duration = max_duration
        
        # For shorter audio, process normally
        if duration <= 30:
            logger.info("Short audio, processing normally")
            return process_audio_chunk(audio_file_path, 0)
        
        # For longer audio, split into chunks
        logger.info("Long audio detected, splitting into chunks")
        chunk_files = split_audio_into_chunks(audio_file_path, chunk_duration_seconds=25)
        
        if not chunk_files:
            return False, "", "Failed to split audio into chunks"
        
        # Process chunks in parallel (but limit concurrency)
        transcripts = []
        errors = []
        
        def process_chunk_wrapper(args):
            chunk_path, chunk_index = args
            return process_audio_chunk(chunk_path, chunk_index)
        
        # Process chunks with limited concurrency
        chunk_args = [(chunk_path, i) for i, chunk_path in enumerate(chunk_files)]
        
        # Process in batches to avoid overwhelming the system
        batch_size = 2
        for i in range(0, len(chunk_args), batch_size):
            batch = chunk_args[i:i+batch_size]
            
            # Process batch
            with ThreadPoolExecutor(max_workers=batch_size) as batch_executor:
                batch_results = list(batch_executor.map(process_chunk_wrapper, batch))
            
            # Collect results
            for success, transcript, error in batch_results:
                if success:
                    if transcript:  # Only add non-empty transcripts
                        transcripts.append(transcript)
                else:
                    errors.append(error)
            
            # Small delay between batches
            time.sleep(0.5)
        
        # Clean up chunk files
        for chunk_path in chunk_files:
            try:
                os.unlink(chunk_path)
            except:
                pass
        
        # Check results
        if not transcripts and errors:
            return False, "", f"All chunks failed: {'; '.join(errors[:3])}"
        
        if not transcripts:
            return False, "", "No speech detected in any chunk"
        
        # Combine transcripts
        combined_transcript = ' '.join(transcripts)
        logger.info(f"Combined transcript: {len(combined_transcript)} characters from {len(transcripts)} chunks")
        
        if len(combined_transcript.strip()) < 10:
            return False, "", "Combined transcript too short"
        
        return True, combined_transcript, None
        
    except Exception as e:
        logger.error(f"Error in parallel processing: {str(e)}")
        return False, "", str(e)

def validate_audio_file(file_path):
    """Enhanced audio file validation with more detailed checks."""
    try:
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

def convert_to_wav_ffmpeg(input_path, output_path):
    # Full path to ffmpeg.exe — replace with your actual path
    ffmpeg_path = r"C:\ffmpeg\ffmpeg-2025-07-12-git-35a6de137a-essentials_build\ffmpeg-2025-07-12-git-35a6de137a-essentials_build\bin\ffmpeg.exe"

    try:
        logger.info(f"Converting {input_path} to {output_path} using ffmpeg")

        result = subprocess.run([
            ffmpeg_path,
            '-y',
            '-i', input_path,
            '-ar', '16000',
            '-ac', '1',
            '-f', 'wav',
            output_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=30)

        logger.info("ffmpeg conversion successful")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg conversion failed: {e.stderr.decode('utf-8')}")
        return False
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {str(e)} - Tried path: {ffmpeg_path}")
        return False
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg conversion timed out")
        return False





def process_audio_with_fallback(audio_file_path):
    """
    Process audio with multiple fallback strategies, optimized for long audio.
    """
    
    # Strategy 1: Try parallel processing for long audio
    logger.info("Strategy 1: Parallel processing for long audio")
    try:
        success, transcript, error_message = process_long_audio_parallel(audio_file_path)
        if success:
            return True, transcript, None
        else:
            logger.warning(f"Parallel processing failed: {error_message}")
    except Exception as e:
        logger.error(f"Parallel processing error: {str(e)}")
    
    # Strategy 2: Try ffmpeg conversion and retry
    logger.info("Strategy 2: Converting audio format using ffmpeg")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as converted_file:
            converted_path = converted_file.name

        if convert_to_wav_ffmpeg(audio_file_path, converted_path):
            if validate_audio_file(converted_path):
                try:
                    success, transcript, error_message = process_long_audio_parallel(converted_path)
                    if success:
                        return True, transcript, None
                except Exception as e:
                    logger.error(f"Strategy 2 error: {str(e)}")

        # Clean up converted file
        try:
            os.unlink(converted_path)
        except:
            pass

    except Exception as e:
        logger.error(f"ffmpeg conversion failed: {str(e)}")
    
    # All strategies failed
    return False, "", "Could not process audio. Please ensure the recording contains clear speech and try with a shorter recording if the issue persists."

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Process uploaded audio file and return transcript and summary"""


    logger.info("Processing audio request received")

    try:
        if 'audio' not in request.files:
            logger.error("No audio file provided in request")
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']

        if audio_file.filename == '':
            logger.error("No audio file selected")
            return jsonify({"error": "No audio file selected"}), 400

        # Save uploaded file temporarily (as webm, ogg, etc.)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
            audio_file.save(temp_file.name)
            temp_file_path = temp_file.name
            logger.info(f"Original audio file saved: {temp_file_path}")

        # Convert to proper WAV using ffmpeg
        converted_path = temp_file_path.replace('.webm', '.wav')

        if not convert_to_wav_ffmpeg(temp_file_path, converted_path):
            logger.error("Conversion to WAV failed")
            return jsonify({"error": "Failed to convert audio to supported format"}), 500

        logger.info(f"Converted WAV file: {converted_path}")

        # Optionally check file size and metadata
        if not validate_audio_file(converted_path):
            logger.warning("WAV file validation failed")
            return jsonify({"error": "Invalid or unsupported audio format"}), 400

        # Transcribe using fallback strategy
        success, transcript, error_message = process_audio_with_fallback(converted_path)

        if not success:
            logger.error(f"Speech recognition failed: {error_message}")
            return jsonify({"error": error_message}), 400

        if len(transcript.strip()) < 10:
            return jsonify({"error": "Transcription too short. Please speak more clearly or record longer audio."}), 400


        # Analyze transcript

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


        # Generate clinical summary
        try:
            clinical_report = generate_clinical_report(transcript, template_analysis)
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
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

    finally:
        # Clean up temporary files
        try:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            if os.path.exists(converted_path):
                os.unlink(converted_path)
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up temp files: {cleanup_error}")




def clean_ai_response(text):
    """Cleans LLM responses by removing <think> sections and non-clinical commentary."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    thinking_patterns = [
        r'<thinking>.*?</thinking>',
        r'\*\*.*?\*\*',
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
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
    text = text.strip()
    
    if not text.startswith('HISTORY OF PRESENT ILLNESS'):
        hpi_match = re.search(r'HISTORY OF PRESENT ILLNESS:', text, re.IGNORECASE)
        if hpi_match:
            text = text[hpi_match.start():]
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if any(thinking_phrase in line.lower() for thinking_phrase in [
            'let me', 'i need to', 'okay,', 'the user wants', 'the instructions',
            'first,', 'now,', 'check for', 'avoid', 'so stick', 'the analysis',
            'the template', 'i\'ll', 'also,', 'putting it all together'
        ]):
            continue
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()

def generate_clinical_report(transcript, template_analysis):
    """Generate HPI summary using AI with the correct template enforced."""
    
    selected_template = template_analysis['best_template']
    template_text = template_analysis['template_text']
    confidence = template_analysis['confidence']
    
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
       
       PHYSICAL EXAMINATION: {selected_template.upper()}
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
        cleaned_output = clean_ai_response(generated_text)
        
        logger.info(f"Successfully generated clinical report for {selected_template} template")
        return cleaned_output
            
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return create_fallback_report(transcript, template_analysis)

def create_fallback_report(transcript, template_analysis):
    """Create a basic fallback summary when AI generation fails."""
    template_text = template_analysis['template_text']
    selected_template = template_analysis['best_template']
    
    return f"""HISTORY OF PRESENT ILLNESS:
Patient presents with clinical concerns as documented in the interview transcript. Further details available in the recorded conversation.

Transcript summary: {transcript[:300]}{'...' if len(transcript) > 300 else ''}

PHYSICAL EXAMINATION:
{template_text}

NOTE: Physical examination template ({selected_template}) selected automatically based on keyword analysis. Confidence: {template_analysis['confidence']:.3f}"""

# production env
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

# local env
# if __name__ == '__main__':
#     print("=== Enhanced AI Scribe App Starting ===")
#     print(f"Available templates: {template_mapper.get_available_templates()}")
#     print("Template integration: ENABLED")
#     print("Server starting on http://localhost:5000")
#     app.run(debug=True)