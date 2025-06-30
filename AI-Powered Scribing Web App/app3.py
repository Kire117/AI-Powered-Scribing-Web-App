from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import os
from dotenv import load_dotenv
from together import Together
import re

# Load environment variables
load_dotenv()

client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))

app = Flask(__name__)

# Dictionary of short → long form medical terms
MEDICAL_VOCAB = {
    # Original Core Terms
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "cad": "coronary artery disease",
    "mi": "myocardial infarction",
    "sob": "shortness of breath",
    "copd": "chronic obstructive pulmonary disease",
    "ckd": "chronic kidney disease",
    "afib": "atrial fibrillation",
    "cva": "cerebrovascular accident",
    "gi": "gastrointestinal",
    "abd": "abdomen",
    "rx": "prescription",
    "tx": "treatment",
    "sx": "symptoms",
    "dx": "diagnosis",

    # Emergency Department Essentials
    "aox3": "alert and oriented to person, place, and time",
    "aox4": "alert and oriented to person, place, time, and situation",
    "chf": "congestive heart failure",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "pna": "pneumonia",
    "uti": "urinary tract infection",
    "pid": "pelvic inflammatory disease",
    "ibd": "inflammatory bowel disease",
    "gerd": "gastroesophageal reflux disease",
    "cp": "chest pain",
    "tia": "transient ischemic attack",
    "loc": "loss of consciousness",
    "gcs": "Glasgow Coma Scale",
    "od": "overdose",
    "si": "suicidal ideation",
    "tbi": "traumatic brain injury",
    "sbo": "small bowel obstruction",
    "aaa": "abdominal aortic aneurysm",
    "ards": "acute respiratory distress syndrome",
    "aki": "acute kidney injury",
    "dka": "diabetic ketoacidosis",
    "sz": "seizure",
    "etoh": "alcohol",
    "etoh abuse": "alcohol abuse",
    "etoh intox": "alcohol intoxication",
    "n/v": "nausea and vomiting",

    # Vital Signs/Assessments
    "vss": "vital signs stable",
    "t": "temperature",
    "hr": "heart rate",
    "rr": "respiratory rate",
    "bp": "blood pressure",
    "o2": "oxygen",
    "spo2": "oxygen saturation",
    "fio2": "fraction of inspired oxygen",
    "wnl": "within normal limits",

    # Labs/Tests
    "cbc": "complete blood count",
    "bmp": "basic metabolic panel",
    "cmp": "comprehensive metabolic panel",
    "lytes": "electrolytes",
    "abg": "arterial blood gas",
    "ua": "urinalysis",
    "pt": "prothrombin time",
    "inr": "international normalized ratio",
    "lft": "liver function tests",
    "tsh": "thyroid stimulating hormone",
    "crp": "C-reactive protein",
    "esr": "erythrocyte sedimentation rate",
    "hba1c": "hemoglobin A1c",

    # Imaging
    "ct": "computed tomography",
    "mri": "magnetic resonance imaging",
    "us": "ultrasound",
    "cxr": "chest x-ray",
    "kubu": "kidney ureter bladder x-ray",
    "xray": "x-ray",

    # Procedures
    "cpr": "cardiopulmonary resuscitation",
    "io": "intraosseous",
    "iv": "intravenous",
    "im": "intramuscular",
    "neb": "nebulizer treatment",
    "ecg": "electrocardiogram",
    "lp": "lumbar puncture",
    "ng": "nasogastric",
    "foley": "foley catheter",

    # Medications/Routes
    "po": "by mouth",
    "pr": "per rectum",
    "sl": "sublingual",
    "prn": "as needed",
    "stat": "immediately",
    "d50": "dextrose 50%",
    "ns": "normal saline",
    "lr": "lactated ringers",
    "ivf": "intravenous fluids",
    "abx": "antibiotics",
    "asa": "acetylsalicylic acid (aspirin)",
    "ntg": "nitroglycerin",

    # Hospital Terms
    "ed": "emergency department",
    "icu": "intensive care unit",
    "or": "operating room",
    "ccu": "cardiac care unit",
    "medsurg": "medical surgical unit",

    # Additional High-Yield
    "npo": "nothing by mouth",
    "fbo": "foreign body obstruction",
    "fx": "fracture",
    "lac": "laceration",
    "anaphylaxis": "anaphylaxis",
    "sepsis": "sepsis",
    "r/o": "rule out",
    "hpi": "history of present illness",
    "ros": "review of systems",
    "pmhx": "past medical history",
    "pshx": "past surgical history",
    "fhx": "family history",
    "shx": "social history",

    # Toronto & Ontario-Specific
    "mvc": "motor vehicle collision",
    "lhsc": "London Health Sciences Centre",
    "tgh": "Toronto General Hospital",
    "sunnybrook": "Sunnybrook Health Sciences Centre",
    "uh": "University Hospital",
    "mts": "Canadian Triage and Acuity Scale"
}

def normalize_medical_terms(text):
    for short, full in MEDICAL_VOCAB.items():
        text = text.replace(short.lower(), full.lower())
    return text

# Clean model response to remove filler text or reasoning
def clean_ai_output(text):
    noisy_starts = [
        "based on", "as a language model", "it appears that", "in summary",
        "the patient seems to", "overall", "given the information"
    ]
    lines = text.strip().splitlines()
    clean_lines = [
        line for line in lines
        if not any(line.lower().strip().startswith(start) for start in noisy_starts)
    ]
    clean_text = "\n".join(clean_lines).strip()
    clean_text = re.sub(r"(i am an ai|as an ai model).*", "", clean_text, flags=re.I)
    return clean_text

def load_exam_templates(filepath="ed_exam_templates.txt"):
    sections = {}
    current_label = None
    current_lines = []

    with open(filepath, "r") as file:
        for line in file:
            stripped = line.strip()
            # Detect section headers (ends with ':' and not a negation line starting with 'No ')
            if stripped.endswith(":") and not stripped.startswith("No "):
                # Save previous section if exists
                if current_label and current_lines:
                    sections[current_label] = "\n".join(current_lines).strip()
                    current_lines = []
                current_label = stripped.rstrip(":")
            elif current_label:
                current_lines.append(line.rstrip())

        # Save the last section
        if current_label and current_lines:
            sections[current_label] = "\n".join(current_lines).strip()

    return sections

SECTION_OVERRIDES = {
    "abdominal pain": {
        "Abdo": "Abdo: Tender in the RLQ with guarding. No rebound or rigidity."
    },
    "chest pain": {
        "Cardio": "Cardio: Irregular rhythm noted. Mild tenderness to palpation in the chest wall.",
        "Resp": "Resp: Clear to auscultation, no respiratory distress."
    },
    "shortness of breath": {
        "Resp": "Resp: Diffuse wheezing noted bilaterally. No rales."
    },
    "trauma": {
        "Back": "Back: Step-off deformity over the lumbar spine. Tender to palpation."
    }
}

def apply_symptom_overrides(sections, hpi_summary):
    for symptom, overrides in SECTION_OVERRIDES.items():
        if symptom in hpi_summary.lower():
            for label, new_text in overrides.items():
                sections[label] = new_text
    return sections

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
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=120)
        print("Recording finished.")

    try:
        text = recognizer.recognize_google(audio, language='en-EN')
        print("Raw transcript:", text)

        # Apply medical vocabulary normalization
        text = normalize_medical_terms(text)
        print("Normalized transcript:", text)

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
        model="Qwen/Qwen3-235B-A22B-fp8-tput",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a physician in an emergency department. "
                    "Respond ONLY with the final clinical summary in a clear, concise, and professional manner. "
                    "Do not explain your reasoning or use introductory phrases."
                    )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream=False
    )
    return clean_ai_output(response.choices[0].message.content.strip())

def generate_physical_exam(hpi_summary):
    base_sections = load_exam_templates()

    prompt = f"""
You are an emergency physician. Based on the following history of present illness (HPI), generate a full physical exam using the following examples as references.

HPI:
{hpi_summary}

These are common formatted physical exams used in the ED:

{"\n\n".join(f"{k}:\n{v}" for k, v in base_sections.items())}

Only generate a final, formatted physical exam based on the HPI above. Do not explain your reasoning. Structure your response like the templates.
"""

    response = client.chat.completions.create(
        model="Qwen/Qwen3-235B-A22B-fp8-tput",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream=False
    )

    ai_output = clean_ai_output(response.choices[0].message.content.strip())

    # Parse AI output into sections
    generated_sections = {}
    current_label = None
    current_lines = []

    for line in ai_output.splitlines():
        if line.strip().endswith(":") and not line.strip().startswith("No "):
            if current_label and current_lines:
                generated_sections[current_label] = "\n".join(current_lines).strip()
                current_lines = []
            current_label = line.strip().rstrip(":")
        elif current_label:
            current_lines.append(line.strip())

    if current_label and current_lines:
        generated_sections[current_label] = "\n".join(current_lines).strip()

    # Apply symptom overrides
    final_sections = apply_symptom_overrides(generated_sections, hpi_summary)

    # Return combined PE text with sections and spacing preserved
    return "\n\n".join(f"{label}:\n{text}" for label, text in final_sections.items())


if __name__ == '__main__':
    app.run(debug=True)
