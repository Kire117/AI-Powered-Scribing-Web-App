# template_mapper.py
"""
Template Mapping System for Medical AI Scribe
This module handles the mapping of patient problems to appropriate examination templates.
"""

import re
from typing import Dict, List, Tuple

class TemplateMapper:
    """
    A class to map patient transcripts to appropriate medical examination templates.
    """
    
    def __init__(self):
        """Initialize the template mapper with keywords and templates."""
        self.template_keywords = self._load_keywords()
        self.templates = self._load_templates()
    
    def _load_keywords(self) -> Dict[str, List[str]]:
        """Load comprehensive keyword mappings for each template type."""
        return {
            'Trauma': [
                'accident', 'fall', 'fell', 'injury', 'injured', 'trauma', 'hit', 'crash', 
                'collision', 'car accident', 'motor vehicle', 'fracture', 'broken', 'cut',
                'laceration', 'bruise', 'contusion', 'wound', 'bleeding', 'blood',
                'head injury', 'concussion', 'whiplash', 'sprain', 'strain', 'hurt',
                'motorcycle', 'bicycle', 'pedestrian', 'assault', 'fight', 'attack', 'stabbed'
            ],
            
            'Neurological': [
                'headache', 'migraine', 'dizziness', 'dizzy', 'vertigo', 'seizure',
                'confusion', 'memory loss', 'stroke', 'numbness', 'tingling',
                'weakness', 'paralysis', 'vision problems', 'double vision',
                'speech problems', 'slurred speech', 'tremor', 'shaking',
                'balance problems', 'coordination', 'neurological', 'confused',
                'disoriented', 'altered mental status', 'syncope', 'fainting',
                'loss of consciousness', 'blackout'
            ],
            
            'HEENT': [
                'ear pain', 'earache', 'hearing loss', 'tinnitus', 'ear infection',
                'sore throat', 'throat pain', 'swallowing', 'hoarse', 'voice',
                'eye pain', 'vision', 'blurry vision', 'red eye', 'eye discharge',
                'runny nose', 'nasal congestion', 'sinus', 'sinusitis',
                'nosebleed', 'epistaxis', 'dental pain', 'tooth pain', 'jaw pain',
                'earwax', 'ear discharge', 'neck pain', 'swollen glands',
                'lymph nodes', 'cold symptoms', 'flu symptoms'
            ],
            
            'Lumbar': [
                'back pain', 'lower back', 'lumbar', 'sciatica', 'spine',
                'spinal pain', 'herniated disc', 'disc', 'back injury',
                'back spasm', 'muscle spasm', 'shooting pain down leg',
                'leg pain from back', 'can\'t bend', 'stiff back',
                'pinched nerve', 'radiating pain', 'chronic back pain',
                'acute back pain', 'lifted something heavy'
            ],
            
            'MaleGU': [
                'groin pain', 'testicular pain', 'testicle', 'scrotum', 'penis',
                'urinary', 'urine', 'burning urination', 'difficulty urinating',
                'blood in urine', 'kidney stone', 'prostate', 'discharge',
                'genital', 'erectile', 'epididymitis',
                'orchitis', 'UTI', 'urinary tract infection', 'hematuria',
                'dysuria', 'frequency', 'urgency'
            ],
            
            'Musculoskeletal': [
                'joint pain', 'muscle pain', 'shoulder pain', 'arm pain',
                'wrist pain', 'hand pain', 'knee pain', 'ankle pain',
                'foot pain', 'hip pain', 'arthritis', 'swelling',
                'sprain', 'strain', 'torn muscle', 'pulled muscle',
                'can\'t move', 'stiff joint', 'locked joint', 'tendonitis',
                'bursitis', 'ligament', 'cartilage', 'sports injury',
                'overuse injury', 'repetitive strain'
            ],
            
            'Ocular': [
                'eye pain', 'vision loss', 'blurred vision', 'double vision',
                'flashing lights', 'floaters', 'eye injury', 'foreign body in eye',
                'chemical in eye', 'eye discharge', 'red eye', 'pink eye',
                'light sensitivity', 'photophobia', 'eye trauma', 'conjunctivitis',
                'corneal abrasion', 'something in eye', 'eye irritation',
                'sudden vision loss', 'blind spot'
            ],
            
            'Mental status exam': [
                'anxiety', 'depression', 'panic', 'suicidal', 'mental health',
                'psychiatric', 'hallucinations', 'voices', 'paranoid',
                'manic', 'bipolar', 'mood', 'emotional', 'psychological',
                'stress', 'worried', 'sad', 'angry', 'agitated',
                'panic attack', 'social anxiety', 'PTSD', 'trauma',
                'substance abuse', 'addiction', 'withdrawal'
            ],
            'Pediatrics': [
                'child', 'baby', 'infant', 'pediatric', 'toddler', 'teenager',
                'school-age', 'high fever', 'rash on child', 'child not eating',
                'dehydrated', 'diaper rash', 'sick kid', 'baby vomiting'
            ]
        }
    
    def _load_templates(self) -> Dict[str, str]:
        """Load all the examination templates."""
        return {
            'Normal': """NORMAL\n
            General: Appears well and non-toxic.\n
Neuro: Alert and oriented x3, normal phonation. moving extremities x4.\n
HEENT: Normal, moist oropharynx.\n
Cardio: Normal heart sounds without murmur.\n
Resp: Lungs clear bilaterally without adventitious sounds.\n
No accessory muscle use.\n
Abdo: Soft, non-tender. No rebound or guarding. No CVA tenderness.\n
Extremities:\n
Cap refill <2 seconds, pulses intact.\n
No pedal edema.""",

            'HEENT': """HEENT\n
            General: Appears well.\n
Neuro: Alert and oriented x3, normal phonation. moving extremities x4.\n
HEENT: No ocular injection or exudate. No proptosis.\n
Normal extraocular movements. PERRLA.\n
No periorbital edema or erythema.\n
No frank rhinorrhea or nasal crusting.\n
No tonsillar swelling or exudate.\n
No peritonsillar swelling. Uvula midline.\n
No tragus or pinna tenderness.\n
Normal TMs without opacification or bulging.\n
No mastoid tenderness.\n
No head and neck lymphadenopathy/mass.\n
Adequate neck ROM with no neck/spine tenderness.\n
Cardio: Normal heart sounds without murmur.\n
Resp: No audible stridor. Lungs clear bilaterally without adventitious sounds.\n
Abdo: Soft, non-tender.\n
Lower ext: No edema bilaterally.""",

            'Lumbar': """LUMBAR\n
            General: Appears well.\n
Neuro: Alert, normal phonation, moving extremities x4.\n
Lower back: No deformity or skin changes.\n
No spinal point tenderness.\n
No paraspinal tenderness bilaterally.\n
Range of motion for flexion, extension, and rotation preserved.\n
Normal (5/5) strength bilaterally for hip flexion (L2), knee extension (L3),
ankle dorsiflexion (L4), great toe extension (L5), ankle plantarflexion (S1),
and knee flexion (S2)\n
Normal and symmetric sensation in lower extremities bilaterally.\n
Normal (+2) patellar and Achilles reflexes bilaterally.\n
Pulses intact legs bilaterally.\n
Straight leg raise negative bilaterally.\n
Normal gait without antalgia.\n
DRE deferred.""",

            'MaleGU': """MALEGU\n
            General: Appears well.\n
Neuro: Alert, normal phonation. moving extremities x4.\n
Abdo: Soft, non-tender.\n
GU: No external genital lesions noted.\n
No frank penile discharge.\n
No testicular swelling.\n
No high riding testes. No transverse lie.\n
No testicular or epididymal tenderness.\n
Testicles soft to palpation.\n
Cremasteric reflex present bilaterally. No perineal tenderness.\n
DRE deferred.""",

            'MSE': """MSE\n
            General: Alert, no apparent distress, no apparent intoxication, well-groomed, dressed appropriately.\n
Behaviour: Calm, cooperative.\n
Neurological: A&Ox3, CN II-XII intact, normal balance, normal coordination, power/sensation normal bilateral, gait normal.\n
Speech: Normal in tone, volume, fluency and speed.\n
Mental status: Euthymic, congruent affect, no evidence of mania, thought process normal, no suicidal ideation, no homicidal ideation.\n
Perception: No delusions, no illusions, no auditory/visual/somatic hallucinations, not responding to internal stimuli.\n
Insight: Adequate
Judgment: Fair
Cognition: Not formally tested but grossly intact""",

            'MSK': """MSK\n
            Looks well. range of motion and strength adequate. neurovascular intact. no open wound. no swelling or deformities. no bony tenderness. Joint above and below intact. Ligamentous testing intact.""",

            'Neuro': """NEURO\n
            General: Appears well. Alert and oriented x3. No aphasia. No hemispatial neglect.\n
Cranial nerves: Visual fields, extraocular movements intact bilaterally, no nystagmus (II, III, IV, VI).\n
PERRLA, no eyelid ptosis (III), Facial sensation intact bilaterally, able to clench jaw bilaterally (V)\n
Symmetric facial muscle movements, no droop (VII).\n
Hearing grossly normal with a conversational tone (VIII). Phonation intact (IX, X).\n
Shoulder shrug and head rotation intact bilaterally (XI). Tongue protrusion intact (XII).\n
Motor: 5/5 power bilaterally in major joints of upper and lower extremities. Normal tone without any involuntary movements (i.e. tremors, jerks,etc).\n
Sensation: Intact sensation bilaterally in upper and lower extremities.\n
Reflexes: 2+ biceps, patellar, achilles bilaterally.\n
Cerebellar: Normal gait. Normal rapid alternating movements. No postural instability.\n
No basal skull fracture signs.\n
No C-spine tenderness, range of motion adequate.\n
Oropharynx: Normal, moist oropharynx.\n
Cardio: Normal heart sounds without murmur.\n
Resp: Lungs clear bilaterally without adventitious sounds.\n
Abdo: Soft, non-tender.\n
Lower ext: No edema bilaterally.""",

            'Ocular': """OCULAR\n
            General: Appears well.\n
Neuro: Alert, normal phonation. moving extremities x4.\n
Eyes: Acuity Rt 20/20, Lt 20/20.\n
PERLLA without RAPD.\n
Normal visual fields. Normal EOMs and alignment.\n
No injection. No exudate. No hyphema, hypopyon, or iritis. No anterior chamber cells or flares.\n
Lids: No stye. Eversion of lids do not reveal any lesion or FB.\n
Fluoroscein: no abnormal uptake, FB or extravasation of fluid.\n
Tonometry: deferred.""",

            'Trauma': """TRAUMA\n
            General: Appears well.\n
Neuro: GCS 15, normal phonation. moving extremities x4.\n
HEENT: No sign of scalp, facial, or jaw injury. Orbits/eyes appear grossly intact.\n
PERLLA. No hyphema.\n
No signs of basilar, open or depressed skull fracture.\n
No evidence of oropharyngeal injury.\n
No facial bleeding or discharge\n
Neck: Trachea midline.\n
No jugular venous distension.\n
No C-spine tenderness, range of motion intact.\n
Cardio: Normal heart sounds without murmur.\n
Resp: Lungs clear bilaterally.\n
No increased work of breathing.\n
Chest: No sign of injury.\n
No subcutaneous emphysema.\n
Symmetric respirations.\n
Abdo: No sign of injury. Soft, non-tender.\n
Pelvis: Stable. No sign of injury.\n
Back: No step deformity or spinal tenderness T-L-S spine. No other signs of injury.\n
Rt upper ext: No sign of injury. Range of motion, neurovascular status intact.\n
Lt upper ext: No sign of injury. Range of motion, neurovascular status intact.\n
Rt lower ext: No sign of injury. Range of motion, neurovascular status intact.\n
Lt lower ext: No sign of injury. Range of motion, neurovascular status intact.\n
GU and Rectal: Not assessed.\n
E-FAST: Not assessed.""",

            'Pediatrics': """Pediatrics\n
            General: Appears well. Normal skin turgor. No jaundice.\n
Normal cap refill in extremities.\n
Neuro: Alert, moving all extremities, good tone and reflexes x4, pupils are equal, round, and reactive to light and accommodation.\n
HEENT: No ocular injection or exudate.\n
Normal, moist oropharynx. No cracked lips or strawberry tongue.\n
No tonsillar swelling or exudate.\n
Normal tympanic membranes bilaterally.\n
No cervical lymphadenopathy.\n
Neck supple without cervical spine tenderness.\n
Cardio: Normal heart sounds without murmur.\n
Resp: Lungs clear bilaterally without adventitious sounds.\n
Abdo: Soft, non-tender.\n
Extremities: No edema bilaterally. No rash, dactylitis or desquamation."""
        }
    
    def calculate_match_score(self, text: str, keywords: List[str]) -> float:
        """
        Calculate how well the text matches a set of keywords.
        
        Args:
            text (str): The transcript text to analyze
            keywords (List[str]): List of keywords to match against
            
        Returns:
            float: Weighted match score normalized by text length
        """
        text_lower = text.lower()
        weighted_score = 0
        
        for keyword in keywords:
            # Count occurrences of each keyword
            count = text_lower.count(keyword.lower())
            if count > 0:
                # Give higher weight to longer, more specific keywords
                weight = len(keyword.split())
                weighted_score += count * weight
        
        # Normalize by text length to avoid bias toward longer texts
        text_words = len(text.split())
        if text_words == 0:
            return 0
        
        return weighted_score / text_words
    
    def find_best_template(self, transcript: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Find the best matching template for the given transcript.
        
        Args:
            transcript (str): The patient transcript to analyze
            
        Returns:
            Tuple containing:
                - best_template (str): Name of the best matching template
                - best_score (float): Confidence score of the best match
                - all_scores (Dict[str, float]): Scores for all templates
        """
        best_template = 'Normal'
        best_score = 0
        scores = {}
        
        # Calculate scores for all template types
        for template_name, keywords in self.template_keywords.items():
            score = self.calculate_match_score(transcript, keywords)
            scores[template_name] = score
            
            if score > best_score:
                best_score = score
                best_template = template_name
        
        # If no significant matches found, use Normal template
        if best_score < 0.01:  # Threshold for minimum match confidence
            best_template = 'Normal'
        
        return best_template, best_score, scores
    
    def get_template(self, template_name: str) -> str:
        """
        Get the template text for a given template name.
        
        Args:
            template_name (str): Name of the template to retrieve
            
        Returns:
            str: The template text, or Normal template if not found
        """
        return self.templates.get(template_name, self.templates['Normal'])
    
    def analyze_transcript(self, transcript: str) -> Dict:
        """
        Analyze transcript and return detailed results.
        
        Args:
            transcript (str): The patient transcript to analyze
            
        Returns:
            Dict containing analysis results including best template, confidence, etc.
        """
        best_template, confidence, all_scores = self.find_best_template(transcript)
        
        return {
            'best_template': best_template,
            'confidence': confidence,
            'template_text': self.get_template(best_template),
            'all_scores': all_scores,
            'transcript_length': len(transcript.split()),
            'top_matches': dict(sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:3])
        }
    
    def get_available_templates(self) -> List[str]:
        """
        Get a list of all available template names.
        
        Returns:
            List[str]: List of template names
        """
        return list(self.templates.keys())
    
    def add_keywords(self, template_name: str, new_keywords: List[str]) -> bool:
        """
        Add new keywords to an existing template category.
        
        Args:
            template_name (str): Name of the template category
            new_keywords (List[str]): List of new keywords to add
            
        Returns:
            bool: True if successful, False if template doesn't exist
        """
        if template_name in self.template_keywords:
            self.template_keywords[template_name].extend(new_keywords)
            return True
        return False

# Convenience function for easy import
def create_template_mapper() -> TemplateMapper:
    """
    Factory function to create a new TemplateMapper instance.
    
    Returns:
        TemplateMapper: A new instance of the template mapper
    """
    return TemplateMapper()

# Example usage and testing
if __name__ == "__main__":
    # Test cases for different medical scenarios
    test_cases = [
        "Patient fell down the stairs and hit their head, complaining of pain and bleeding",
        "Patient has severe headache and dizziness, feeling confused and nauseous",
        "Patient has lower back pain radiating down the leg, can't bend over or walk properly",
        "Patient has ear pain and sore throat for 3 days, difficulty swallowing",
        "Patient is feeling anxious and having panic attacks, can't sleep",
        "Patient has knee pain and swelling after sports injury, can't put weight on it",
        "Patient has blurry vision and eye pain, light sensitivity",
        "Patient has testicular pain and burning urination, blood in urine",
        "24 years old male picks up by EMS outside bar after being stabbed in the chest with a knife during a bar brawl patient was found sitting on the curve clutching his chest and complaining of pain patient denied any medical problems but admitted to having several drinks during the course of the evening"
    ]
    
    # Create mapper and test
    mapper = create_template_mapper()
    
    print("=== Template Mapping Test Results ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"Input: {test_case}")
        
        result = mapper.analyze_transcript(test_case)
        
        print(f"Best Template: {result['best_template']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Top 3 matches: {result['top_matches']}")
        print("-" * 60)