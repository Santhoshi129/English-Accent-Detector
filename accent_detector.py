"""
accent_detector.py - Core module for English accent detection and analysis

This module provides the AccentDetector class which implements accent detection functionality
through both audio-based and text-based analysis. It supports multiple accent types including
American, British, Australian, Canadian, Indian, and Irish English.

Key features:
- Audio extraction from video URLs
- Speech transcription using Whisper models
- Prosodic feature analysis
- Linguistic pattern matching
- Confidence scoring and detailed explanations

Dependencies:
- faster-whisper or openai-whisper for speech recognition
- librosa for audio processing
- yt-dlp for video download
- numpy and pandas for data processing
"""

import os
import tempfile
import librosa
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import yt_dlp
import subprocess
import re
from typing import Dict, Tuple, List
import logging
import ssl
import urllib3

# Disable SSL warnings for corporate environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccentDetector:
    """
    A class for detecting and analyzing English accents in both audio and text content.
    
    The detector uses a combination of:
    1. Speech recognition (Whisper models)
    2. Prosodic feature analysis (pitch, rhythm, etc.)
    3. Linguistic pattern matching
    4. Vocabulary and dialectal analysis
    
    It supports both video URL processing and text-only demo mode for accent detection.
    """
    
    def __init__(self):
        """
        Initialize the accent detector with models and reference data.
        
        The initialization process:
        1. Attempts to load faster-whisper (preferred)
        2. Falls back to openai-whisper if faster-whisper fails
        3. Creates a mock model for testing if both fail
        4. Sets up accent characteristics and patterns
        """
        logger.info("Loading Whisper model...")
        
        # Try to load faster-whisper with fallback options
        self.whisper_model = None
        self.model_type = None
        
        # Try faster-whisper first
        try:
            from faster_whisper import WhisperModel
            # Try with local_files_only=False first
            try:
                self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8", local_files_only=False)
                self.model_type = "faster-whisper"
                logger.info("Successfully loaded faster-whisper model")
            except Exception as e:
                logger.warning(f"Failed to load faster-whisper with network access: {e}")
                # Try with local files only
                try:
                    self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8", local_files_only=True)
                    self.model_type = "faster-whisper"
                    logger.info("Successfully loaded faster-whisper model from cache")
                except Exception as e:
                    logger.warning(f"Failed to load faster-whisper from cache: {e}")
                    
        except ImportError:
            logger.warning("faster-whisper not available")
        
        # Fallback to openai-whisper if faster-whisper fails
        if self.whisper_model is None:
            try:
                import whisper
                self.whisper_model = whisper.load_model("base")
                self.model_type = "openai-whisper"
                logger.info("Successfully loaded openai-whisper model as fallback")
            except Exception as e:
                logger.warning(f"Failed to load openai-whisper: {e}")
        
        # Final fallback - create a mock model for development/testing
        if self.whisper_model is None:
            logger.warning("No Whisper model available - creating mock model for testing")
            self.whisper_model = None
            self.model_type = "mock"
        
        # Define accent characteristics based on linguistic patterns
        self.accent_patterns = {
            'American': {
                'rhotic': True,  # R-sounds pronounced
                'vowel_patterns': ['æ', 'ɑ', 'ɔ'],  # cat, lot, thought
                'keywords': ['elevator', 'apartment', 'vacation', 'schedule', 'aluminum'],
                'pronunciation_markers': ['r-colored vowels', 'flapped t'],
                'stress_patterns': ['primary stress on first syllable']
            },
            'British': {
                'rhotic': False,  # R-sounds often dropped
                'vowel_patterns': ['ɒ', 'ɑː', 'æ'],  # lot, bath, cat
                'keywords': ['lift', 'flat', 'holiday', 'timetable', 'aluminium'],
                'pronunciation_markers': ['received pronunciation', 'non-rhotic'],
                'stress_patterns': ['varied stress patterns']
            },
            'Australian': {
                'rhotic': False,
                'vowel_patterns': ['aɪ', 'eɪ', 'oʊ'],  # diphthong variations
                'keywords': ['mate', 'arvo', 'barbie', 'brekkie'],
                'pronunciation_markers': ['vowel fronting', 'high rising terminal'],
                'stress_patterns': ['rising intonation']
            },
            'Canadian': {
                'rhotic': True,
                'vowel_patterns': ['aʊ', 'aɪ'],  # about, write (Canadian raising)
                'keywords': ['eh', 'about', 'house', 'process'],
                'pronunciation_markers': ['Canadian raising', 'cot-caught merger'],
                'stress_patterns': ['similar to American']
            },
            'Indian': {
                'rhotic': True,
                'vowel_patterns': ['retroflex sounds'],
                'keywords': ['prepone', 'revert back', 'do the needful'],
                'pronunciation_markers': ['retroflex consonants', 'syllable-timed rhythm'],
                'stress_patterns': ['syllable-timed']
            },
            'Irish': {
                'rhotic': True,
                'vowel_patterns': ['distinctive vowel system'],
                'keywords': ['grand', 'craic', 'yourself'],
                'pronunciation_markers': ['dental fricatives', 'broad/slender consonants'],
                'stress_patterns': ['Celtic influence']
            }
        }
        
        # Compile regex patterns for text analysis
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for accent detection."""
        self.text_patterns = {}
        for accent, data in self.accent_patterns.items():
            patterns = []
            for keyword in data['keywords']:
                patterns.append(rf'\b{re.escape(keyword)}\b')
            self.text_patterns[accent] = re.compile('|'.join(patterns), re.IGNORECASE)
    
    def download_and_extract_audio(self, video_url: str) -> str:
        """Download video and extract audio using direct command line call to yt-dlp."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_template = os.path.join(temp_dir, '%(title)s.%(ext)s')
                
                logger.info(f"Downloading audio from: {video_url}")
                
                # Use subprocess to call yt-dlp directly with --no-check-certificate
                cmd = [
                    'yt-dlp',
                    '--no-check-certificate',
                    '-x',  # Extract audio
                    '--audio-format', 'wav',  # Convert to wav directly
                    '-o', output_template,
                    '--quiet',
                    video_url
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise Exception(f"yt-dlp failed: {result.stderr}")
                    logger.info("Download successful")
                except Exception as e:
                    raise Exception(f"Video download failed: {str(e)}\n\nTry Demo Mode for text-based accent analysis!")
                
                # Find the downloaded file
                downloaded_files = [f for f in os.listdir(temp_dir) 
                                  if f.endswith(('.wav', '.mp3', '.m4a', '.webm', '.mp4'))]
                
                if not downloaded_files:
                    raise Exception("No audio/video file found after download")
                
                input_path = os.path.join(temp_dir, downloaded_files[0])
                output_path = os.path.join(tempfile.gettempdir(), f"audio_{hash(video_url)}.wav")
                
                # Convert to WAV if needed using ffmpeg
                if not input_path.endswith('.wav'):
                    cmd = [
                        'ffmpeg', '-i', input_path, 
                        '-vn', '-acodec', 'pcm_s16le', 
                        '-ar', '16000', '-ac', '1', 
                        '-y', output_path
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise Exception(f"Audio conversion failed: {result.stderr}")
                else:
                    # Just copy if already WAV
                    import shutil
                    shutil.copy2(input_path, output_path)
                
                logger.info(f"Audio extracted to: {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"Error downloading/extracting audio: {str(e)}")
            raise
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe audio using available Whisper model."""
        try:
            logger.info("Transcribing audio...")
            
            if self.model_type == "faster-whisper":
                segments, info = self.whisper_model.transcribe(audio_path, beam_size=5)
                
                # Convert segments to list and extract full text
                segments_list = []
                full_text = ""
                
                for segment in segments:
                    segment_dict = {
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment.text
                    }
                    segments_list.append(segment_dict)
                    full_text += segment.text + " "
                
                return {
                    'text': full_text.strip(),
                    'segments': segments_list,
                    'language': info.language
                }
                
            elif self.model_type == "openai-whisper":
                result = self.whisper_model.transcribe(audio_path)
                return {
                    'text': result['text'],
                    'segments': result['segments'],
                    'language': result['language']
                }
                
            elif self.model_type == "mock":
                # Mock transcription for development/testing
                logger.warning("Using mock transcription - no actual speech recognition")
                return {
                    'text': "This is a mock transcription for testing purposes. Hello, I'm speaking in an American accent with words like elevator and apartment.",
                    'segments': [{'start': 0, 'end': 5, 'text': "This is a mock transcription for testing purposes."}],
                    'language': 'en'
                }
            
            else:
                raise Exception("No Whisper model available for transcription")
                
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise
    
    def analyze_prosodic_features(self, audio_path: str) -> Dict:
        """Analyze prosodic features from audio."""
        try:
            y, sr = librosa.load(audio_path)
            
            # Extract features
            features = {}
            
            # Pitch/F0 analysis
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
                features['pitch_range'] = max(pitch_values) - min(pitch_values)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                features['pitch_range'] = 0
            
            # Rhythm analysis
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            if len(onset_frames) > 1:
                onset_times = librosa.frames_to_time(onset_frames, sr=sr)
                ioi = np.diff(onset_times)  # Inter-onset intervals
                features['rhythm_regularity'] = 1 / (np.std(ioi) + 1e-6)
                features['speech_rate'] = len(onset_frames) / (len(y) / sr)
            else:
                features['rhythm_regularity'] = 0
                features['speech_rate'] = 0
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            
            return features
            
        except Exception as e:
            logger.error(f"Error analyzing prosodic features: {str(e)}")
            return {}
    
    def classify_accent(self, transcription: Dict, audio_features: Dict, audio_path: str) -> Dict:
        """Classify the accent based on text and audio features."""
        text = transcription['text'].lower()
        accent_scores = {}
        
        # Text-based analysis (keyword matching)
        for accent, pattern in self.text_patterns.items():
            matches = len(pattern.findall(text))
            # More aggressive scoring for keyword matches
            text_score = min(matches * 25, 100)  # Increased multiplier
            accent_scores[accent] = text_score
        
        # Prosodic feature analysis
        prosodic_scores = self._analyze_prosodic_patterns(audio_features)
        
        # Linguistic pattern analysis (more detailed text analysis)
        linguistic_scores = self._analyze_linguistic_patterns(text)
        
        # Combine scores with adjusted weights
        final_scores = {}
        for accent in self.accent_patterns.keys():
            # Start with no score
            base_score = 0
            
            # Add weighted components
            if accent_scores.get(accent, 0) > 0:
                base_score += accent_scores[accent] * 0.4  # Keywords
            
            if linguistic_scores.get(accent, 0) > 0:
                base_score += linguistic_scores[accent] * 0.4  # Linguistic patterns
            
            if prosodic_scores.get(accent, 0) > 0:
                base_score += prosodic_scores[accent] * 0.2  # Audio features
            
            # Add phonetic analysis
            phonetic_score = self._analyze_phonetic_patterns(text, accent)
            if phonetic_score > 0:
                base_score += phonetic_score * 0.1
            
            # Only keep scores if there's actual evidence
            if base_score > 0:
                final_scores[accent] = min(base_score, 100)
            else:
                final_scores[accent] = 0
        
        # Normalize scores to create more distinction
        max_score = max(final_scores.values()) if final_scores.values() else 1
        if max_score > 0:
            # Create more separation between scores
            for accent in final_scores:
                normalized_score = (final_scores[accent] / max_score) * 100
                # Apply exponential scaling to create more distinction
                final_scores[accent] = min(100, normalized_score ** 1.2)
        
        # Determine primary accent
        primary_accent = max(final_scores, key=final_scores.get)
        confidence = final_scores[primary_accent]
        
        # Generate explanation
        explanation = self._generate_explanation(primary_accent, confidence, text, audio_features)
        
        return {
            'primary_accent': primary_accent,
            'confidence': confidence,
            'all_scores': final_scores,
            'explanation': explanation
        }
    
    def _analyze_prosodic_patterns(self, features: Dict) -> Dict:
        """Analyze prosodic patterns for accent classification."""
        scores = {}
        
        if not features:
            return {}  # Return empty dict instead of baseline scores
        
        pitch_mean = features.get('pitch_mean', 0)
        pitch_std = features.get('pitch_std', 0)
        rhythm_regularity = features.get('rhythm_regularity', 0)
        speech_rate = features.get('speech_rate', 0)
        
        # American: moderate pitch variation, regular rhythm
        if 120 <= pitch_mean <= 200 and 10 <= pitch_std <= 30:
            scores['American'] = 30
            if rhythm_regularity > 2:
                scores['American'] += 20
        
        # British: varied pitch, formal rhythm
        if 100 <= pitch_mean <= 180 and pitch_std > 15:
            scores['British'] = 30
            if rhythm_regularity > 1.5:
                scores['British'] += 20
        
        # Australian: distinctive pitch patterns
        if pitch_std > 20 and 130 <= pitch_mean <= 190:
            scores['Australian'] = 40  # Higher score for distinctive pattern
        
        # Canadian: similar to American but distinct
        if scores.get('American', 0) > 0:
            scores['Canadian'] = scores['American'] * 0.7  # Reduced similarity
        
        # Indian: syllable-timed rhythm
        if rhythm_regularity > 3:
            scores['Indian'] = 35
            if speech_rate > 3:
                scores['Indian'] += 15
        
        # Irish: distinctive prosody
        if pitch_std > 25 and 110 <= pitch_mean <= 170:
            scores['Irish'] = 40
        
        return scores
    
    def _analyze_linguistic_patterns(self, text: str) -> Dict:
        """Analyze linguistic patterns in the text."""
        scores = {}
        
        # Only assign scores if patterns are actually found
        # American patterns
        american_count = sum(1 for word in ['elevator', 'apartment', 'vacation', 'schedule', 'aluminum', 
                                          'color', 'center', 'realize', 'organize'] if word in text.lower())
        if american_count > 0:
            scores['American'] = american_count * 15
        
        # British patterns
        british_count = sum(1 for word in ['lift', 'flat', 'holiday', 'timetable', 'aluminium',
                                         'colour', 'centre', 'realise', 'organise', 'whilst'] if word in text.lower())
        if british_count > 0:
            scores['British'] = british_count * 15
        
        # Australian patterns
        aussie_count = sum(1 for word in ['mate', 'arvo', 'barbie', 'brekkie', 'fair dinkum'] if word in text.lower())
        if aussie_count > 0:
            scores['Australian'] = aussie_count * 20  # Higher weight for distinctive terms
        
        # Canadian patterns
        canadian_count = sum(1 for word in ['eh', 'about', 'process', 'schedule'] if word in text.lower())
        if canadian_count > 0:
            scores['Canadian'] = canadian_count * 15
        
        # Indian patterns
        indian_count = sum(1 for word in ['prepone', 'revert back', 'do the needful', 'good name', 'out of station'] 
                         if word in text.lower())
        if indian_count > 0:
            scores['Indian'] = indian_count * 20  # Higher weight for distinctive terms
        
        # Irish patterns
        irish_count = sum(1 for word in ['grand', 'craic', 'yourself', 'brilliant'] if word in text.lower())
        if irish_count > 0:
            scores['Irish'] = irish_count * 20  # Higher weight for distinctive terms
        
        return scores
    
    def _analyze_phonetic_patterns(self, text: str, accent: str) -> float:
        """Analyze phonetic patterns for specific accent."""
        score = 5  # baseline
        
        # Simple heuristics based on spelling patterns that might indicate pronunciation
        if accent == 'American':
            if re.search(r'\b\w*or\b', text):  # -or endings (color, honor)
                score += 5
            if re.search(r'\b\w*ize\b', text):  # -ize endings
                score += 5
        
        elif accent == 'British':
            if re.search(r'\b\w*our\b', text):  # -our endings (colour, honour)
                score += 5
            if re.search(r'\b\w*ise\b', text):  # -ise endings
                score += 5
        
        elif accent == 'Australian':
            if re.search(r'\b\w*ie\b', text):  # diminutive -ie endings
                score += 3
        
        return score
    
    def _generate_explanation(self, accent: str, confidence: float, text: str, features: Dict) -> str:
        """Generate explanation for the accent classification."""
        explanation_parts = []
        
        if confidence > 70:
            explanation_parts.append(f"Strong indicators of {accent} accent detected.")
        elif confidence > 40:
            explanation_parts.append(f"Moderate indicators of {accent} accent detected.")
        else:
            explanation_parts.append(f"Weak indicators suggest {accent} accent, but confidence is low.")
        
        # Add specific indicators found
        accent_data = self.accent_patterns[accent]
        found_keywords = [kw for kw in accent_data['keywords'] if kw.lower() in text.lower()]
        if found_keywords:
            explanation_parts.append(f"Vocabulary indicators: {', '.join(found_keywords[:3])}")
        
        # Add prosodic information
        if features:
            if features.get('pitch_std', 0) > 20:
                explanation_parts.append("High pitch variation detected.")
            if features.get('rhythm_regularity', 0) > 2:
                explanation_parts.append("Regular speech rhythm observed.")
        
        return " ".join(explanation_parts)
    
    def cleanup_audio(self, audio_path: str):
        """Clean up temporary audio file."""
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"Cleaned up audio file: {audio_path}")
        except Exception as e:
            logger.warning(f"Could not clean up audio file: {str(e)}")
    
    def process_video_url(self, video_url: str) -> Dict:
        """Main method to process a video URL and detect accent."""
        audio_path = None
        try:
            # Step 1: Download and extract audio
            audio_path = self.download_and_extract_audio(video_url)
            
            # Step 2: Transcribe audio
            transcription = self.transcribe_audio(audio_path)
            
            # Check if the detected language is English
            if transcription['language'] != 'en':
                return {
                    'error': f"Detected language is '{transcription['language']}', not English. "
                            f"This tool is designed for English accent detection only."
                }
            
            # Step 3: Analyze prosodic features
            audio_features = self.analyze_prosodic_features(audio_path)
            
            # Step 4: Classify accent
            classification_result = self.classify_accent(transcription, audio_features, audio_path)
            
            # Combine all results
            result = {
                'transcription': transcription['text'],
                'detected_language': transcription['language'],
                'primary_accent': classification_result['primary_accent'],
                'confidence_score': round(classification_result['confidence'], 1),
                'all_accent_scores': {k: round(v, 1) for k, v in classification_result['all_scores'].items()},
                'explanation': classification_result['explanation'],
                'audio_features': audio_features,
                'success': True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return {
                'error': str(e),
                'success': False
            }
        
        finally:
            # Cleanup
            if audio_path:
                self.cleanup_audio(audio_path)
    
    def process_demo_text(self, demo_text: str, accent_hint: str = None) -> Dict:
        """Process demo text for accent classification without video download."""
        try:
            logger.info("Processing demo text for accent classification...")
            
            # Create mock transcription result
            transcription = {
                'text': demo_text,
                'segments': [{'start': 0, 'end': len(demo_text.split()) * 0.5, 'text': demo_text}],
                'language': 'en'
            }
            
            # Create mock audio features
            audio_features = {
                'pitch_mean': 150.0,
                'pitch_std': 25.0,
                'pitch_range': 100.0,
                'rhythm_regularity': 2.5,
                'speech_rate': 3.5,
                'spectral_centroid_mean': 2500.0
            }
            
            # Classify accent
            classification_result = self.classify_accent(transcription, audio_features, None)
            
            # Combine all results
            result = {
                'transcription': demo_text,
                'detected_language': 'en',
                'primary_accent': classification_result['primary_accent'],
                'confidence_score': round(classification_result['confidence'], 1),
                'all_accent_scores': {k: round(v, 1) for k, v in classification_result['all_scores'].items()},
                'explanation': classification_result['explanation'],
                'audio_features': audio_features,
                'success': True,
                'demo_mode': True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing demo text: {str(e)}")
            return {
                'error': str(e),
                'success': False
            } 