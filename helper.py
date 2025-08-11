import torch
import whisper
from pyannote.audio import Pipeline
from pyannote.core import Segment
import librosa
import numpy as np
from typing import Dict, List, Tuple
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pydub import AudioSegment
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

HF_TOKEN = "HF_TOKEN"
WHISPER_MODEL = "medium"
AUDIO_FILE = None

#============================================================================================================
# Colab GPU Speech Diarization Class
#============================================================================================================

class ColabGPUSpeechDiarizer:
    def __init__(self, whisper_model_size="base", hf_token=None):
        """
        Args:
            whisper_model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
        """
        self.device = self.setup_gpu()
        self.whisper_model_size = whisper_model_size
        self.hf_token = hf_token

        self.load_models()

    def setup_gpu(self):

        print("\n🔧 Setting up GPU configuration...")

        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

            print(f"✅ GPU Detected: {gpu_name} ({gpu_memory:.1f} GB)")
            print(f"✅ CUDA Version: {torch.version.cuda}")

            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.cuda.empty_cache()

            print("✅ GPU optimizations enabled")
            return device
        else:
            print("⚠️  No GPU detected. Using CPU (will be slower)")
            return torch.device("cpu")

    def load_models(self):
        print(f"\n🤖 Loading Whisper model ({self.whisper_model_size}) on GPU...")

        self.whisper_model = whisper.load_model(self.whisper_model_size)
        if self.device.type == "cuda":
            self.whisper_model = self.whisper_model.cuda()

        print("✅ Whisper model loaded on GPU")

        if self.hf_token:
            print("🎯 Loading speaker diarization pipeline...")
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.hf_token
                )
                if self.device.type == "cuda":
                    self.diarization_pipeline.to(self.device)
                print("✅ Diarization pipeline loaded on GPU")
            except Exception as e:
                print(f"❌ Error loading diarization pipeline: {e}")
                print("💡 Make sure you have a valid Hugging Face token")
                self.diarization_pipeline = None
        else:
            print("⚠️  No HF token provided. Using fallback diarization.")
            self.diarization_pipeline = None

    def monitor_gpu_usage(self):
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"📊 GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    def preprocess_audio(self, audio_path: str) -> str:
        print(f"🔧 Preprocessing audio file: {audio_path}")

        try:
            audio = AudioSegment.from_file(audio_path)

            if audio.channels > 1:
                audio = audio.set_channels(1)
                print("✅ Converted to mono")

            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000)
                print(f"✅ Resampled to 16kHz (was {audio.frame_rate}Hz)")

            audio = audio.set_sample_width(2)

            temp_path = "/tmp/preprocessed_audio.wav"
            audio.export(temp_path, format="wav")

            print(f"✅ Preprocessed audio saved to {temp_path}")
            return temp_path

        except Exception as e:
            print(f"⚠️  Preprocessing failed: {e}")
            return audio_path

    def extract_audio_segment(self, audio_path: str, start_time: float, end_time: float) -> np.ndarray:
        """Extract audio segment with proper padding and length handling"""
        try:
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)

            start_sample = max(0, int(start_time * sr))
            end_sample = min(len(audio), int(end_time * sr))

            segment = audio[start_sample:end_sample]

            min_length = 8000
            target_length = 160000

            if len(segment) < min_length:
                padding = min_length - len(segment)
                segment = np.pad(segment, (0, padding), mode='constant', constant_values=0)
            elif len(segment) > target_length:
                segment = segment[:target_length]

            if np.max(np.abs(segment)) > 0:
                segment = segment / np.max(np.abs(segment)) * 0.95

            return segment.astype(np.float32)

        except Exception as e:
            print(f"⚠️  Error extracting segment: {e}")
            return np.zeros(8000, dtype=np.float32)

    def transcribe_segment_gpu(self, audio_segment: np.ndarray) -> str:
        if len(audio_segment) < 8000:
            return ""

        try:
            audio_segment = audio_segment.astype(np.float32)

            if np.max(np.abs(audio_segment)) < 1e-6:
                return ""

            max_val = np.max(np.abs(audio_segment))
            if max_val > 0:
                audio_segment = audio_segment / max_val * 0.95

            with torch.no_grad():
                result = self.whisper_model.transcribe(
                    audio_segment,
                    fp16=True if self.device.type == "cuda" else False,
                    verbose=False,
                    temperature=0.0,
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=False
                )

            text = result["text"].strip()

            if len(text) < 3 or text.lower() in ["", ".", "?", "!", "the", "a", "an"]:
                return ""

            return text

        except Exception as e:
            print(f"⚠️  Transcription error for segment: {e}")
            return ""

    def diarize_and_transcribe_gpu(self, audio_path: str) -> Dict[str, List[Dict]]:
        print(f"\n🎵 Processing audio file: {audio_path}")

        processed_audio_path = self.preprocess_audio(audio_path)

        try:
            audio_info = librosa.get_duration(filename=processed_audio_path)
            print(f"📊 Audio duration: {audio_info:.2f} seconds")

            if audio_info < 1.0:
                print("⚠️  Audio is very short, may not work well with diarization")

        except Exception as e:
            print(f"⚠️  Could not read audio duration: {e}")

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        results = {}

        if self.diarization_pipeline:
            try:
                print("🎯 Performing speaker diarization...")

                try:
                    diarization = self.diarization_pipeline(processed_audio_path)
                except Exception as diar_error:
                    print(f"❌ Diarization error: {diar_error}")
                    if "tensor" in str(diar_error).lower() or "size" in str(diar_error).lower():
                        print("🔄 Tensor size mismatch - trying fallback...")
                        return self.simple_segmentation_gpu(processed_audio_path)
                    else:
                        raise diar_error

                speaker_segments = {}
                segment_count = 0
                processed_segments = 0

                print("📝 Transcribing segments...")
                for segment, _, speaker in diarization.itertracks(yield_label=True):
                    segment_count += 1

                    if segment.end - segment.start < 1.0:
                        continue

                    if speaker not in speaker_segments:
                        speaker_segments[speaker] = []

                    try:
                        audio_segment = self.extract_audio_segment(
                            processed_audio_path, segment.start, segment.end
                        )

                        if len(audio_segment) > 0:
                            transcript = self.transcribe_segment_gpu(audio_segment)

                            if transcript and len(transcript.strip()) > 2:
                                speaker_segments[speaker].append({
                                    'start_time': round(segment.start, 2),
                                    'end_time': round(segment.end, 2),
                                    'transcript': transcript,
                                    'duration': round(segment.end - segment.start, 2)
                                })
                                processed_segments += 1
                                print(f"   ✅ Segment {processed_segments}: {speaker} - {transcript[:50]}...")
                            else:
                                print(f"   ⚠️  Segment {segment_count}: No valid speech")

                        if segment_count % 5 == 0:
                            print(f"   📊 Processed {processed_segments} valid segments from {segment_count} total")
                            if self.device.type == "cuda":
                                torch.cuda.empty_cache()

                    except Exception as e:
                        print(f"⚠️  Error processing segment {segment_count}: {e}")
                        continue

                # Remove speakers with no valid segments
                results = {k: v for k, v in speaker_segments.items() if v}

                if not results:
                    print("⚠️  No valid segments found. Trying fallback method...")
                    results = self.simple_segmentation_gpu(processed_audio_path)

            except Exception as e:
                print(f"❌ Diarization error: {e}")
                print("🔄 Falling back to simple segmentation...")
                results = self.simple_segmentation_gpu(processed_audio_path)
        else:
            # Fallback: Simple time-based segmentation
            print("🔄 Using fallback segmentation (no speaker diarization)...")
            results = self.simple_segmentation_gpu(processed_audio_path)

        if processed_audio_path != audio_path and os.path.exists(processed_audio_path):
            os.remove(processed_audio_path)

        print(f"✅ Processing complete! Found {len(results)} speakers with valid segments")
        return results

    def simple_segmentation_gpu(self, audio_path: str, segment_duration: float = 20.0) -> Dict[str, List[Dict]]:
        """
        Fallback method: Simple time-based segmentation with GPU transcription
        """
        try:
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            duration = len(audio) / sr

            print(f"📊 Audio loaded: {duration:.2f} seconds, {len(audio)} samples")

            if duration < 1.0:
                print("⚠️  Audio too short for segmentation")
                return {"Speaker_Unknown": []}

            segments = []
            current_time = 0
            segment_num = 0

            while current_time < duration:
                end_time = min(current_time + segment_duration, duration)
                segment_num += 1

                if end_time - current_time < 1.0:
                    current_time = end_time
                    continue

                try:
                    start_sample = int(current_time * sr)
                    end_sample = int(end_time * sr)

                    end_sample = min(end_sample, len(audio))

                    if end_sample > start_sample:
                        audio_segment = audio[start_sample:end_sample]

                        if len(audio_segment) >= 8000:
                            if np.max(np.abs(audio_segment)) > 0:
                                audio_segment = audio_segment / np.max(np.abs(audio_segment)) * 0.95

                            transcript = self.transcribe_segment_gpu(audio_segment)

                            if transcript and len(transcript.strip()) > 2:
                                segments.append({
                                    'start_time': round(current_time, 2),
                                    'end_time': round(end_time, 2),
                                    'transcript': transcript,
                                    'duration': round(end_time - current_time, 2)
                                })
                                print(f"   ✅ Segment {segment_num}: {transcript[:50]}...")
                            else:
                                print(f"   ⚠️  Segment {segment_num}: No speech detected")
                        else:
                            print(f"   ⚠️  Segment {segment_num}: Too short ({len(audio_segment)} samples)")

                except Exception as e:
                    print(f"   ❌ Error processing segment {segment_num}: {e}")

                current_time = end_time

                if segment_num % 3 == 0 and self.device.type == "cuda":
                    torch.cuda.empty_cache()

            if not segments:
                print("⚠️  No valid segments found in fallback method")
                return {"Speaker_Unknown": []}

            return {"Speaker_Unknown": segments}

        except Exception as e:
            print(f"❌ Error in simple segmentation: {e}")
            return {"Speaker_Unknown": []}

    def format_output(self, speaker_segments: Dict[str, List[Dict]]) -> str:
        """Format output with enhanced styling"""
        output = []
        output.append("🎙️ " + "=" * 60)
        output.append("    SPEAKER DIARIZATION & TRANSCRIPTION RESULTS")
        output.append("🎙️ " + "=" * 60)

        total_speakers = len(speaker_segments)
        total_segments = sum(len(segments) for segments in speaker_segments.values())

        output.append(f"\n📊 Summary: {total_speakers} speakers, {total_segments} segments")
        output.append("")

        for speaker_id, segments in speaker_segments.items():
            output.append(f"🎤 {speaker_id.upper()}:")
            output.append("─" * 50)

            total_duration = sum(seg['duration'] for seg in segments)
            output.append(f"⏱️  Total speaking time: {total_duration:.1f} seconds")
            output.append("")

            for i, segment in enumerate(segments, 1):
                timestamp = f"[{segment['start_time']:>6.1f}s - {segment['end_time']:>6.1f}s]"
                output.append(f"{i:2d}. {timestamp} {segment['transcript']}")

            combined = ' '.join([seg['transcript'] for seg in segments])
            output.append(f"\n📝 Combined transcript:")
            output.append(f"   {combined}")
            output.append("")

        return "\n".join(output)

    def save_results(self, speaker_segments: Dict[str, List[Dict]], base_filename: str = "diarization_results"):
        """Save results in multiple formats"""

        json_file = f"{base_filename}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(speaker_segments, f, indent=2, ensure_ascii=False)

        txt_file = f"{base_filename}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(self.format_output(speaker_segments))

        for speaker_id, segments in speaker_segments.items():
            speaker_file = f"{base_filename}_{speaker_id}.txt"
            with open(speaker_file, 'w', encoding='utf-8') as f:
                f.write(f"Transcript for {speaker_id.upper()}\n")
                f.write("=" * 50 + "\n\n")

                for segment in segments:
                    timestamp_str = f"[{segment['start_time']:.1f}s - {segment['end_time']:.1f}s]"
                    f.write(f"{timestamp_str} {segment['transcript']}\n")

                f.write(f"\nCombined transcript:\n{'-' * 20}\n")
                combined = ' '.join([seg['transcript'] for seg in segments])
                f.write(combined)

        print(f"💾 Results saved:")
        print(f"   📄 JSON: {json_file}")
        print(f"   📄 Text: {txt_file}")
        print(f"   📄 Individual speaker files created")

        return json_file, txt_file

    def visualize_timeline(self, speaker_segments: Dict[str, List[Dict]]):
        """Create a timeline visualization of speakers"""
        plt.figure(figsize=(15, 6))

        colors = plt.cm.Set3(np.linspace(0, 1, len(speaker_segments)))

        for i, (speaker_id, segments) in enumerate(speaker_segments.items()):
            for segment in segments:
                plt.barh(i, segment['duration'],
                        left=segment['start_time'],
                        height=0.6,
                        color=colors[i],
                        alpha=0.8,
                        label=speaker_id if segment == segments[0] else "")

        plt.xlabel('Time (seconds)')
        plt.ylabel('Speakers')
        plt.title('Speaker Timeline')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

#============================================================================================================
# Emotion Detection and Visualization
#============================================================================================================

class HuggingFaceEmotionAnalyzer:
    def __init__(self, audio_file_path: str, model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        self.audio_file_path = audio_file_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        # Available models - you can change this
        self.available_models = {
            "wav2vec2-emotion": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            "superb-emotion": "superb/wav2vec2-base-superb-er",
            "facebook-emotion": "facebook/wav2vec2-large-xlsr-53-speech-emotion-recognition",
            "harshit345-emotion": "harshit345/xlsr-wav2vec-speech-emotion-recognition",
            "m3hrdadfi-emotion": "m3hrdadfi/wav2vec2-xlsr-greek-speech-emotion-recognition"
        }

        print(f"🤖 Loading emotion detection model: {model_name}")
        self.load_model()

    def load_model(self):
        """Load the Hugging Face emotion detection model"""
        try:
            # Try multiple model loading approaches
            model_loaded = False

            # Method 1: Direct pipeline (most common)
            try:
                self.emotion_classifier = pipeline(
                    "audio-classification",
                    model=self.model_name,
                    device=0 if self.device == "cuda" else -1,
                    return_all_scores=True
                )
                model_loaded = True
                print("✅ Model loaded using pipeline method")
            except Exception as e:
                print(f"⚠️  Pipeline method failed: {e}")

            # Method 2: Manual loading if pipeline fails
            if not model_loaded:
                try:
                    self.processor = AutoProcessor.from_pretrained(self.model_name)
                    self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)

                    if self.device == "cuda":
                        self.model = self.model.to(self.device)

                    self.emotion_classifier = None  # Will use manual method
                    model_loaded = True
                    print("✅ Model loaded using manual method")
                except Exception as e:
                    print(f"⚠️  Manual method failed: {e}")

            # Method 3: Fallback to a different model
            if not model_loaded:
                fallback_models = [
                    "jonatasgrosman/wav2vec2-large-xlsr-53-english",
                    "facebook/wav2vec2-base-960h"
                ]

                for fallback in fallback_models:
                    try:
                        print(f"🔄 Trying fallback model: {fallback}")
                        self.emotion_classifier = pipeline(
                            "audio-classification",
                            model=fallback,
                            device=0 if self.device == "cuda" else -1,
                            return_all_scores=True
                        )
                        model_loaded = True
                        self.model_name = fallback
                        print(f"✅ Fallback model loaded: {fallback}")
                        break
                    except:
                        continue

            if not model_loaded:
                raise Exception("Could not load any emotion detection model")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔄 Using lightweight fallback approach...")
            self.use_fallback_model()

    def use_fallback_model(self):
        """Use a simpler, more reliable model as fallback"""
        try:
            # Use a more basic but reliable model
            self.emotion_classifier = pipeline(
                "audio-classification",
                model="MIT/ast-finetuned-audioset-10-10-0.4593",
                device=0 if self.device == "cuda" else -1
            )
            print("✅ Fallback AST model loaded")
        except Exception as e:
            print(f"❌ Even fallback failed: {e}")
            self.emotion_classifier = None

    def extract_audio_clip(self, start_time: float, end_time: float, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """Extract audio clip for the given time segment"""
        try:
            audio, sr = librosa.load(self.audio_file_path, sr=target_sr, mono=True)
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            clip = audio[start_sample:end_sample]

            # Ensure minimum length for model
            if len(clip) < target_sr * 0.5:  # Minimum 0.5 seconds
                padding = target_sr * 0.5 - len(clip)
                clip = np.pad(clip, (0, int(padding)), mode='constant')

            # Normalize audio
            if np.max(np.abs(clip)) > 0:
                clip = clip / np.max(np.abs(clip)) * 0.95

            return clip, sr
        except Exception as e:
            print(f"❌ Error extracting audio clip: {e}")
            return np.zeros(16000), 16000

    def predict_emotion(self, audio_clip: np.ndarray, sr: int = 16000) -> Dict[str, float]:
        """Predict emotion using the loaded model"""
        if self.emotion_classifier is None:
            return self.fallback_emotion_detection(audio_clip, sr)

        try:
            if hasattr(self, 'processor') and hasattr(self, 'model'):
                # Manual prediction
                inputs = self.processor(audio_clip, sampling_rate=sr, return_tensors="pt", padding=True)

                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

                # Get emotion labels (this varies by model)
                emotion_labels = self.model.config.id2label
                results = []

                for i, score in enumerate(predictions[0]):
                    emotion = emotion_labels.get(i, f"emotion_{i}")
                    results.append({'label': emotion, 'score': float(score)})

            else:
                # Pipeline prediction
                results = self.emotion_classifier(audio_clip)

            # Convert to standardized format
            return self.standardize_emotions(results)

        except Exception as e:
            print(f"❌ Error in emotion prediction: {e}")
            return self.fallback_emotion_detection(audio_clip, sr)

    def standardize_emotions(self, raw_results: List[Dict]) -> Dict[str, float]:
        """Standardize emotion labels across different models"""

        # Mapping from various model outputs to standard emotions
        emotion_mapping = {
            # Common mappings
            'happy': 'happy', 'happiness': 'happy', 'joy': 'happy', 'positive': 'happy',
            'sad': 'sad', 'sadness': 'sad', 'sorrow': 'sad', 'grief': 'sad',
            'angry': 'angry', 'anger': 'angry', 'rage': 'angry', 'fury': 'angry',
            'fear': 'fear', 'afraid': 'fear', 'scared': 'fear', 'terror': 'fear',
            'disgust': 'disgust', 'disgusted': 'disgust', 'revulsion': 'disgust',
            'surprise': 'surprise', 'surprised': 'surprise', 'shock': 'surprise',
            'neutral': 'neutral', 'calm': 'calm', 'peaceful': 'calm',
            'excited': 'excited', 'enthusiasm': 'excited', 'energetic': 'excited',

            # Model-specific mappings
            'LABEL_0': 'neutral', 'LABEL_1': 'happy', 'LABEL_2': 'sad', 'LABEL_3': 'angry',
            'LABEL_4': 'fear', 'LABEL_5': 'disgust', 'LABEL_6': 'surprise',

            # Additional mappings
            'contempt': 'disgust', 'boredom': 'neutral', 'stress': 'fear',
            'relaxed': 'calm', 'confident': 'happy', 'frustrated': 'angry'
        }

        standardized = {}

        for result in raw_results:
            label = result['label'].lower()
            score = result['score']

            # Map to standard emotion
            standard_emotion = emotion_mapping.get(label, label)

            # Accumulate scores for the same emotion
            if standard_emotion in standardized:
                standardized[standard_emotion] += score
            else:
                standardized[standard_emotion] = score

        # Get top emotion
        if standardized:
            top_emotion = max(standardized, key=standardized.get)
            confidence = standardized[top_emotion]

            return {
                'emotion': top_emotion,
                'confidence': confidence,
                'all_scores': standardized
            }
        else:
            return {
                'emotion': 'neutral',
                'confidence': 0.5,
                'all_scores': {'neutral': 0.5}
            }

    def fallback_emotion_detection(self, audio_clip: np.ndarray, sr: int = 16000) -> Dict[str, float]:
        """Fallback emotion detection using acoustic features"""
        try:
            # Extract basic acoustic features
            energy = np.mean(audio_clip ** 2)
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio_clip))

            # Simple heuristic classification
            if energy > 0.01 and zcr > 0.1:
                return {'emotion': 'excited', 'confidence': 0.6, 'all_scores': {'excited': 0.6}}
            elif energy < 0.005:
                return {'emotion': 'sad', 'confidence': 0.6, 'all_scores': {'sad': 0.6}}
            else:
                return {'emotion': 'neutral', 'confidence': 0.5, 'all_scores': {'neutral': 0.5}}

        except Exception as e:
            print(f"❌ Fallback emotion detection failed: {e}")
            return {'emotion': 'neutral', 'confidence': 0.0, 'all_scores': {'neutral': 0.0}}

    def process_segments(self, segments_data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Process all segments and add emotion information"""
        print("🎭 Analyzing emotions using Hugging Face models...")

        enhanced_segments = {}
        total_segments = sum(len(segments) for segments in segments_data.values())
        processed = 0

        for speaker_id, segments in segments_data.items():
            enhanced_segments[speaker_id] = []

            for segment in segments:
                start_time = segment['start_time']
                end_time = segment['end_time']

                # Extract audio clip
                audio_clip, sr = self.extract_audio_clip(start_time, end_time)

                # Predict emotion
                emotion_result = self.predict_emotion(audio_clip, sr)

                # Add emotion info to segment
                enhanced_segment = segment.copy()
                enhanced_segment['emotion'] = emotion_result['emotion']
                enhanced_segment['emotion_confidence'] = emotion_result['confidence']
                enhanced_segment['emotion_scores'] = emotion_result['all_scores']

                enhanced_segments[speaker_id].append(enhanced_segment)

                processed += 1
                print(f"   ✅ {processed}/{total_segments}: {speaker_id} - {emotion_result['emotion']} ({emotion_result['confidence']:.3f}) - {segment['transcript'][:40]}...")

                # Clear GPU memory periodically
                if processed % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return enhanced_segments

    def create_emotion_summary(self, enhanced_segments: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """Create emotion summary for each speaker"""
        summary = {}

        for speaker_id, segments in enhanced_segments.items():
            emotions = [seg['emotion'] for seg in segments]
            confidences = [seg['emotion_confidence'] for seg in segments]

            # Count emotions
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            # Calculate statistics
            total_segments = len(segments)
            avg_confidence = np.mean(confidences) if confidences else 0
            dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'neutral'

            summary[speaker_id] = {
                'total_segments': total_segments,
                'emotion_distribution': emotion_counts,
                'dominant_emotion': dominant_emotion,
                'average_confidence': avg_confidence,
                'emotions': emotions
            }

        return summary

    def visualize_emotions(self, enhanced_segments: Dict[str, List[Dict]]):
        """Create enhanced visualizations for emotions"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Color palette for emotions
        colors = {
            'happy': '#FFD700', 'sad': '#4169E1', 'angry': '#DC143C',
            'calm': '#32CD32', 'excited': '#FF8C00', 'fear': '#9932CC',
            'disgust': '#8B4513', 'neutral': '#808080', 'surprise': '#FF1493'
        }

        # 1. Emotion timeline with confidence
        ax1 = axes[0, 0]
        y_pos = 0
        speaker_labels = []

        for speaker_id, segments in enhanced_segments.items():
            speaker_labels.append(speaker_id)
            for segment in segments:
                emotion = segment['emotion']
                confidence = segment['emotion_confidence']
                color = colors.get(emotion, '#808080')

                # Alpha based on confidence
                alpha = 0.4 + (confidence * 0.6)

                ax1.barh(y_pos, segment['duration'], left=segment['start_time'],
                        height=0.8, color=color, alpha=alpha,
                        edgecolor='black', linewidth=0.5)
            y_pos += 1

        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Speakers')
        ax1.set_title('Emotion Timeline (Intensity = Confidence)')
        ax1.set_yticks(range(len(enhanced_segments)))
        ax1.set_yticklabels(speaker_labels)
        ax1.grid(True, alpha=0.3)

        # 2. Emotion distribution pie chart
        ax2 = axes[0, 1]
        all_emotions = []
        for segments in enhanced_segments.values():
            all_emotions.extend([seg['emotion'] for seg in segments])

        emotion_counts = {}
        for emotion in all_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        pie_colors = [colors.get(emotion, '#808080') for emotion in emotion_counts.keys()]
        wedges, texts, autotexts = ax2.pie(emotion_counts.values(),
                                          labels=emotion_counts.keys(),
                                          autopct='%1.1f%%',
                                          colors=pie_colors,
                                          startangle=90)
        ax2.set_title('Overall Emotion Distribution')

        # 3. Average confidence by emotion
        ax3 = axes[1, 0]
        emotion_confidences = {}
        for segments in enhanced_segments.values():
            for segment in segments:
                emotion = segment['emotion']
                confidence = segment['emotion_confidence']

                if emotion not in emotion_confidences:
                    emotion_confidences[emotion] = []
                emotion_confidences[emotion].append(confidence)

        emotions_list = list(emotion_confidences.keys())
        avg_confidences = [np.mean(emotion_confidences[emotion]) for emotion in emotions_list]
        bar_colors = [colors.get(emotion, '#808080') for emotion in emotions_list]

        bars = ax3.bar(emotions_list, avg_confidences, color=bar_colors, alpha=0.7)
        ax3.set_title('Average Confidence by Emotion')
        ax3.set_ylabel('Confidence Score')
        ax3.set_xticklabels(emotions_list, rotation=45)

        # Add value labels on bars
        for bar, conf in zip(bars, avg_confidences):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{conf:.3f}', ha='center', va='bottom', fontsize=9)

        # 4. Emotion heatmap by speaker
        ax4 = axes[1, 1]
        speakers = list(enhanced_segments.keys())
        all_unique_emotions = list(set(all_emotions))

        # Create matrix for heatmap
        heatmap_data = []
        for speaker in speakers:
            speaker_emotions = [seg['emotion'] for seg in enhanced_segments[speaker]]
            speaker_row = []
            for emotion in all_unique_emotions:
                count = speaker_emotions.count(emotion)
                speaker_row.append(count)
            heatmap_data.append(speaker_row)

        sns.heatmap(heatmap_data,
                   xticklabels=all_unique_emotions,
                   yticklabels=speakers,
                   annot=True,
                   fmt='d',
                   cmap='YlOrRd',
                   ax=ax4)
        ax4.set_title('Emotion Frequency by Speaker')

        plt.tight_layout()
        plt.show()

        # Create legend for emotions
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors.get(emotion, '#808080'),
                                       alpha=0.7, label=emotion.capitalize())
                         for emotion in colors.keys() if emotion in all_emotions]

        if legend_elements:
            plt.figure(figsize=(10, 2))
            plt.legend(handles=legend_elements, loc='center', ncol=len(legend_elements))
            plt.axis('off')
            plt.title('Emotion Legend')
            plt.tight_layout()
            plt.show()

    def save_enhanced_results(self, enhanced_segments: Dict[str, List[Dict]], base_filename: str = "hf_emotion_enhanced_results"):
        """Save enhanced results with HF emotion information"""

        # Save JSON with full emotion data
        json_file = f"{base_filename}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_segments, f, indent=2, ensure_ascii=False)

        # Save detailed text report
        txt_file = f"{base_filename}_detailed.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("🎭 HUGGING FACE EMOTION-ENHANCED TRANSCRIPTION RESULTS\n")
            f.write("=" * 70 + "\n")
            f.write(f"Model used: {self.model_name}\n")
            f.write("=" * 70 + "\n\n")

            for speaker_id, segments in enhanced_segments.items():
                f.write(f"🎤 {speaker_id.upper()}:\n")
                f.write("─" * 60 + "\n")

                for i, segment in enumerate(segments, 1):
                    emotion = segment['emotion']
                    confidence = segment['emotion_confidence']
                    timestamp = f"[{segment['start_time']:>6.1f}s - {segment['end_time']:>6.1f}s]"

                    f.write(f"{i:2d}. {timestamp} 🎭 {emotion.upper()} ({confidence:.3f})\n")
                    f.write(f"    \"{segment['transcript']}\"\n")

                    # Show all emotion scores
                    if 'emotion_scores' in segment:
                        f.write(f"    Emotion scores: {segment['emotion_scores']}\n")
                    f.write("\n")

                f.write("\n")

        # Create emotion summary
        summary = self.create_emotion_summary(enhanced_segments)
        summary_file = f"{base_filename}_emotion_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("📊 HUGGING FACE EMOTION ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Model: {self.model_name}\n")
            f.write("=" * 50 + "\n\n")

            for speaker_id, stats in summary.items():
                f.write(f"🎤 {speaker_id.upper()}:\n")
                f.write(f"   Total segments: {stats['total_segments']}\n")
                f.write(f"   Dominant emotion: {stats['dominant_emotion']}\n")
                f.write(f"   Average confidence: {stats['average_confidence']:.3f}\n")
                f.write(f"   Emotion distribution:\n")

                for emotion, count in stats['emotion_distribution'].items():
                    percentage = (count / stats['total_segments']) * 100
                    f.write(f"     - {emotion}: {count} ({percentage:.1f}%)\n")
                f.write("\n")

        print(f"💾 Enhanced results saved:")
        print(f"   📄 JSON: {json_file}")
        print(f"   📄 Detailed report: {txt_file}")
        print(f"   📄 Summary: {summary_file}")

        return enhanced_segments

'''
"wav2vec2-emotion": ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
"superb-emotion": superb/wav2vec2-base-superb-er
"facebook-emotion": facebook/wav2vec2-large-xlsr-53-speech-emotion-recognition
"harshit345-emotion": harshit345/xlsr-wav2vec-speech-emotion-recognition
"m3hrdadfi-emotion": m3hrdadfi/wav2vec2-xlsr-greek-speech-emotion-recognition
'''

def main(audio_file = "/content/audio_abusive.mp3"):
    
    global AUDIO_FILE, WHISPER_MODEL, HF_TOKEN

    AUDIO_FILE = audio_file

    diarizer = ColabGPUSpeechDiarizer(
        whisper_model_size=WHISPER_MODEL,
        hf_token=HF_TOKEN
    )

    results = diarizer.diarize_and_transcribe_gpu(AUDIO_FILE)

    if results and any(segments for segments in results.values()):
        formatted_output = diarizer.format_output(results)
        diarizer.save_results(results)

    else:
        print("❌ No valid transcriptions found. This could be due to:")
        print("   1. Audio file is too short or quiet")
        print("   2. Audio format is not supported")
        print("   3. No speech detected in the audio")
        print("   4. GPU memory issues")
    
    emotion_analyzer = HuggingFaceEmotionAnalyzer(
        AUDIO_FILE,
        model_name="superb/wav2vec2-base-superb-er"
    )

    with open('diarization_results.json', 'r', encoding='utf-8') as f:
        existing_results = json.load(f)

    enhanced_results = emotion_analyzer.process_segments(existing_results)

    emotion_analyzer.save_enhanced_results(enhanced_results)

    emotion_analyzer.visualize_emotions(enhanced_results)

    summary = emotion_analyzer.create_emotion_summary(enhanced_results)

    return enhanced_results, summary