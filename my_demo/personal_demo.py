#!/usr/bin/env python3
"""
Personal Look Once to Hear Demo
Test the system with your own audio files on your laptop
"""

import os
import sys
# Add parent directory to path so we can import src modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav
import src.utils as utils
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import warnings
warnings.filterwarnings('ignore')

class PersonalDemo:
    def __init__(self):
        """Initialize the Look Once to Hear system"""
        print("🚀 Initializing Look Once to Hear Demo...")
        
        # Load the main TSH model
        print("📦 Loading target speech hearing model...")
        config = utils.Params(os.path.join("..", "configs", "tsh_cipic_only.json"))
        self.model = utils.import_attr(config.pl_module)(**config.pl_module_args)
        checkpoint = torch.load(os.path.join("..", "runs", "tsh", "best.ckpt"), map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        
        # Load speaker encoder (works with any speaker)
        print("🎤 Loading universal speaker encoder...")
        self.speaker_encoder = VoiceEncoder()
        
        print("✅ System ready!")
        
    def enroll_speaker(self, enrollment_audio_path):
        """
        Step 1: "Look Once" - Enroll your target speaker
        
        Args:
            enrollment_audio_path: Path to clean audio of the person you want to hear
                                 (3-5 seconds of clear speech, preferably without background noise)
        """
        print(f"\n👀 ENROLLING SPEAKER from: {os.path.basename(enrollment_audio_path)}")
        
        if not os.path.exists(enrollment_audio_path):
            raise FileNotFoundError(f"Audio file not found: {enrollment_audio_path}")
        
        # Load and preprocess enrollment audio
        wav = preprocess_wav(enrollment_audio_path)
        
        # Extract speaker embedding (voice fingerprint)
        print("🧬 Extracting voice fingerprint...")
        speaker_embedding = self.speaker_encoder.embed_utterance(wav)
        
        # Prepare for model
        self.current_speaker_embedding = torch.from_numpy(speaker_embedding).unsqueeze(0).unsqueeze(0)
        
        # Display enrollment info
        duration = len(wav) / 16000  # Assuming 16kHz
        print(f"✅ Speaker enrolled successfully!")
        print(f"   • Duration: {duration:.2f} seconds")
        print(f"   • Embedding size: {speaker_embedding.shape}")
        
        return Audio(wav, rate=16000)
    
    def separate_target_speaker(self, mixture_audio_path, output_path=None):
        """
        Step 2: "To Hear" - Extract the enrolled speaker from noisy mixture
        
        Args:
            mixture_audio_path: Path to noisy audio containing multiple speakers
            output_path: Where to save the enhanced audio (optional)
        """
        if not hasattr(self, 'current_speaker_embedding'):
            raise ValueError("❌ No speaker enrolled! Call enroll_speaker() first.")
        
        print(f"\n🎧 PROCESSING MIXTURE: {os.path.basename(mixture_audio_path)}")
        
        if not os.path.exists(mixture_audio_path):
            raise FileNotFoundError(f"Audio file not found: {mixture_audio_path}")
        
        # Load mixture audio
        mixture, sr = sf.read(mixture_audio_path)
        print(f"📊 Input audio: {mixture.shape}, {sr}Hz")
        
        # Convert to binaural if mono
        if mixture.ndim == 1:
            mixture = np.stack([mixture, mixture])  # Duplicate mono to stereo
            print("🔄 Converted mono to binaural")
        else:
            mixture = mixture.T  # [channels, samples]
        
        # Prepare for model
        mixture_tensor = torch.from_numpy(mixture).float().unsqueeze(0)  # [1, 2, samples]
        
        print("🔮 Running AI separation...")
        with torch.no_grad():
            enhanced = self.model(mixture_tensor, self.current_speaker_embedding)
        
        # Convert back to audio
        enhanced_audio = enhanced.squeeze(0).numpy()  # [2, samples]
        
        # Save if output path provided
        if output_path:
            enhanced_output = enhanced_audio.T  # [samples, channels]
            # Ensure audio is in proper range and format
            enhanced_output = np.clip(enhanced_output, -1.0, 1.0)
            sf.write(output_path, enhanced_output, sr, subtype='PCM_16')
            print(f"💾 Enhanced audio saved to: {output_path}")
        
        print("✅ Target speaker separation complete!")
        
        # Return both for comparison
        return {
            'original': mixture.T,  # [samples, channels]
            'enhanced': enhanced_audio.T,  # [samples, channels]
            'sample_rate': sr
        }
    
    def analyze_results(self, results):
        """
        Analyze and display the separation results
        """
        original = results['original']
        enhanced = results['enhanced']
        sr = results['sample_rate']
        
        print(f"\n📈 ANALYSIS:")
        
        # Calculate energy levels
        orig_energy = np.mean(original**2)
        enh_energy = np.mean(enhanced**2)
        
        print(f"   • Original energy: {orig_energy:.6f}")
        print(f"   • Enhanced energy: {enh_energy:.6f}")
        print(f"   • Energy ratio: {enh_energy/orig_energy:.2f}")
        
        # Calculate basic SNR estimate
        noise_estimate = original - enhanced
        signal_power = np.mean(enhanced**2)
        noise_power = np.mean(noise_estimate**2)
        snr_improvement = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        print(f"   • Estimated SNR improvement: {snr_improvement:.2f} dB")
        
        try:
            # Ensure proper audio format for IPython display
            # Normalize audio to [-1, 1] range and ensure correct sample rate
            original_norm = original / np.max(np.abs(original)) if np.max(np.abs(original)) > 0 else original
            enhanced_norm = enhanced / np.max(np.abs(enhanced)) if np.max(np.abs(enhanced)) > 0 else enhanced
            
            # Ensure sample rate is int and reasonable
            sample_rate = int(sr) if sr <= 48000 else 16000
            
            return {
                'original_audio': Audio(original_norm, rate=sample_rate),
                'enhanced_audio': Audio(enhanced_norm, rate=sample_rate)
            }
        except Exception as e:
            print(f"⚠️ Could not create audio widgets: {e}")
            return {
                'snr_improvement': snr_improvement,
                'original_energy': orig_energy,
                'enhanced_energy': enh_energy
            }

def create_demo_instructions():
    """Create instructions for setting up your demo"""
    instructions = """
🎯 HOW TO SET UP YOUR PERSONAL DEMO:

1. PREPARE YOUR AUDIO FILES:
   
   a) ENROLLMENT AUDIO (target speaker):
      • Record 3-5 seconds of clear speech from the person you want to hear
      • Save as: "demo_recordings/my_target_speaker.wav"
      • Should be clean, minimal background noise
      • Must be .wav format
   
   b) MIXTURE AUDIO (noisy scene):
      • Record or find audio with multiple people talking + background noise
      • Should contain your target speaker mixed with others
      • Save as: "demo_recordings/noisy_mixture.wav"
      • Can be longer (10+ seconds), must be .wav format

2. RUN THE DEMO:
   
   ```python
   # Initialize demo
   demo = PersonalDemo()
   
   # Step 1: Enroll your target speaker
   demo.enroll_speaker("demo_recordings/my_target_speaker.wav")
   
   # Step 2: Separate them from noisy mixture  
   results = demo.separate_target_speaker("demo_recordings/noisy_mixture.wav", "enhanced_output.wav")
   
   # Step 3: Analyze results
   audio_comparison = demo.analyze_results(results)
   ```

3. LISTEN TO RESULTS:
   • Original mixture vs Enhanced target speaker
   • The system should isolate your target speaker!

💡 TIPS FOR BEST RESULTS:
   • Use clear enrollment audio (good microphone, quiet room)
   • Target speaker should be clearly audible in the mixture
   • Binaural/stereo audio works better than mono
   • Avoid extreme background noise in enrollment
"""
    return instructions

def quick_demo():
    """
    Quick demo using existing test data to show how it works
    """
    print("🚀 QUICK DEMO - Using Test Data")
    print("="*50)
    
    try:
        # Initialize system
        demo = PersonalDemo()
        
        # Check if we have sample data
        test_audio_dir = os.path.join("..", "data", "MixLibriSpeech", "librispeech_scaper_fmt", "test-clean")
        
        if os.path.exists(test_audio_dir):
            # Find a sample speaker
            speakers = os.listdir(test_audio_dir)
            if speakers:
                sample_speaker = speakers[0]
                speaker_dir = os.path.join(test_audio_dir, sample_speaker)
                audio_files = [f for f in os.listdir(speaker_dir) if f.endswith('.flac')]
                
                if len(audio_files) >= 2:
                    # Use first file for enrollment, second for mixture simulation
                    enrollment_file = os.path.join(speaker_dir, audio_files[0])
                    
                    print(f"\n📋 Using sample data from speaker {sample_speaker}")
                    print(f"🎤 Enrollment: {audio_files[0]}")
                    
                    # Enroll speaker
                    enrollment_audio = demo.enroll_speaker(enrollment_file)
                    print("🎵 Enrollment audio:")
                    display(enrollment_audio)
                    
                    # For demo, we'll use the same speaker but could simulate mixture
                    mixture_file = os.path.join(speaker_dir, audio_files[1])
                    print(f"🔀 Processing: {audio_files[1]}")
                    
                    # Process
                    results = demo.separate_target_speaker(mixture_file, "demo_output.wav")
                    
                    # Analyze
                    audio_comparison = demo.analyze_results(results)
                    
                    print("\n🎵 COMPARISON:")
                    print("Original:")
                    display(audio_comparison['original_audio'])
                    print("Enhanced:")
                    display(audio_comparison['enhanced_audio'])
                    
                    print("\n🎉 Demo complete! Check 'demo_output.wav' for the enhanced audio.")
                    
                else:
                    print("❌ Not enough audio files found for demo")
            else:
                print("❌ No speakers found in test data")
        else:
            print("❌ Test data not found. Please set up your own audio files.")
            print(create_demo_instructions())
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\nTo run with your own files:")
        print(create_demo_instructions())

if __name__ == "__main__":
    print("🎤 LOOK ONCE TO HEAR - Personal Demo")
    print("="*40)
    
    # Show instructions
    print(create_demo_instructions())
    
    # Ask user what they want to do
    print("\nChoose an option:")
    print("1. Run quick demo with test data")
    print("2. Set up with my own audio files")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        quick_demo()
    else:
        print("\n📝 SETUP INSTRUCTIONS:")
        print(create_demo_instructions())
        print("\nAfter preparing your files in demo_recordings/, run:")
        print("```python")
        print("from personal_demo import PersonalDemo")
        print("demo = PersonalDemo()")
        print("demo.enroll_speaker('demo_recordings/your_target_speaker.wav')")
        print("results = demo.separate_target_speaker('demo_recordings/your_mixture.wav', 'output.wav')")
        print("demo.analyze_results(results)")
        print("```")