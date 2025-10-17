#!/usr/bin/env python3
"""
Simple Interactive Demo for Look Once to Hear
Easy way to test with your own audio files
"""

import os
import sys
# Add parent directory to path so we can import src modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def main():
    """Main interactive demo"""
    print("🎤 LOOK ONCE TO HEAR - Interactive Demo")
    print("="*45)
    print("This demo lets you test the AI system with your own audio files.")
    print()
    
    try:
        from personal_demo import PersonalDemo
        
        # Initialize the system
        print("⏳ Loading AI models...")
        demo = PersonalDemo()
        print()
        
        # Check for audio files in demo_recordings folder
        recordings_dir = "demo_recordings"
        print(f"📁 Looking for .wav files in {recordings_dir} folder...")
        
        if not os.path.exists(recordings_dir):
            os.makedirs(recordings_dir)
            print(f"📂 Created {recordings_dir} folder")
        
        audio_files = []
        for file in os.listdir(recordings_dir):
            if file.lower().endswith('.wav'):
                audio_files.append(os.path.join(recordings_dir, file))
        
        if audio_files:
            print(f"Found {len(audio_files)} .wav file(s):")
            for i, file in enumerate(audio_files, 1):
                print(f"  {i}. {os.path.basename(file)}")
            print()
        else:
            print(f"No .wav files found in {recordings_dir} folder.")
            print(f"Please add .wav files to the {recordings_dir} folder and run the demo again.")
            print()
            print("💡 You need:")
            print("   1. Enrollment audio: Clear 3-5 second recording of target speaker")
            print("   2. Mixture audio: Recording with multiple speakers + background noise")
            print(f"   3. Save them as .wav files in the {recordings_dir} folder")
            return
        
        # Get enrollment file
        print("STEP 1: Choose ENROLLMENT audio (target speaker)")
        print("This should be 3-5 seconds of clear speech from the person you want to hear.")
        
        if len(audio_files) == 1:
            enrollment_file = audio_files[0]
            print(f"Using only available file: {enrollment_file}")
        else:
            while True:
                try:
                    choice = input(f"Enter number (1-{len(audio_files)}): ").strip()
                    idx = int(choice) - 1
                    if 0 <= idx < len(audio_files):
                        enrollment_file = audio_files[idx]
                        break
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a number.")
        
        print(f"✅ Enrollment file: {os.path.basename(enrollment_file)}")
        
        # Enroll the speaker
        print("\n⏳ Enrolling target speaker...")
        try:
            demo.enroll_speaker(enrollment_file)
        except Exception as e:
            print(f"❌ Enrollment failed: {e}")
            return
        
        # Get mixture file
        print("\nSTEP 2: Choose MIXTURE audio (multiple speakers)")
        print("This should contain your target speaker mixed with other voices/noise.")
        
        remaining_files = [f for f in audio_files if f != enrollment_file]
        
        if not remaining_files:
            print("⚠️  Only one audio file available. Using same file for mixture.")
            print("(In real use, you'd have a separate noisy mixture file)")
            mixture_file = enrollment_file
        else:
            print("Available files:")
            for i, file in enumerate(remaining_files, 1):
                print(f"  {i}. {os.path.basename(file)}")
            
            while True:
                try:
                    choice = input(f"Enter number (1-{len(remaining_files)}): ").strip()
                    idx = int(choice) - 1
                    if 0 <= idx < len(remaining_files):
                        mixture_file = remaining_files[idx]
                        break
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a number.")
        
        print(f"✅ Mixture file: {os.path.basename(mixture_file)}")
        
        # Process the mixture
        print("\n⏳ Processing mixture to extract target speaker...")
        output_file = f"enhanced_{os.path.splitext(os.path.basename(mixture_file))[0]}.wav"
        
        try:
            results = demo.separate_target_speaker(mixture_file, output_file)
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            return
        
        # Analyze results
        print("\n📊 Analyzing results...")
        demo.analyze_results(results)
        
        print(f"\n🎉 SUCCESS!")
        print(f"Enhanced audio saved as: {output_file}")
        print("\n💡 Next steps:")
        print("   • Listen to the original and enhanced audio files")
        print("   • Compare the quality - target speaker should be clearer")
        print("   • Try with different audio files for better results")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the correct directory.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()