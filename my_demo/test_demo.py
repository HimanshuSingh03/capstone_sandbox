#!/usr/bin/env python3
"""
Quick test to verify the personal demo works
"""

import os
import sys
# Add parent directory to path so we can import src modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_demo():
    """Test the demo with existing data"""
    print("🧪 TESTING YOUR PERSONAL DEMO")
    print("="*35)
    
    try:
        from personal_demo import PersonalDemo
        
        # Initialize
        print("⏳ Loading system...")
        demo = PersonalDemo()
        
        # Find test audio
        test_dir = os.path.join("..", "data", "MixLibriSpeech", "librispeech_scaper_fmt", "test-clean")
        if not os.path.exists(test_dir):
            print("❌ Test data not found. Please run the full setup first.")
            return False
        
        # Get a sample speaker
        speakers = os.listdir(test_dir)
        if not speakers:
            print("❌ No test speakers found.")
            return False
        
        speaker = speakers[0]
        speaker_dir = os.path.join(test_dir, speaker)
        audio_files = [f for f in os.listdir(speaker_dir) if f.endswith('.flac')]
        
        if len(audio_files) < 2:
            print("❌ Not enough audio files for test.")
            return False
        
        enrollment_file = os.path.join(speaker_dir, audio_files[0])
        test_file = os.path.join(speaker_dir, audio_files[1])
        
        print(f"🎤 Testing with speaker: {speaker}")
        print(f"📄 Enrollment: {audio_files[0]}")
        print(f"📄 Test audio: {audio_files[1]}")
        
        # Test enrollment
        print("\n⏳ Testing speaker enrollment...")
        demo.enroll_speaker(enrollment_file)
        print("✅ Enrollment successful!")
        
        # Test separation
        print("\n⏳ Testing audio separation...")
        results = demo.separate_target_speaker(test_file, "test_output.wav")
        print("✅ Separation successful!")
        
        # Test analysis
        print("\n⏳ Testing result analysis...")
        demo.analyze_results(results)
        print("✅ Analysis successful!")
        
        print("\n🎉 ALL TESTS PASSED!")
        print(f"✅ Enhanced audio saved as: test_output.wav")
        print("\n🎯 Your personal demo is ready to use!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_demo()
    
    if success:
        print("\n" + "="*50)
        print("🚀 HOW TO USE YOUR PERSONAL DEMO:")
        print("="*50)
        print()
        print("1. PREPARE YOUR AUDIO FILES:")
        print("   • Put your audio files in this directory")
        print("   • Need: enrollment audio (3-5 sec clear speech)")
        print("   • Need: mixture audio (multiple speakers + noise)")
        print()
        print("2. RUN INTERACTIVE DEMO:")
        print("   python interactive_demo.py")
        print()
        print("3. OR USE PROGRAMMATICALLY:")
        print("   ```python")
        print("   from personal_demo import PersonalDemo")
        print("   demo = PersonalDemo()")
        print("   demo.enroll_speaker('your_target.wav')")
        print("   results = demo.separate_target_speaker('mixture.wav', 'output.wav')")
        print("   demo.analyze_results(results)")
        print("   ```")
        print()
        print("🎵 The system works with ANY speaker - not just training data!")
    else:
        print("\n❌ Setup incomplete. Please complete the full installation first.")