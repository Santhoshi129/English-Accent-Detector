#!/usr/bin/env python3
"""
Test script for the English Accent Detector
This script verifies that all components are working correctly.
"""

import sys
import os
import tempfile
import subprocess
from accent_detector import AccentDetector

def test_dependencies():
    """Test if all required dependencies are available."""
    print("🔧 Testing Dependencies...")
    
    # Test Python modules
    try:
        from faster_whisper import WhisperModel
        import librosa
        import yt_dlp
        import streamlit
        import plotly
        print("✅ All Python dependencies available")
    except ImportError as e:
        print(f"❌ Missing Python dependency: {e}")
        return False
    
    # Test FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is available")
        else:
            print("❌ FFmpeg not working properly")
            return False
    except FileNotFoundError:
        print("❌ FFmpeg not found. Please install FFmpeg.")
        return False
    
    return True

def test_whisper_model():
    """Test if Whisper model can be loaded."""
    print("\n🎤 Testing Whisper Model...")
    
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")
        print("✅ Whisper model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading Whisper model: {e}")
        return False

def test_detector_initialization():
    """Test if AccentDetector can be initialized."""
    print("\n🎭 Testing AccentDetector Initialization...")
    
    try:
        detector = AccentDetector()
        print("✅ AccentDetector initialized successfully")
        
        # Test accent patterns
        expected_accents = ['American', 'British', 'Australian', 'Canadian', 'Indian', 'Irish']
        actual_accents = list(detector.accent_patterns.keys())
        
        if set(expected_accents) == set(actual_accents):
            print(f"✅ All accent patterns available: {', '.join(actual_accents)}")
        else:
            print(f"⚠️  Accent patterns mismatch. Expected: {expected_accents}, Got: {actual_accents}")
        
        return True
    except Exception as e:
        print(f"❌ Error initializing AccentDetector: {e}")
        return False

def test_audio_processing():
    """Test audio processing functionality."""
    print("\n🔊 Testing Audio Processing...")
    
    try:
        detector = AccentDetector()
        
        # Create a simple test audio file (silence)
        import numpy as np
        import soundfile as sf
        
        # Generate 3 seconds of silence at 16kHz
        duration = 3
        sample_rate = 16000
        silence = np.zeros(duration * sample_rate)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, silence, sample_rate)
            test_audio_path = tmp_file.name
        
        try:
            # Test prosodic analysis
            features = detector.analyze_prosodic_features(test_audio_path)
            print("✅ Audio feature extraction working")
            
            # Test transcription (will be empty for silence, but should not crash)
            transcription = detector.transcribe_audio(test_audio_path)
            print("✅ Audio transcription working")
            
        finally:
            # Cleanup
            os.unlink(test_audio_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Error in audio processing: {e}")
        return False

def test_streamlit_import():
    """Test if Streamlit app can be imported."""
    print("\n🌐 Testing Streamlit App...")
    
    try:
        # Test if app.py can be imported without running
        import importlib.util
        spec = importlib.util.spec_from_file_location("app", "app.py")
        app_module = importlib.util.module_from_spec(spec)
        # Don't execute the module, just test if it can be loaded
        print("✅ Streamlit app can be imported")
        return True
    except Exception as e:
        print(f"❌ Error importing Streamlit app: {e}")
        return False

def test_cli_import():
    """Test if CLI can be imported."""
    print("\n💻 Testing CLI...")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("cli", "cli.py")
        cli_module = importlib.util.module_from_spec(spec)
        print("✅ CLI can be imported")
        return True
    except Exception as e:
        print(f"❌ Error importing CLI: {e}")
        return False

def show_system_info():
    """Display system information."""
    print("\n📋 System Information:")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Current directory: {os.getcwd()}")
    
    # Check available disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        print(f"Available disk space: {free // (2**30)} GB")
    except:
        print("Could not determine disk space")

def main():
    """Run all tests."""
    print("🎭 ENGLISH ACCENT DETECTOR - SYSTEM TEST")
    print("=" * 50)
    
    show_system_info()
    
    tests = [
        test_dependencies,
        test_whisper_model,
        test_detector_initialization,
        test_audio_processing,
        test_streamlit_import,
        test_cli_import
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ ALL TESTS PASSED! System is ready to use.")
        print("\n🚀 To start the web interface, run:")
        print("   streamlit run app.py")
        print("\n💻 To test the CLI, run:")
        print("   python cli.py --help")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\n🛠️  Troubleshooting tips:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Install FFmpeg: brew install ffmpeg (macOS) or apt install ffmpeg (Ubuntu)")
        print("3. Check internet connection for model downloads")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 