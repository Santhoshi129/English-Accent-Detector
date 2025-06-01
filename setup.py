#!/usr/bin/env python3
"""
Setup script for English Accent Detector
Handles various installation scenarios and network restrictions.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors gracefully."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} successful")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def install_dependencies():
    """Install dependencies with fallback options."""
    print("🎭 ENGLISH ACCENT DETECTOR - SETUP")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    # Install basic requirements first
    basic_requirements = [
        "streamlit>=1.28.0",
        "yt-dlp>=2023.12.30", 
        "librosa>=0.10.1",
        "numpy>=1.24.0,<2.0.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "textblob>=0.17.1",
        "python-speech-features>=0.6",
        "pronouncing>=0.2.0",
        "requests>=2.31.0",
        "plotly>=5.17.0",
        "soundfile>=0.12.1"
    ]
    
    print("\n📦 Installing basic dependencies...")
    for req in basic_requirements:
        if not run_command(f"pip install '{req}'", f"Installing {req.split('>=')[0]}"):
            print(f"⚠️  Failed to install {req}, continuing...")
    
    # Try to install Whisper models
    print("\n🎤 Installing Whisper models...")
    
    # Try faster-whisper first
    faster_whisper_success = run_command(
        "pip install 'faster-whisper>=0.10.0'", 
        "Installing faster-whisper"
    )
    
    # If faster-whisper fails, try openai-whisper
    if not faster_whisper_success:
        print("\n⚠️  faster-whisper failed, trying openai-whisper as fallback...")
        openai_whisper_success = run_command(
            "pip install openai-whisper", 
            "Installing openai-whisper"
        )
        
        if not openai_whisper_success:
            print("\n⚠️  Both Whisper implementations failed to install.")
            print("The system will use mock transcription for testing.")
    
    # Try to install torch if not already installed
    print("\n🔥 Installing PyTorch...")
    run_command("pip install 'torch>=2.0.0' 'torchaudio>=2.0.0'", "Installing PyTorch")
    
    print("\n✅ Setup completed! Some components may use fallback implementations.")
    return True

def test_installation():
    """Test the installation."""
    print("\n🧪 Testing installation...")
    
    # Test basic imports
    try:
        import streamlit
        import yt_dlp
        import librosa
        import numpy
        import pandas
        import sklearn
        print("✅ Core dependencies working")
    except ImportError as e:
        print(f"❌ Core dependency missing: {e}")
        return False
    
    # Test Whisper models
    whisper_available = False
    try:
        from faster_whisper import WhisperModel
        print("✅ faster-whisper available")
        whisper_available = True
    except ImportError:
        print("⚠️  faster-whisper not available")
    
    try:
        import whisper
        print("✅ openai-whisper available")
        whisper_available = True
    except ImportError:
        print("⚠️  openai-whisper not available")
    
    if not whisper_available:
        print("⚠️  No Whisper models available - will use mock transcription")
    
    # Test FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
        if result.returncode == 0:
            print("✅ FFmpeg available")
        else:
            print("❌ FFmpeg not working")
    except FileNotFoundError:
        print("❌ FFmpeg not found - please install FFmpeg")
        return False
    
    print("\n🎉 Installation test completed!")
    return True

def main():
    """Main setup function."""
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_installation()
    else:
        install_dependencies()
        test_installation()
        
        print("\n🚀 Next steps:")
        print("1. Start the web app: streamlit run app.py")
        print("2. Or use CLI: python cli.py --help")
        print("3. Or run system test: python test_system.py")

if __name__ == "__main__":
    main() 