#!/usr/bin/env python3
"""
Simple launcher for English Accent Detector
Automatically handles setup and starts the application.
"""

import subprocess
import sys
import os

def check_dependencies():
    """Quick dependency check."""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_basic_deps():
    """Install basic dependencies if missing."""
    print("🔄 Installing basic dependencies...")
    basic_deps = [
        "streamlit", "yt-dlp", "librosa", "numpy", "pandas", 
        "scikit-learn", "requests", "plotly", "soundfile"
    ]
    
    for dep in basic_deps:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                         check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print(f"⚠️  Could not install {dep}, continuing...")

def main():
    print("🎭 ENGLISH ACCENT DETECTOR")
    print("=" * 40)
    
    # Check if we have basic dependencies
    if not check_dependencies():
        print("📦 Missing dependencies, installing...")
        install_basic_deps()
    
    # Try to install whisper (optional)
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "faster-whisper"], 
                     check=True, capture_output=True)
        print("✅ faster-whisper installed")
    except subprocess.CalledProcessError:
        print("⚠️  faster-whisper not available, using fallback")
    
    print("\n🚀 Starting the web application...")
    print("📍 The app will open at: http://localhost:8501")
    print("💡 Use Demo Mode if you have network restrictions!")
    print("\n" + "=" * 40)
    
    # Start Streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting application: {e}")
        print("\n🛠️  Troubleshooting:")
        print("1. Run: pip install streamlit")
        print("2. Run: streamlit run app.py")
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")

if __name__ == "__main__":
    main() 