#!/usr/bin/env python3
"""
CLI interface for the English Accent Detector
Usage: python cli.py <video_url>
"""

import sys
import argparse
import json
from accent_detector import AccentDetector

def print_banner():
    """Print application banner."""
    print("=" * 60)
    print("🎭 ENGLISH ACCENT DETECTOR CLI")
    print("=" * 60)
    print("Analyzes video content to detect English accents")
    print("for hiring evaluation purposes.")
    print("=" * 60)

def print_results(result):
    """Print formatted results."""
    if not result.get('success'):
        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
        return
    
    print("\n✅ ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 50)
    
    # Primary results
    print(f"\n🎯 PRIMARY ACCENT: {result['primary_accent']}")
    print(f"📊 CONFIDENCE SCORE: {result['confidence_score']}%")
    
    # Confidence level
    confidence = result['confidence_score']
    if confidence >= 70:
        confidence_level = "HIGH"
        emoji = "🟢"
    elif confidence >= 40:
        confidence_level = "MEDIUM"
        emoji = "🟡"
    else:
        confidence_level = "LOW"
        emoji = "🔴"
    
    print(f"{emoji} CONFIDENCE LEVEL: {confidence_level}")
    
    # Explanation
    print(f"\n📝 EXPLANATION:")
    print(f"   {result['explanation']}")
    
    # All scores
    print(f"\n📈 ALL ACCENT SCORES:")
    accent_scores = result['all_accent_scores']
    for accent, score in sorted(accent_scores.items(), key=lambda x: x[1], reverse=True):
        bar_length = int(score / 5)  # Scale to 20 chars max
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"   {accent:12} [{bar}] {score:5.1f}%")
    
    # Language detection
    print(f"\n🌐 DETECTED LANGUAGE: {result['detected_language']}")
    
    # Transcription (truncated)
    transcription = result['transcription']
    if len(transcription) > 200:
        transcription = transcription[:200] + "..."
    print(f"\n📄 TRANSCRIPTION (excerpt):")
    print(f"   \"{transcription}\"")
    
    # Audio features
    audio_features = result.get('audio_features', {})
    if audio_features:
        print(f"\n🔧 AUDIO FEATURES:")
        for feature, value in audio_features.items():
            if isinstance(value, float):
                print(f"   {feature.replace('_', ' ').title():20}: {value:.2f}")
            else:
                print(f"   {feature.replace('_', ' ').title():20}: {value}")

def main():
    parser = argparse.ArgumentParser(
        description="English Accent Detector CLI - Analyze video content for accent classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py "https://www.youtube.com/watch?v=VIDEO_ID"
  python cli.py "https://www.loom.com/share/VIDEO_ID"
  python cli.py "https://example.com/video.mp4"

Supported video sources:
  - YouTube videos
  - Loom recordings  
  - Direct MP4/audio links
  - Most publicly accessible video URLs

Output formats:
  --json    Output results in JSON format
  --verbose Include detailed technical information
        """
    )
    
    parser.add_argument(
        'video_url',
        help='Public video URL to analyze'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results in JSON format'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Include detailed technical information'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        help='Save results to file (JSON format)'
    )
    
    if len(sys.argv) == 1:
        print_banner()
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    if not args.json:
        print_banner()
        print(f"\n🎥 Analyzing video: {args.video_url}")
        print("\nThis may take a few minutes...")
    
    try:
        # Initialize detector
        if not args.json:
            print("\n⏳ Initializing accent detector...")
        detector = AccentDetector()
        
        # Process video
        if not args.json:
            print("⏳ Processing video (downloading, transcribing, analyzing)...")
        
        result = detector.process_video_url(args.video_url)
        
        # Output results
        if args.json:
            output_data = result
            if args.verbose:
                # Include additional metadata in JSON output
                output_data['metadata'] = {
                    'tool_version': '1.0',
                    'video_url': args.video_url,
                    'supported_accents': list(detector.accent_patterns.keys())
                }
            
            json_output = json.dumps(output_data, indent=2)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(json_output)
                print(f"Results saved to {args.output}")
            else:
                print(json_output)
        else:
            print_results(result)
            
            if args.verbose and result.get('success'):
                print(f"\n🔍 DETAILED TECHNICAL INFORMATION:")
                print(f"   Tool Version: 1.0")
                print(f"   Supported Accents: {', '.join(detector.accent_patterns.keys())}")
                print(f"   Analysis Method: Linguistic + Prosodic + Phonetic")
                print(f"   Transcription Engine: OpenAI Whisper")
                print(f"   Audio Features: Pitch, Rhythm, Spectral Analysis")
            
            if args.output:
                # Save detailed results to file
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\n💾 Results saved to: {args.output}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        if args.json:
            error_result = {
                'success': False,
                'error': str(e),
                'video_url': args.video_url
            }
            print(json.dumps(error_result, indent=2))
        else:
            print(f"\n❌ Error: {str(e)}")
            print("\nTroubleshooting tips:")
            print("- Ensure the video URL is publicly accessible")
            print("- Check your internet connection")
            print("- Verify the video contains English speech")
            print("- Try a different video URL")
        sys.exit(1)

if __name__ == "__main__":
    main() 