"""
app.py - Streamlit web interface for the English Accent Detector

This module provides a user-friendly web interface built with Streamlit for the accent detection system.
It supports two main modes of operation:
1. Video URL Analysis - Process video content for accent detection
2. Demo Mode - Text-based accent analysis for quick evaluation

The interface includes:
- Interactive input methods
- Real-time progress tracking
- Visualizations of accent scores
- Detailed result explanations
- Responsive layout and styling
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from accent_detector import AccentDetector
import logging

# Configure logging for the web application
logging.basicConfig(level=logging.INFO)

# Page configuration with custom styling and layout
st.set_page_config(
    page_title="English Accent Detector",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI/UX
st.markdown("""
<style>
/* Main header styling for app title */
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}

/* Accent badge styling for result display */
.accent-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    background-color: #e6f3ff;
    border-radius: 20px;
    margin: 0.25rem;
    font-weight: bold;
}

/* Confidence level indicators */
.confidence-high {
    background-color: #d4edda;
    color: #155724;
}
.confidence-medium {
    background-color: #fff3cd;
    color: #856404;
}
.confidence-low {
    background-color: #f8d7da;
    color: #721c24;
}

/* Card components for metrics and demo mode */
.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
.demo-card {
    background-color: #e3f2fd;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 5px solid #2196f3;
}
</style>
""", unsafe_allow_html=True)

def get_confidence_class(confidence: float) -> str:
    """
    Determine the CSS class for confidence level visualization.
    
    Args:
        confidence (float): Confidence score (0-100)
        
    Returns:
        str: CSS class name for styling
    """
    if confidence >= 70:
        return "confidence-high"
    elif confidence >= 40:
        return "confidence-medium"
    else:
        return "confidence-low"

def create_accent_visualization(accent_scores: dict) -> go.Figure:
    """
    Create a horizontal bar chart visualization of accent confidence scores.
    
    Args:
        accent_scores (dict): Dictionary of accent types and their scores
        
    Returns:
        go.Figure: Plotly figure object for the visualization
    """
    df = pd.DataFrame(list(accent_scores.items()), columns=['Accent', 'Score'])
    df = df.sort_values('Score', ascending=True)
    
    fig = px.bar(
        df, 
        x='Score', 
        y='Accent',
        orientation='h',
        title="Accent Classification Scores",
        color='Score',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Confidence Score",
        yaxis_title="Accent Type",
        showlegend=False
    )
    
    return fig

def main():
    """
    Main application function that handles:
    1. UI layout and components
    2. Mode selection (Video/Demo)
    3. User input processing
    4. Result visualization
    5. Error handling
    """
    # Header
    st.markdown('<h1 class="main-header">🎭 English Accent Detector</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This tool analyzes video content to detect English accents for hiring evaluation purposes.
    You can either paste a public video URL or try the demo mode with sample text.
    """)
    
    # Mode selection
    mode = st.radio(
        "Choose analysis mode:",
        ["📹 Video URL Analysis", "📝 Demo Mode (Text Analysis)"],
        horizontal=True
    )
    
    # Sidebar information
    with st.sidebar:
        st.header("📋 How it works")
        st.markdown("""
        1. **Video Processing**: Downloads and extracts audio from your video URL
        2. **Speech Recognition**: Uses Whisper to transcribe the speech
        3. **Accent Analysis**: Analyzes linguistic patterns, vocabulary, and prosodic features
        4. **Classification**: Provides accent classification with confidence scores
        
        **Supported Accents:**
        - American English
        - British English
        - Australian English
        - Canadian English
        - Indian English
        - Irish English
        """)
        
        if mode == "📹 Video URL Analysis":
            st.header("🔗 Supported URLs")
            st.markdown("""
            - YouTube videos
            - Loom recordings
            - Direct MP4/audio links
            - Most publicly accessible video URLs
            """)
        else:
            st.header("📝 Demo Mode")
            st.markdown("""
            Demo mode analyzes text patterns to predict accent characteristics.
            Perfect for testing when you can't access videos or have network restrictions.
            """)
        
        st.header("⚠️ Important Notes")
        st.markdown("""
        - Only English language content is supported
        - Results are for evaluation purposes only
        - Demo mode uses text analysis only (no audio processing)
        """)
    
    if mode == "📹 Video URL Analysis":
        # Video URL Mode
        st.header("🎥 Video URL Input")
        
        video_url = st.text_input(
            "Enter a public video URL:",
            placeholder="https://www.youtube.com/watch?v=... or https://www.loom.com/share/...",
            help="Paste any public video URL containing English speech"
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button("🔍 Analyze Accent", type="primary", use_container_width=True)
        
        if analyze_button and video_url:
            if not video_url.strip():
                st.error("Please enter a valid video URL.")
                return
            
            # Initialize detector
            with st.spinner("Initializing accent detector..."):
                try:
                    detector = AccentDetector()
                except Exception as e:
                    st.error(f"Error initializing detector: {str(e)}")
                    return
            
            # Process video
            with st.spinner("Processing video... This may take a few minutes."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Downloading and extracting audio...")
                progress_bar.progress(25)
                
                result = detector.process_video_url(video_url)
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
            
            display_results(result)
        
        elif analyze_button:
            st.warning("Please enter a video URL to analyze.")
    
    else:
        # Demo Mode
        st.header("📝 Demo Mode - Text Analysis")
        
        # Demo explanation
        st.markdown("""
        <div class="demo-card">
            <h4>🚀 Try the Demo!</h4>
            <p>Demo mode analyzes text patterns to detect accent characteristics. 
            This works offline and is perfect for testing when you have network restrictions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample texts
        sample_texts = {
            "American Sample": "Hello, I'm going to take the elevator up to my apartment. I have a vacation scheduled next week, and I need to organize my schedule. I'll grab some aluminum foil from the store.",
            "British Sample": "Good morning, I'll take the lift up to my flat. I'm going on holiday next week, and I need to check my timetable. I'll pick up some aluminium foil whilst I'm at the shops.",
            "Australian Sample": "G'day mate, how's your arvo going? I'm having a barbie this weekend with some brekkie included. It's going to be a fair dinkum good time!",
            "Canadian Sample": "Hey there, eh? I'm talking about the process of moving house. The weather's been pretty good lately, don't you think?",
            "Indian Sample": "I need to prepone our meeting because I have to revert back to the client. Please do the needful and confirm your good name for the attendance.",
            "Irish Sample": "That's grand altogether! We had great craic at the pub last night. How are you yourself doing today? It was brilliant!"
        }
        
        # Text input options
        input_method = st.radio(
            "Choose input method:",
            ["Use sample text", "Enter custom text"],
            horizontal=True
        )
        
        if input_method == "Use sample text":
            selected_sample = st.selectbox(
                "Choose a sample text:",
                list(sample_texts.keys())
            )
            demo_text = sample_texts[selected_sample]
            st.text_area("Sample text:", demo_text, height=100, disabled=True)
        else:
            demo_text = st.text_area(
                "Enter your text to analyze:",
                placeholder="Type or paste English text here (minimum 20 words recommended for better accuracy)...",
                height=150
            )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            demo_analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
        
        if demo_analyze_button and demo_text:
            if len(demo_text.strip().split()) < 5:
                st.warning("Please enter at least 5 words for meaningful analysis.")
                return
            
            # Initialize detector
            with st.spinner("Initializing accent detector..."):
                try:
                    detector = AccentDetector()
                except Exception as e:
                    st.error(f"Error initializing detector: {str(e)}")
                    return
            
            # Process demo text
            with st.spinner("Analyzing text patterns..."):
                result = detector.process_demo_text(demo_text)
            
            display_results(result)
        
        elif demo_analyze_button:
            st.warning("Please enter some text to analyze.")

def display_results(result):
    """Display analysis results."""
    # Display results
    if result.get('success'):
        success_message = "✅ Analysis completed successfully!"
        if result.get('demo_mode'):
            success_message += " (Demo Mode - Text Analysis Only)"
        st.success(success_message)
        
        # Main results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("🎯 Classification Results")
            
            primary_accent = result['primary_accent']
            confidence = result['confidence_score']
            confidence_class = get_confidence_class(confidence)
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Primary Accent: <span class="accent-badge {confidence_class}">{primary_accent}</span></h3>
                <h4>Confidence Score: {confidence}%</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Explanation
            st.subheader("📝 Analysis Explanation")
            if result.get('demo_mode'):
                st.info("**Demo Mode**: " + result['explanation'] + " (Based on text analysis only)")
            else:
                st.info(result['explanation'])
            
            # Transcription
            st.subheader("📄 Text Content")
            with st.expander("View full text"):
                st.text_area("Analyzed Text:", result['transcription'], height=150, disabled=True)
        
        with col2:
            st.header("📊 Detailed Scores")
            
            # All accent scores
            accent_scores = result['all_accent_scores']
            for accent, score in sorted(accent_scores.items(), key=lambda x: x[1], reverse=True):
                confidence_class = get_confidence_class(score)
                st.markdown(f"""
                <div class="accent-badge {confidence_class}">
                    {accent}: {score}%
                </div>
                """, unsafe_allow_html=True)
        
        # Visualization
        st.header("📈 Visualization")
        fig = create_accent_visualization(accent_scores)
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical details (expandable)
        with st.expander("🔧 Technical Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Language Detection")
                st.write(f"**Detected Language:** {result['detected_language']}")
                
                st.subheader("All Accent Scores")
                scores_df = pd.DataFrame(list(accent_scores.items()), columns=['Accent', 'Score'])
                st.dataframe(scores_df, use_container_width=True)
            
            with col2:
                st.subheader("Analysis Features")
                if result.get('demo_mode'):
                    st.write("**Analysis Type:** Text Pattern Analysis")
                    st.write("**Audio Processing:** Not used in demo mode")
                else:
                    st.write("**Analysis Type:** Full Audio + Text Analysis")
                    audio_features = result.get('audio_features', {})
                    if audio_features:
                        for feature, value in audio_features.items():
                            if isinstance(value, float):
                                st.write(f"**{feature.replace('_', ' ').title()}:** {value:.2f}")
                            else:
                                st.write(f"**{feature.replace('_', ' ').title()}:** {value}")
        
        # Download results
        st.header("💾 Export Results")
        results_summary = {
            'Primary Accent': primary_accent,
            'Confidence Score': f"{confidence}%",
            'All Scores': accent_scores,
            'Explanation': result['explanation'],
            'Text Content': result['transcription'],
            'Analysis Type': 'Demo Mode' if result.get('demo_mode') else 'Video Analysis'
        }
        
        st.download_button(
            label="📄 Download Results as JSON",
            data=pd.Series(results_summary).to_json(indent=2),
            file_name=f"accent_analysis_{primary_accent.lower()}.json",
            mime="application/json"
        )
    
    else:
        # Error handling
        st.error("❌ Analysis failed!")
        error_msg = result.get('error', 'Unknown error occurred')
        st.error(f"Error: {error_msg}")
        
        if 'ssl' in error_msg.lower() or 'certificate' in error_msg.lower():
            st.info("""
            **Network/SSL Error Detected**: This appears to be a certificate verification issue, 
            common in corporate environments.
            
            **Solutions:**
            1. Try the **Demo Mode** (📝 tab above) which works offline
            2. Contact your IT department about SSL certificate issues
            3. Try using a personal network connection
            """)
        elif 'language' in error_msg.lower():
            st.info("""
            **Note:** This tool is specifically designed for English accent detection. 
            If your video contains English content but was misidentified, please ensure:
            - The audio quality is clear
            - There's sufficient English speech content
            - Background noise is minimal
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>English Accent Detector v1.0 | Built for hiring evaluation purposes</p>
        <p>⚠️ Results are for screening purposes only and should be used as part of a comprehensive evaluation process.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
