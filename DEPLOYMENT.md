# 🚀 Deployment Guide - English Accent Detector

## Quick Start (Recommended)

### Option 1: One-Click Launch
```bash
python3 run.py
```
This automatically installs dependencies and starts the web app at `http://localhost:8501`

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start the application
streamlit run app.py
```

## 🌐 Deployment Options

### 1. Streamlit Cloud (Easiest)
1. Push code to GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy automatically

**Benefits:**
- Free hosting
- Automatic HTTPS
- Easy sharing with public URL
- No server management

### 2. Docker Deployment
```bash
# Build the image
docker build -t accent-detector .

# Run the container
docker run -p 8501:8501 accent-detector
```

### 3. Cloud Platforms

#### Heroku
```bash
# Add buildpacks for FFmpeg
heroku buildpacks:add --index 1 https://github.com/jonathanong/heroku-buildpack-ffmpeg-latest.git
heroku buildpacks:add heroku/python

# Deploy
git push heroku main
```

#### Railway
1. Connect GitHub repository to Railway
2. Add environment variables if needed
3. Deploy automatically

#### AWS/GCP/Azure
Use Docker container deployment on cloud container services.

## 🔧 Environment Configuration

### Required System Dependencies
- **Python 3.8+**
- **FFmpeg** (for audio processing)

### Optional Environment Variables
```bash
# Custom model cache directory
export WHISPER_CACHE_DIR=/path/to/cache

# Disable SSL verification (corporate environments)
export PYTHONHTTPSVERIFY=0
```

## 🏢 Corporate Environment Setup

### For Networks with SSL/Certificate Issues:

1. **Use Demo Mode**: The application includes an offline demo mode that works without internet
2. **SSL Configuration**: The app includes SSL bypass options for yt-dlp
3. **Proxy Settings**: Set proxy environment variables if needed

### Firewall Considerations:
- **Outbound HTTPS (443)**: For downloading Whisper models
- **YouTube/Video APIs**: For video processing (if using video mode)
- **Streamlit Port (8501)**: For web interface access

## 📊 Performance & Scaling

### Resource Requirements:
- **Memory**: 2-4GB during processing
- **CPU**: Any modern processor (CPU-optimized for corporate environments)
- **Storage**: ~1GB for models and temporary files

### Scaling Options:
- **Horizontal**: Deploy multiple instances behind load balancer
- **Vertical**: Increase memory for faster processing
- **Queue System**: Add task queue for high-volume processing

## 🔒 Security Considerations

### Data Privacy:
- Audio files are processed temporarily and deleted
- No permanent storage of user content
- All processing happens locally or in your deployed environment

### Access Control:
- Add authentication layer if needed (Streamlit supports various auth methods)
- Use HTTPS in production
- Consider VPN access for internal tools

## 📱 Mobile Access

The web interface is mobile-responsive and works on:
- Desktop browsers
- Tablets
- Mobile phones

## 🔄 Updates & Maintenance

### Update Application:
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

### Model Updates:
Whisper models are automatically downloaded and cached. Clear cache if needed:
```bash
rm -rf ~/.cache/whisper
```

## 🆘 Troubleshooting

### Common Issues:

1. **"FFmpeg not found"**
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu
   sudo apt install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org
   ```

2. **Memory Issues**
   - Use shorter video clips
   - Restart the application
   - Increase system memory

3. **Slow Processing**
   - Ensure good internet connection for model downloads
   - Use faster hardware
   - Consider shorter video samples

## 📞 Support

For technical issues:
1. Check the troubleshooting section above
2. Review application logs
3. Use Demo Mode as fallback
---

**Production Checklist:**
- [ ] FFmpeg installed
- [ ] Dependencies installed
- [ ] Firewall configured
- [ ] SSL certificates working
- [ ] Demo mode tested
- [ ] Resource monitoring setup
- [ ] Backup/update procedures documented 