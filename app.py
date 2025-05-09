from flask import Flask, request, render_template, url_for
import os
import threading
import time
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import google.generativeai as genai

app = Flask(__name__)

# Cache for storing previously summarized videos
summary_cache = {}
sentiment_cache = {}

def get_video_id(url):
    """Extract the video ID from a YouTube URL."""
    parsed_url = urlparse(url)
    video_id = parse_qs(parsed_url.query).get('v')
    return video_id[0] if video_id else None

def get_transcript(video_id):
    """Fetch the transcript for a given YouTube video ID."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([item['text'] for item in transcript])
    except Exception as e:
        return f"Error: {str(e)}"

def clean_transcript(text):
    """Clean the transcript by removing timestamps and extra whitespace."""
    import re
    text = re.sub(r'\[\d{2}:\d{2}:\d{2}\]', '', text)
    return ' '.join(text.split())

def chunk_text(text, max_length=8000):
    """Split text into chunks that don't exceed max_length (Gemini can handle larger chunks)."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        if current_length + len(word) > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def analyze_sentiment(text):
    """Analyze the sentiment of the transcript."""
    try:
        device = 0 if torch.cuda.is_available() else -1
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)
        
        # For longer texts, split into chunks and analyze each
        chunks = chunk_text(text, max_length=500)
        results = []
        
        for chunk in chunks:
            sentiment = sentiment_analyzer(chunk)[0]
            results.append(sentiment)
        
        # Calculate overall sentiment based on chunk results
        positive_count = sum(1 for r in results if r['label'] == 'POSITIVE')
        negative_count = len(results) - positive_count
        
        # Get dominant sentiment and confidence
        if positive_count > negative_count:
            overall_sentiment = "POSITIVE"
            confidence = positive_count / len(results)
        else:
            overall_sentiment = "NEGATIVE"
            confidence = negative_count / len(results)
            
        # Determine sentiment intensity based on confidence
        if confidence > 0.8:
            intensity = "very " + overall_sentiment.lower()
        elif confidence > 0.6:
            intensity = "moderately " + overall_sentiment.lower()
        else:
            intensity = "slightly " + overall_sentiment.lower()
            
        return {
            "dominant_sentiment": overall_sentiment,
            "confidence": confidence,
            "intensity": intensity,
            "detailed_results": results
        }
    except Exception as e:
        print(f"Error during sentiment analysis: {str(e)}")
        return {
            "dominant_sentiment": "NEUTRAL",
            "confidence": 0,
            "intensity": "neutral",
            "error": str(e)
        }

def summarize_with_gemini(text, api_key):
    """Summarize text using Google's Gemini AI."""
    try:
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Get available models
        model = genai.GenerativeModel('gemini-pro')
        
        # Create prompt
        prompt = f"""Please provide a concise summary of the following YouTube video transcript. 
        Focus on the main points and key takeaways:
        
        {text}
        
        Summary:"""
        
        # Generate response
        response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        print(f"Error with Gemini API: {str(e)}")
        # Return None to allow fallback to other models
        return None

def summarize_with_t5(text):
    """Fallback summarization using T5-small model."""
    try:
        device = 0 if torch.cuda.is_available() else -1
        summarizer = pipeline("summarization", model="t5-small", device=device)
        
        # For longer texts, split into chunks and summarize each
        if len(text) > 1000:
            chunks = chunk_text(text, max_length=500)
            summaries = []
            for chunk in chunks:
                result = summarizer(chunk, max_length=100, min_length=30)[0]['summary_text']
                summaries.append(result)
            return " ".join(summaries)
        else:
            return summarizer(text, max_length=100, min_length=30)[0]['summary_text']
    except Exception as e:
        print(f"Error during T5 summarization: {str(e)}")
        return f"Error during summarization: {str(e)}"

def summarize_text(text, video_id, api_key=None):
    """Try to summarize text using Gemini API first, with fallback to T5."""
    # Check cache first
    if video_id in summary_cache:
        return summary_cache[video_id]
    
    summary = None
    
    # Try Gemini API if key is provided
    if api_key:
        summary = summarize_with_gemini(text, api_key)
    
    # Fallback to T5 if Gemini fails or no API key
    if not summary:
        summary = summarize_with_t5(text)
    
    # Cache the result
    summary_cache[video_id] = summary
    return summary

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = None
    video_id = None
    error = None
    sentiment = None
    
    if request.method == 'POST':
        url = request.form['url']
        api_key = request.form.get('api_key', '')  # Get API key if provided
        
        video_id = get_video_id(url)
        
        if not video_id:
            error = "Invalid YouTube URL. Please make sure you're using a URL like: https://www.youtube.com/watch?v=VIDEO_ID"
        else:
            transcript = get_transcript(video_id)
            
            if transcript.startswith("Error"):
                error = transcript
            elif not transcript:
                error = "No transcript available for this video."
            else:
                cleaned_transcript = clean_transcript(transcript)
                try:
                    summary = summarize_text(cleaned_transcript, video_id, api_key)
                    
                    # Check if sentiment is in cache
                    if video_id in sentiment_cache:
                        sentiment = sentiment_cache[video_id]
                    else:
                        # Perform sentiment analysis
                        sentiment = analyze_sentiment(cleaned_transcript)
                        sentiment_cache[video_id] = sentiment
                        
                except Exception as e:
                    error = f"An error occurred during processing: {str(e)}"
    
    return render_template('index.html', summary=summary, video_id=video_id, error=error, sentiment=sentiment)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    # Set a timeout for requests (300 seconds/5 minutes)
    app.config['PERMANENT_SESSION_LIFETIME'] = 300
    app.run(debug=True) 