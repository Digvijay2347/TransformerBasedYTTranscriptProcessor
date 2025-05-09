import sys
import os
import argparse
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import torch

# Import Gemini API if available
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

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

def chunk_text(text, max_length=1000):
    """Split text into chunks that don't exceed max_length."""
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
        
        print("Analyzing sentiment (this may take a moment)...")
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
        chunks = chunk_text(text)
        summaries = []
        for chunk in chunks:
            if len(chunk.split()) < 30:  # Skip very short chunks
                summaries.append(chunk)
                continue
                
            summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        
        return " ".join(summaries)
    except Exception as e:
        print(f"Error during T5 summarization: {str(e)}")
        return f"Error during summarization: {str(e)}"

def summarize_text(text, video_id=None, api_key=None):
    """Try to summarize text using Gemini API first, with fallback to T5."""
    # Check cache if video_id is provided
    if video_id and video_id in summary_cache:
        return summary_cache[video_id]
    
    summary = None
    
    # Try Gemini API if key is provided and library is available
    if api_key and GEMINI_AVAILABLE:
        print("Using Gemini AI for summarization...")
        summary = summarize_with_gemini(text, api_key)
    
    # Fallback to T5 if Gemini fails or no API key
    if not summary:
        print("Using T5 model for summarization (this may take a moment)...")
        summary = summarize_with_t5(text)
    
    # Cache the result if video_id is provided
    if video_id:
        summary_cache[video_id] = summary
    return summary

def main():
    """Main function to process a YouTube URL and generate a summary."""
    parser = argparse.ArgumentParser(description='Summarize a YouTube video transcript')
    parser.add_argument('url', help='YouTube video URL')
    parser.add_argument('--api-key', '-k', help='Google Gemini API key for faster summarization')
    parser.add_argument('--sentiment', '-s', action='store_true', help='Include sentiment analysis')
    
    args = parser.parse_args()
    
    url = args.url
    api_key = args.api_key
    analyze_sentiment_flag = args.sentiment
    
    video_id = get_video_id(url)
    if not video_id:
        print("Invalid YouTube URL")
        sys.exit(1)
    
    print(f"Fetching transcript for video ID: {video_id}")
    transcript = get_transcript(video_id)
    if transcript.startswith("Error"):
        print(transcript)
        sys.exit(1)
    
    if not transcript:
        print("No transcript available for this video.")
        sys.exit(1)
    
    print("Cleaning transcript...")
    cleaned_transcript = clean_transcript(transcript)
    
    try:
        summary = summarize_text(cleaned_transcript, video_id, api_key)
        print("\nSummary:\n")
        print(summary)
        
        if analyze_sentiment_flag:
            # Check sentiment cache first
            if video_id in sentiment_cache:
                sentiment = sentiment_cache[video_id]
            else:
                sentiment = analyze_sentiment(cleaned_transcript)
                # Cache the result
                if video_id:
                    sentiment_cache[video_id] = sentiment
            
            print("\nSentiment Analysis:\n")
            print(f"Dominant Sentiment: {sentiment['dominant_sentiment']}")
            print(f"Emotional Tone: {sentiment['intensity']}")
            print(f"Confidence: {sentiment['confidence']:.2f}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 