# YouTube Video Summarizer

A web application that allows users to summarize YouTube videos by extracting their transcripts and generating concise summaries using natural language processing. The tool supports both command-line and web-based interfaces.

## Features

- Extract transcripts from YouTube videos automatically
- Generate concise summaries of video content
- Dual interface: Command-line tool and web application
- Intelligent model selection: Uses Google's Gemini AI if an API key is provided, falls back to local models if not
- Embedded video player for convenient viewing alongside summaries
- Caching system to avoid re-processing previously summarized videos

## Technologies Used

### Core Technologies

- **Python 3.7+**: The primary programming language used for development
- **Flask**: A lightweight web framework used to create the web application interface
- **HTML/CSS/JavaScript**: Front-end technologies for the web interface

### Natural Language Processing

- **Transformers Library**: Hugging Face's transformers library is used for text summarization with the T5 model
- **T5-small Model**: A transformer-based language model that can perform text summarization
- **Google Gemini AI API**: A powerful language model from Google that provides high-quality summarization (optional)

### YouTube Data Extraction

- **youtube-transcript-api**: A Python library used to extract transcripts from YouTube videos

### Additional Components

- **PyTorch**: Used as the backend for the transformers library
- **Argparse**: Used for command-line argument parsing in the CLI version

## How the Technologies Work Together

1. **YouTube Transcript Extraction**: When a user inputs a YouTube URL, the application uses the youtube-transcript-api to retrieve the transcript from the video.

2. **Text Processing Pipeline**:
   - The raw transcript is cleaned to remove timestamps and formatting
   - For longer transcripts, the text is chunked into manageable segments

3. **Summarization Engine Selection**:
   - If a Gemini API key is provided, the application sends the transcript to Google's Gemini AI via API
   - If no key is provided, it uses the local T5-small model through the transformers library
   - This dual approach allows for both high-quality (Gemini) and offline (T5) summarization

4. **Result Caching**:
   - All summaries are cached by video ID to improve performance for repeat requests
   - This reduces API calls and computational load

5. **Web Interface**:
   - The Flask web server handles user requests and displays results
   - The responsive HTML/CSS interface provides a user-friendly experience
   - JavaScript is used to handle loading states and toggle API key input

## Setup and Installation

1. Ensure Python 3.7 or higher is installed
2. Clone this repository
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

Run the script with a YouTube URL as an argument:

```bash
python youtube_summarizer.py https://www.youtube.com/watch?v=VIDEO_ID
```

To use Gemini API (recommended for better summaries):

```bash
python youtube_summarizer.py https://www.youtube.com/watch?v=VIDEO_ID --api-key YOUR_GEMINI_API_KEY
```

### Web Interface

Run the Flask web application:

```bash
python app.py
```

Then open a web browser and navigate to: http://127.0.0.1:5000/

## Using Gemini API (Recommended)

For faster and higher-quality summaries, you can use Google's Gemini AI:

1. Get a Gemini API key from [Google AI Studio](https://ai.google.dev/)
2. Enter the API key in the web interface when prompted or use the --api-key parameter with CLI
3. Enjoy much faster and more coherent summaries

## Limitations

- Only works with YouTube videos that have available transcripts
- Optimized for English transcripts
- First-time model download may take several minutes if using local models
- T5-small model has limited context window, which may affect summary quality for very long videos 