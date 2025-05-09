from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoModelForSequenceClassification
import os

# Create model_cache directory if it doesn't exist
os.makedirs("model_cache", exist_ok=True)

print("Downloading T5-small model for summarization...")
# Download T5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small", cache_dir="./model_cache")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small", cache_dir="./model_cache")

print("Downloading DistilBERT model for sentiment analysis...")
# Download DistilBERT model and tokenizer for sentiment analysis
sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", cache_dir="./model_cache")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", cache_dir="./model_cache")

print("Models downloaded successfully!")
print("These models will be used by the application without needing to download them again.") 