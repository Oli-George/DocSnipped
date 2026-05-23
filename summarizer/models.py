# pyrefly: ignore [missing-import]
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
from transformers import pipeline

import torch

@st.cache_resource(show_spinner=False)
def load_summarizer():
    """
    Loads the HuggingFace summarization model and tokenizer.
    Using @st.cache_resource ensures the model is only loaded once into memory
    and persists across Streamlit reruns.
    """
    print("Loading summarizer model...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    return tokenizer, model, device

@st.cache_resource(show_spinner=False)
def load_qa_model():
    """
    Loads the HuggingFace question answering model.
    """
    print("Loading QA model...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
    
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    return tokenizer, model, device

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    """
    Loads the HuggingFace sentiment analysis model.
    Using cardiffnlp/twitter-roberta-base-sentiment-latest to support
    Positive, Neutral, and Negative sentiments on general natural text.
    """
    print("Loading sentiment model...")
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device
    )
    return sentiment_pipeline
