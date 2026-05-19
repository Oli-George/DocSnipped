import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
from transformers import pipeline

@st.cache_resource
def load_summarizer():
    """
    Loads the HuggingFace summarization model and tokenizer.
    Using @st.cache_resource ensures the model is only loaded once into memory
    and persists across Streamlit reruns.
    """
    print("Loading summarizer model...")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    return tokenizer, model

@st.cache_resource
def load_qa_model():
    """
    Loads the HuggingFace question answering model.
    """
    print("Loading QA model...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
    return tokenizer, model
