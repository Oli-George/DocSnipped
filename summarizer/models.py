import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
