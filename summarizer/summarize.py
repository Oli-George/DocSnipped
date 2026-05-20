import streamlit as st
from .extract import extract_text_from_file
from .preprocess import chunk_text
from .models import load_summarizer

@st.cache_data
def process_and_summarize_text(text: str, max_words: int = 150) -> str:
    """
    Chunks the input text, summarizes each chunk, and concatenates the results.
    max_words controls the approximate target length of the final summary in words.
    """
    if not text or not text.strip():
        return "No text provided to summarize."
        
    # 1. Chunk text (recursive character split)
    chunks = chunk_text(text, chunk_size=1500, chunk_overlap=150)
    
    # 2. Load model
    tokenizer, model, device = load_summarizer()
    
    # 3. Convert word target to token budget and distribute across chunks.
    #    Tokens ≈ words * 1.3 for English text.
    total_token_budget = int(max_words * 1.3)
    per_chunk_max = max(30, total_token_budget // max(1, len(chunks)))
    per_chunk_min = max(5, per_chunk_max // 4)

    # 4. Summarize chunks
    summarized_chunks = []
    for chunk in chunks:
        # t5 models perform best with the 'summarize: ' prefix
        input_text = "summarize: " + chunk
        
        try:
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
            
            outputs = model.generate(
                **inputs, 
                max_new_tokens=per_chunk_max, 
                min_new_tokens=per_chunk_min, 
                do_sample=False
            )
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            summarized_chunks.append(summary)
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            
    final_summary = " ".join(summarized_chunks)
    return final_summary

def process_and_summarize_doc(file_obj, max_words: int = 150) -> tuple[str, str]:
    """
    Extracts text from a document and summarizes it.
    Returns a tuple of (summary, extracted_text).
    max_words controls the approximate target length of the final summary in words.
    """
    # 1. Extract text from uploaded file
    text = extract_text_from_file(file_obj)
    
    # 2. Summarize
    return process_and_summarize_text(text, max_words=max_words), text
