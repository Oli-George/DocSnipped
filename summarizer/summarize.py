# pyrefly: ignore [missing-import]
import streamlit as st
from .extract import extract_text_from_file
from .preprocess import chunk_text
from .models import load_summarizer

@st.cache_data(show_spinner=False)
def process_and_summarize_text(text: str, max_words: int = 150, _progress_callback=None) -> str:
    """
    Chunks the input text, summarizes each chunk, and concatenates the results.
    max_words controls the approximate target length of the final summary in words.
    _progress_callback: Optional callable(current, total) for progress updates.
                        Prefixed with _ so Streamlit's cache ignores it.
    """
    if not text or not text.strip():
        return "No text provided to summarize."
        
    # 1. Dynamically Chunk text (recursive character split)
    # Increase chunk size to 3500 characters (approx. 600 words) so it fits in a single BART context.
    char_limit = 3500
    if len(text) <= char_limit:
        # If it fits within a single model forward pass, process it as a single chunk
        chunks = [text]
    else:
        # Otherwise, chunk with standard overlap
        chunks = chunk_text(text, chunk_size=char_limit, chunk_overlap=300)
    
    # 2. Load model
    tokenizer, model, device = load_summarizer()
    
    # 3. Convert word target to token budget and configure generation bounds.
    #    Tokens ≈ words * 1.3 for English text.
    total_token_budget = int(max_words * 1.3)
    
    # If single chunk, allocate full budget and enforce minimum summary length.
    # If multiple chunks, allow each to generate a richer summary (up to 75% of budget)
    # rather than strictly dividing it, letting the model write complete sentences.
    if len(chunks) == 1:
        per_chunk_max = total_token_budget
        per_chunk_min = max(30, int(total_token_budget * 0.7))
    else:
        per_chunk_max = max(60, int(total_token_budget * 0.75))
        per_chunk_min = max(20, per_chunk_max // 3)

    # 4. Summarize chunks
    summarized_chunks = []
    for i, chunk in enumerate(chunks):
        input_text = chunk
        
        try:
            inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
            
            outputs = model.generate(
                **inputs, 
                max_new_tokens=per_chunk_max, 
                min_new_tokens=per_chunk_min, 
                num_beams=4,
                length_penalty=2.5,  # Increased from 2.0 to 2.5 to encourage longer generation
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if summary and summary[-1] not in ".!?":
                last_punctuation = max(summary.rfind('.'), summary.rfind('!'), summary.rfind('?'))
                if last_punctuation != -1:
                    summary = summary[:last_punctuation+1]
            summarized_chunks.append(summary)
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
        
        if _progress_callback:
            _progress_callback(i + 1, len(chunks))
            
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

def process_and_summarize_doc_with_progress(file_obj, max_words: int = 150) -> tuple[str, str]:
    """
    Same as process_and_summarize_doc but drives a Streamlit progress bar
    across both the extraction and summarization stages.
    """
    progress_bar = st.progress(0, text="Extracting text from document...")
    
    # --- Stage 1: Extraction (0% – 30% of progress bar) ---
    def extraction_progress(current, total, stage):
        fraction = current / total
        # Map extraction to 0-30% of the bar
        progress_bar.progress(
            min(int(fraction * 30), 30),
            text=f"Extracting text — page {current} of {total}..."
        )
    
    text = extract_text_from_file(file_obj, progress_callback=extraction_progress)
    progress_bar.progress(30, text="Text extracted. Summarizing...")
    
    # --- Stage 2: Summarization (30% – 100% of progress bar) ---
    def summarization_progress(current, total):
        fraction = current / total
        # Map summarization to 30-100% of the bar
        progress_bar.progress(
            30 + min(int(fraction * 70), 70),
            text=f"Summarizing — chunk {current} of {total}..."
        )
    
    summary = process_and_summarize_text(text, max_words=max_words, _progress_callback=summarization_progress)
    progress_bar.progress(100, text="Done!")
    progress_bar.empty()  # Remove the progress bar once complete
    
    return summary, text
