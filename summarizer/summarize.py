from .extract import extract_text_from_file
from .preprocess import chunk_text
from .models import load_summarizer

def process_and_summarize_text(text: str) -> str:
    """
    Chunks the input text, summarizes each chunk, and concatenates the results.
    """
    if not text or not text.strip():
        return "No text provided to summarize."
        
    # 1. Chunk text (recursive character split)
    chunks = chunk_text(text, chunk_size=1500, chunk_overlap=150)
    
    # 2. Load model
    tokenizer, model = load_summarizer()
    
    # 3. Summarize chunks
    summarized_chunks = []
    for chunk in chunks:
        # t5 models perform best with the 'summarize: ' prefix
        input_text = "summarize: " + chunk
        
        try:
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            # approximate tokens as len(chunk) // 4
            input_length_approx = len(chunk) // 4
            max_len = min(200, max(30, input_length_approx - 10))
            min_len = min(30, max_len - 10)
            if min_len < 5: min_len = 5
            
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_len, 
                min_new_tokens=min_len, 
                do_sample=False
            )
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            summarized_chunks.append(summary)
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            
    final_summary = " ".join(summarized_chunks)
    return final_summary

def process_and_summarize_doc(file_obj) -> str:
    """
    Extracts text from a document and summarizes it.
    """
    # 1. Extract text from uploaded file
    text = extract_text_from_file(file_obj)
    
    # 2. Summarize
    return process_and_summarize_text(text)
