import torch
import streamlit as st
from .preprocess import chunk_text
from .models import load_qa_model

@st.cache_data
def answer_question(question: str, context: str) -> str:
    """
    Answers a question based on the provided context.
    Using @st.cache_data makes repeated identical questions instant.
    For first-time speed, we batch the chunks and run them in parallel on GPU/CPU.
    """
    if not context or not context.strip():
        return "No context provided to answer the question."
    if not question or not question.strip():
        return "Please ask a question."
        
    tokenizer, model, device = load_qa_model()
    
    # Chunk context to fit within model limits
    chunks = chunk_text(context, chunk_size=1500, chunk_overlap=150)
    
    if not chunks:
        return "Sorry, I couldn't find a confident answer to that question in the document."
        
    best_answer = None
    best_score = -float('inf')
    
    try:
        # Batch all chunks together so they run in parallel in a single forward pass
        inputs = tokenizer(
            [question] * len(chunks), 
            chunks, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Iterate through the batch outputs to find the best answer
        for i in range(len(chunks)):
            start_logits = outputs.start_logits[i]
            end_logits = outputs.end_logits[i]
            
            answer_start = torch.argmax(start_logits)
            answer_end = torch.argmax(end_logits) + 1
            
            score = torch.max(start_logits).item() + torch.max(end_logits).item()
            
            if score > best_score and answer_end > answer_start:
                best_score = score
                answer_tokens = inputs.input_ids[i, answer_start:answer_end]
                answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)
                
                # Exclude answers that are just the question or empty
                if answer_text.strip() and len(answer_text) > 1:
                    best_answer = answer_text
    except Exception as e:
        print(f"Error during batched question answering: {e}")
        
    if best_answer:
        return best_answer
    else:
        return "Sorry, I couldn't find a confident answer to that question in the document."
