import torch
from .preprocess import chunk_text
from .models import load_qa_model

def answer_question(question: str, context: str) -> str:
    """
    Answers a question based on the provided context.
    Since context can be long, it chunks the context and runs QA on each chunk,
    returning the answer with the highest confidence score.
    """
    if not context or not context.strip():
        return "No context provided to answer the question."
    if not question or not question.strip():
        return "Please ask a question."
        
    tokenizer, model = load_qa_model()
    
    # Chunk context to fit within model limits
    chunks = chunk_text(context, chunk_size=1500, chunk_overlap=150)
    
    best_answer = None
    best_score = -float('inf')
    
    for chunk in chunks:
        try:
            inputs = tokenizer(question, chunk, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get the highest probability from start/end logits
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            
            score = torch.max(outputs.start_logits).item() + torch.max(outputs.end_logits).item()
            
            if score > best_score and answer_end > answer_start:
                best_score = score
                answer_tokens = inputs.input_ids[0, answer_start:answer_end]
                answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)
                
                # Exclude answers that are just the question or empty
                if answer_text.strip() and len(answer_text) > 1:
                    best_answer = answer_text
        except Exception as e:
            print(f"Error answering question on chunk: {e}")
            
    if best_answer:
        return best_answer
    else:
        return "Sorry, I couldn't find a confident answer to that question in the document."
