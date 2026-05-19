import spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

# Load SpaCy model for sentence boundary detection
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:

    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# We increase max_length for SpaCy to handle large documents
nlp.max_length = 5000000 

def clean_text(text: str) -> str:
    """Removes excessive whitespaces and newlines."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list[str]:
    """
    Chunks large texts using RecursiveCharacterTextSplitter.
    Because transformer models have a max token limit, we need to ensure the chunk
    is within the character limit (roughly equating 1000 chars to a safe token count).
    """
    cleaned_text = clean_text(text)
    
    # LangChain's RecursiveCharacterTextSplitter tries to split by paragraphs, then sentences, then words
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_text(cleaned_text)
    return chunks
