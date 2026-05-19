import pdfplumber
import docx
import io

def extract_text_from_file(file_obj) -> str:
    """
    Extracts text from a Streamlit UploadedFile object.
    Supports .txt, .pdf, and .docx files.
    """
    file_name = file_obj.name.lower()
    
    if file_name.endswith('.txt'):
        return file_obj.getvalue().decode("utf-8")
        
    elif file_name.endswith('.pdf'):
        text = ""
        pdf_bytes = io.BytesIO(file_obj.getvalue())
        with pdfplumber.open(pdf_bytes) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
        
    elif file_name.endswith('.docx'):
        # docx needs a file-like object
        doc = docx.Document(io.BytesIO(file_obj.getvalue()))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
        
    else:
        raise ValueError(f"Unsupported file format: {file_name}. Please upload a txt, pdf, or docx file.")
