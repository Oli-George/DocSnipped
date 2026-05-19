import streamlit as st
from summarizer.summarize import process_and_summarize_text, process_and_summarize_doc
from summarizer.qa import answer_question

# Initialize session state variables
if "summary" not in st.session_state:
    st.session_state["summary"] = ""
if "source_text" not in st.session_state:
    st.session_state["source_text"] = ""
if "current_choice" not in st.session_state:
    st.session_state["current_choice"] = None

def TextInference(snippet):
    return process_and_summarize_text(snippet)

def DocInference(document):
    # process_and_summarize_doc returns (summary, text)
    summary, text = process_and_summarize_doc(document)
    return summary, text

def main():
    st.write("Welcome to DocSnipped! This is a simple app that allows you to generate snippets from your documents using a language model. " \
    "Please upload your document and click the button to generate snippets.")

    options = ["Select an Option", "Paste Text", "Upload Document"]
    choice = st.selectbox("Will you be pasting the text snippet or uploading a document?", options)
    choice_index = options.index(choice)

    # Clear state if the user changes the input option
    if st.session_state["current_choice"] != choice_index:
        st.session_state["current_choice"] = choice_index
        st.session_state["summary"] = ""
        st.session_state["source_text"] = ""

    if choice_index == 1:
        st.write("Please paste your text snippet in the text area below and click the button to generate snippets.")
        text_snippet = st.text_area("Paste your text snippet here:")
        if st.button("Generate Summary from Text"):
            if text_snippet:
                with st.spinner("Generating summary..."):
                    text_summary = TextInference(text_snippet)
                    st.session_state["summary"] = text_summary
                    st.session_state["source_text"] = text_snippet
            else:
                st.warning("Please paste some text first.")

    elif choice_index == 2:
        try:
            doc = st.file_uploader("Upload your document here (Please, text files only)", type=["pdf", "docx", "txt"])
        except Exception as e:
            st.write(f"Error occurred while uploading the document: {e}")
            st.text("Please try uploading again.")
            doc = None

        if doc is not None:
            st.write("Document uploaded successfully! Now, click the button below to generate a summary.")
            
            if st.button("Generate Document Summary"):
                with st.spinner("Generating summary..."):
                    doc_summary, doc_text = DocInference(doc)
                    st.session_state["summary"] = doc_summary
                    st.session_state["source_text"] = doc_text

    else:
        st.write("Please select an option to either paste text or upload a document to generate snippets.")

    # Render summary and QA interface if summary exists in session state
    if st.session_state["summary"]:
        st.divider()
        st.subheader("Summary")
        st.write(st.session_state["summary"])
        
        st.divider()
        st.subheader("Ask a Question")
        st.write("You can ask questions about the text or document you provided.")
        
        question = st.text_input("Enter your question here:")
        if st.button("Get Answer"):
            if question:
                with st.spinner("Searching for the answer..."):
                    answer = answer_question(question, st.session_state["source_text"])
                    st.write("**Answer:**")
                    st.write(answer)
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    main()