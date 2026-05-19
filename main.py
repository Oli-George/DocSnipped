import streamlit as st

from summarizer.summarize import process_and_summarize_text, process_and_summarize_doc

def TextInference(snippet):
    return process_and_summarize_text(snippet)

def DocInference(document):
    return process_and_summarize_doc(document)

def main():
    st.write("Welcome to DocSnipped! This is a simple app that allows you to generate snippets from your documents using a language model. " \
    "Please upload your document and click the button to generate snippets.")

    options = ("Select an Option","Paste Text", "Upload Document")
    choice =st.selectbox("Will you be pasting the text snippet or uploading a document?", options)
    choice_index = options.index(choice)

    if choice_index == 1:
        st.write("Please paste your text snippet in the text area below and click the button to generate snippets.")
        text_snippet = st.text_area("Paste your text snippet here:")
        if st.button("Generate Summary from Text") and text_snippet is not None:
            st.divider()
            st.write("Generating summary...")
            text_summary = TextInference(text_snippet)
            st.write("Summary generated successfully! Here it is:")
            st.write(text_summary)

    elif choice_index == 2:
        try:
            doc = st.file_uploader("Upload your document here (Please, text files only)", type=["pdf", "docx", "txt"])
        except Exception as e:
            st.write(f"Error occurred while uploading the document: {e}")
            st.text("Please try uploading again.")

        st.write("Document uploaded successfully! Now, click the button below to generate a summary.")

        if doc is not None:
            if st.button("Generate Document Summary"):
                st.divider()
                st.write("Generating summary...")
                doc_summary = DocInference(doc)
                st.write("Summary generated successfully! Here is your generated summary:")
                
                st.write(doc_summary)


    else:
        st.write("Please select an option to either paste text or upload a document to generate snippets.")


if __name__ == "__main__":
    main()