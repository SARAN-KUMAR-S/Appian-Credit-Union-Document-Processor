import streamlit as st
import os
import tempfile
from datetime import datetime
from document_processor import DocumentProcessor, create_document_summary

def main():
    st.set_page_config(page_title="Appian Credit Union Document Processor", layout="wide")
    
    st.title("Appian Credit Union Document Processor")
    st.write("Upload financial documents for automatic classification and information extraction")
    
    try:
        processor = DocumentProcessor()
    except ValueError as e:
        st.error(f"Configuration error: {e}")
        return
    
    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Process Documents"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        col1, col2 = st.columns(2)
        results = []
        
        for idx, file in enumerate(uploaded_files):
            status_text.text(f"Processing {file.name}...")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name
            
            try:
                doc = processor.process_document(tmp_path)
                doc_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.name}"
                processor.db.save_document(doc_id, doc)
                summary = create_document_summary(doc)
                results.append((file.name, doc, summary))
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
            finally:
                os.unlink(tmp_path)
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.text("Processing complete!")
        
        st.subheader("Processing Results")
        for filename, doc, summary in results:
            with st.expander(f"Document: {filename}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area("Summary", summary, height=400)
                with col2:
                    if st.button(f"View Raw Text ({filename})", key=f"raw_{filename}"):
                        st.text_area("Raw Text", doc.raw_text, height=400)

        st.download_button(
            "Download All Results",
            "\n---\n".join([f"File: {f}\n{s}" for f, _, s in results]),
            "document_analysis.txt"
        )

if __name__ == "__main__":
    main()