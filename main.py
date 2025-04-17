import streamlit as st
import os
import fitz  # PyMuPDF
from docx import Document

from langchain_openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# File text extraction logic
def extract_text_from_file(uploaded_file, file_type):
    if file_type == 'txt':
        return uploaded_file.read().decode()
    elif file_type == 'pdf':
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    elif file_type == 'docx':
        doc = Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

# Main logic to generate a response from file or directly via LLM
def generate_response(uploaded_file, file_type, openai_api_key, query_text):
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Set environment variable for API key
    os.environ["OPENAI_API_KEY"] = openai_api_key
    logging.debug(f"API Key set: {openai_api_key[:4]}...")  # Log partial key for security

    if uploaded_file is not None:
        if file_type not in ['txt', 'pdf', 'docx']:
            raise ValueError("Unsupported file type. Use txt, pdf, or docx.")
        raw_text = extract_text_from_file(uploaded_file, file_type)
        documents = [raw_text]
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(texts, embeddings)
        retriever = db.as_retriever()
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(model="gpt-3.5-turbo-instruct", temperature=0, api_key=openai_api_key),
            chain_type='stuff',
            retriever=retriever
        )
        return qa.run(query_text)
    else:
        llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0, api_key=openai_api_key)
        return llm(query_text)

# Streamlit UI layout
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

uploaded_file = st.file_uploader('Upload an article (optional)', type=['txt', 'pdf', 'docx'])
query_text = st.text_input('Enter your question:', placeholder='Ask a question about the document or general topic')
openai_api_key = st.text_input('OpenAI API Key', type='password')

result = []
with st.form('qa_form', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted:
        if not openai_api_key.startswith('sk-'):
            st.warning("⚠️ Please enter a valid OpenAI API key starting with 'sk-'")
        elif not query_text.strip():
            st.warning("⚠️ Please enter a question to ask")
        else:
            with st.spinner("Processing..."):
                file_type = uploaded_file.name.split('.')[-1] if uploaded_file else None
                response = generate_response(uploaded_file, file_type, openai_api_key, query_text)
                result.append(response)

if result:
    st.info(result[-1])
