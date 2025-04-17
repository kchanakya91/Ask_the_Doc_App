import streamlit as st
from langchain_openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import fitz  # PyMuPDF
from docx import Document

def extract_text_from_file(uploaded_file, file_type):
    if file_type == 'txt':
        return uploaded_file.read().decode()
    elif file_type == 'pdf':
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
        return text
    elif file_type == 'docx':
        doc = Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return ""

def generate_response(uploaded_file, file_type, openai_api_key, query_text):
    if uploaded_file is not None:
        raw_text = extract_text_from_file(uploaded_file, file_type)
        documents = [raw_text]
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = Chroma.from_documents(texts, embeddings)
        retriever = db.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
        return qa.run(query_text)
    else:
        # Fallback to plain LLM response when no document is uploaded
        llm = OpenAI(openai_api_key=openai_api_key)
        return llm(query_text)

# Streamlit UI
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# Allow txt, pdf, docx
uploaded_file = st.file_uploader('Upload an article (optional)', type=['txt', 'pdf', 'docx'])

# Inputs (enabled always now)
query_text = st.text_input('Enter your question:', placeholder='Type your question here.')
openai_api_key = st.text_input('OpenAI API Key', type='password')

# Form and submission
result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted and openai_api_key.startswith('sk-') and query_text:
        with st.spinner('Generating response...'):
            file_type = uploaded_file.name.split('.')[-1] if uploaded_file else None
            response = generate_response(uploaded_file, file_type, openai_api_key, query_text)
            result.append(response)
            del openai_api_key
    elif submitted and not openai_api_key.startswith('sk-'):
        st.warning("Please enter a valid OpenAI API key starting with 'sk-'.")

if result:
    st.info(result[-1])
