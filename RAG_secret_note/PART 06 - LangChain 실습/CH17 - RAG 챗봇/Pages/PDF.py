import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_teddynote import logging
from dotenv import load_dotenv
import os

load_dotenv()

logging.langsmith('[Project] PDF RAG')

st.title('PDF 기반 QA')

if not os.path.exists('.cache'):
    os.mkdir('.cache')

if not os.path.exists('.cache/files'):
    os.mkdir('.cache/files')

if not os.path.exists('.cache/embeddings'):
    os.mkdir('.cache/embeddings')

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'chain' not in st.session_state:
    st.session_state['chain'] = None

with st.sidebar:
    clear_btn = st.button('대화 초기화')
    uploaded_file = st.file_uploader('파일 업로드', type=['pdf'])
    selected_model = st.selectbox(
        'LLM 선택', ['gpt-4o', 'gpt-4-turbo', 'gpt-4o-mini'], index=0
    )

def print_messages():
    for chat_message in st.session_state['messages']:
        st.chat_message(chat_message.role).write(chat_message.content)

def add_message(role, message):
    st.session_state['messages'].append(ChatMessage(role=role, content=message))

@st.cache_resource(show_spinner='업로드한 파일을 처리 중입니다...')
def embed_file(file):
    file_content = file.read()
    file_path = f'./.cache/files/{file.name}'
    with open(file_path, 'wb') as f:
        f.write(file_content)
    
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    retriever = vectorstore.as_retriever()
    return retriever

def create_chain(retriever, model_name='gpt-4o'):
    prompt = load_prompt('prompts/pdf-rag.yaml', encoding='utf-8')
    
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    chain = (
        {'context': retriever, 'question': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

if uploaded_file:
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever, model_name=selected_model)
    st.session_state['chain'] = chain

if clear_btn:
    st.session_state['messages'] = []

print_messages()

user_input = st.chat_input('궁금한 내용을 물어보세요!')

warning_msg = st.empty()

if user_input:
    chain = st.session_state['chain']

    if chain is not None:
        st.chat_message('user').write(user_input)
        response = chain.stream(user_input)
        with st.chat_message('assistant'):
            container = st.empty()

            ai_answer = ''
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)
    
        add_message('user', user_input)
        add_message('assistant', ai_answer)
    else:
        warning_msg.error('파일을 업로드해 주세요.')