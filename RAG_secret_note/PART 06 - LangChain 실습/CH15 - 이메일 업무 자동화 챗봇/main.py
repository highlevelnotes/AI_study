import os
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain_teddynote.prompts import load_prompt
from pydantic import BaseModel, Field

load_dotenv()

st.title('Email 요약기')

class EmailSummary(BaseModel):
    person: str = Field(description='메일을 보낸 사람')
    company: str = Field(description='메일을 보낸 사람의 회사 정보')
    email: str = Field(description='메일을 보낸 사람의 이메일 주소')
    subject: str = Field(description='메일 제목')
    summary: str = Field(description='메일 본문을 요약한 텍스트')
    date: str = Field(description='메일 본문에 언급된 미팅 날짜와 시간')

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

with st.sidebar:
    clean_btn = st.button('대화 초기화')

