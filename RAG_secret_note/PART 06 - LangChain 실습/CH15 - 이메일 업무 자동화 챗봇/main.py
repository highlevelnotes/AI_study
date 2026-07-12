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
    clear_btn = st.button('대화 초기화')

def print_messages():
    for chat_message in st.session_state['messages']:
        st.chat_message(chat_message.role).write(chat_message.content)
    
def add_message(role, message):
    st.session_state['messages'].append(ChatMessage(role=role, content=message))

def create_email_parsing_chain():
    output_parser = PydanticOutputParser(pydantic_object=EmailSummary)

    prompt = PromptTemplate.from_template(
        '''
You are a helpful assistant. Please answer the following questions in KOREAN.

#QUESTION:
다음의 이메일 내용 중에서 주요 내용을 추출해 주세요.

#EMAIL CONVERSATION:
{email_conversation}

#FORMAT:
{format}
'''
    )

    prompt = prompt.partial(format=output_parser.get_format_instructions())

    chain = prompt | ChatOpenAI(model='gpt-4-turbo') | output_parser

    return chain

def create_report_chain():
    prompt = load_prompt('prompts/email.yaml', encoding='utf-8')

    output_parser = StrOutputParser()

    chain = prompt | ChatOpenAI(model='gpt-4-turbo') | output_parser

    return chain

if clear_btn:
    st.session_state['messages'] = []

print_messages()

user_input = st.chat_input('궁금한 내용을 물어보세요!')

if user_input:
    st.chat_message('user').write(user_input)
    
    email_chain = create_email_parsing_chain()
    answer = email_chain.invoke({'email_conversation': user_input})

    params = {'engine': 'google', 'gl': 'kr', 'hl': 'ko', 'num': '3'}
    search = SerpAPIWrapper(params=params)
    search_query = f'{answer.person} {answer.company} {answer.email}'
    search_result = search.run(search_query)
    search_result = eval(search_result)

    search_result_string = '\n'.join(search_result)

    report_chain = create_report_chain()
    report_chain_input = {
        'sender': answer.person,
        'additional_information': search_result_string,
        'company': answer.company,
        'email': answer.email,
        'subject': answer.subject,
        'summary': answer.summary,
        'date': answer.date,
    }

    response = report_chain.stream(report_chain_input)
    with st.chat_message('assistant'):
        container = st.empty()

        ai_answer = ''
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)
        
    add_message('user', user_input)
    add_message('assistant', ai_answer)