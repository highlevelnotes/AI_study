import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain_teddynote.models import MultiModal

from dotenv import load_dotenv
import os

# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("[Project] 이미지 인식")

if not os.path.exists('.cache'):
    os.mkdir('.cache')

if not os.path.exists('.cache/files'):
    os.mkdir('.cache/files')

if not os.path.exists('.cache/embeddings'):
    os.mkdir('.cache/embeddings')

st.title("이미지 인식 기반 챗봇")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

main_tab1, main_tab2 = st.tabs(['이미지', '대화내용'])

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 파일 업로드
    uploaded_file = st.file_uploader("파일 업로드", type=['jpg', 'jpeg', 'png'])

    # 모델 선택 메뉴
    selected_model = st.selectbox(
        "LLM 선택", ['gpt-4o', 'gpt-4o-mini'], index=0
    )

    system_prompt = st.text_area(
        '시스템 프롬프트',
        '당신은 표(재무제표)를 해석하는 금융 AI 어시스턴트 입니다.\n\n당신의 임무는 주어진 테이블 형식의 재무제표를 바탕으로 흥미로운 사실을 정리하여 친절하게 답변하는 것입니다.',
        height=200
    )


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 이미지를 처리 중입니다...")
def process_imagefile(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    return file_path

def generate_answer(image_filepath, system_prompt, user_prompt, model_name='gpt-4o'):
    llm = ChatOpenAI(
        temperature=0,
        model_name=model_name,
    )

    multimodal = MultiModal(llm, system_prompt=system_prompt, user_prompt=user_prompt)

    answer = multimodal.stream(image_filepath)
    return answer

# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = main_tab2.empty()

if uploaded_file:
    image_filepath = process_imagefile(uploaded_file)
    main_tab1.image(image_filepath)

# 만약에 사용자 입력이 들어오면...
if user_input:
    if uploaded_file:
        image_filepath = process_imagefile(uploaded_file)
        response = generate_answer(
            image_filepath, system_prompt, user_input, selected_model
        )

        main_tab2.chat_message('user').write(user_input)

        with main_tab2.chat_message('assistant'):
            container = st.empty()

            ai_answer = ''
            for token in response:
                ai_answer += token.content
                container.markdown(ai_answer)
        
        add_message('user', user_input)
        add_message('assistant', ai_answer)
    else:
        warning_msg.error('이미지를 업로드 해주세요')