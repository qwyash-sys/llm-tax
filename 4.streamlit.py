import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
#########기존 ipynb에서 개발했던 랭체인##########
from dotenv import load_dotenv


from llm import get_ai_message
#########기존 ipynb에서 개발했던 랭체인##########


st.set_page_config(page_title="소득세 챗봇", page_icon="🤖")
st.title("세금 챗봇")
st.caption("세금 관련한 모든것을 답해줘요")

load_dotenv()
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if user_question := st.chat_input(placeholder="세금관련 궁금증을 질문해보세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content":user_question})

    # 직전 대화를 LangChain 메시지로 변환 (방금 넣은 user 질문은 제외)
    history = []
    for msg in st.session_state.message_list[:-1]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))

    with st.spinner("답변 생성중입니다"):
        ai_message_stream = get_ai_message(user_question, history)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_message_stream)   # ← 반환값 받기
        st.session_state.message_list.append({"role": "ai", "content": ai_message})
        
with st.sidebar:
    if st.button("🔧 임베딩 연속 호출 테스트"):
        from llm import debug_embedding_test
        for ok, msg in debug_embedding_test():
            (st.success if ok else st.error)(msg)
