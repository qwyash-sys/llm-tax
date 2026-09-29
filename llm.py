from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st

@st.cache_resource
def get_llm(model='gemini-2.5-flash-lite'):
    return ChatGoogleGenerativeAI(model=model, max_retries=1)
    
@st.cache_resource
def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
        이전 대화 내용을 참고해서, 사용자의 질문을 완전한 형태로 바꿔주세요.
        예를 들어 "그럼 그 사람은?" 같은 질문은 앞 대화를 보고 구체적으로 풀어주세요.
        그리고 우리의 사전을 참고해서 표현을 변경해주세요.
        변경할 필요가 없으면 질문을 그대로 리턴하세요. 질문만 리턴하세요.
        사전: {dictionary}
        """),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    dictionary_chain = prompt | llm | StrOutputParser()
    return dictionary_chain


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def get_qa_chain():
    embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    database = PineconeVectorStore.from_existing_index(
        index_name='tax-gemini-index', embedding=embedding
    )
    retriever = database.as_retriever(search_kwargs={'k': 4})
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an assistant for question-answering tasks. "
         "Use the following retrieved context to answer the question. "
         "If you don't know the answer, just say that you don't know. "
         "Use three sentences maximum and keep the answer concise.\n\n"
         "Context: {context}"),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return retriever, chain   # 튜플로 따로 반환



# def get_ai_message(user_message, history):
#     if history:
#         rewritten_question = get_dictionary_chain().invoke(
#             {"question": user_message, "history": history}
#         )
#     else:
#         rewritten_question = user_message
#     print("=== 재작성된 질문:", rewritten_question)   # 추가
#     return get_qa_chain().stream(rewritten_question)

def get_ai_message(user_message, history):
    if history:
        rewritten_question = get_dictionary_chain().invoke(
            {"question": user_message, "history": history}
        )
    else:
        rewritten_question = user_message

    st.info(f"재작성된 질문: {rewritten_question!r}")   # ← 채팅 화면에 바로 표시됨

    retriever, chain = get_qa_chain()
    try:
        docs = retriever.invoke(rewritten_question)
    except Exception as e:
        st.error(f"임베딩 실패! 질문 내용: {rewritten_question!r} / 에러: {e!r}")
        raise
    context = format_docs(docs)

    return chain.stream({"context": context, "question": rewritten_question})


def debug_embedding_test(n=3):
    embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    results = []
    for i in range(n):
        try:
            vec = embedding.embed_query("테스트 질문입니다")
            results.append((True, f"성공, 차원: {len(vec)}"))
        except Exception as e:
            results.append((False, repr(e)))
    return results
