from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def get_llm(model='gemini-2.5-flash-lite'):
    return ChatGoogleGenerativeAI(model=model, max_retries=1)
    

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


def get_qa_chain():
    embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # 파인콘 저장 데이터 불러오기
    index_name = 'tax-gemini-index'
    database = PineconeVectorStore.from_existing_index(
        index_name=index_name, embedding=embedding
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

    # RetrievalQA 대신 LCEL 체인 (스트리밍 가능)
    qa_chain = (
        #1단계에서 context와 question에 대한 값을 넣어주는 용도임...
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain


def get_ai_message(user_message, history):
    if history:
        rewritten_question = get_dictionary_chain().invoke(
            {"question": user_message, "history": history}
        )
    else:
        rewritten_question = user_message
    print("=== 재작성된 질문:", rewritten_question)   # 추가
    return get_qa_chain().stream(rewritten_question)