from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # 인덱스가 Gemini 임베딩이라 유지
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

LLM_MODEL = "gpt-4.1-mini"   # 더 아끼려면 "gpt-4.1-nano"
INDEX_NAME = "tax-gemini-index"
DICTIONARY = ["사람을 나타내는 표현 -> 거주자"]


def get_llm(temperature=0, max_tokens=None):
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,   # 출력 토큰 상한 (비용 방어)
        max_retries=1,
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@lru_cache(maxsize=1)
def get_dictionary_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "이전 대화 내용을 참고해서, 사용자의 질문을 완전한 형태로 바꿔주세요.\n"
         '예를 들어 "그럼 그 사람은?" 같은 질문은 앞 대화를 보고 구체적으로 풀어주세요.\n'
         "그리고 우리의 사전을 참고해서 표현을 변경해주세요.\n"
         "변경할 필요가 없으면 질문을 그대로 리턴하세요. 질문만 리턴하세요.\n"
         "사전: {dictionary}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]).partial(dictionary=str(DICTIONARY))
    # 질문 재작성은 짧으니 출력 상한을 작게
    return prompt | get_llm(max_tokens=200) | StrOutputParser()


@lru_cache(maxsize=1)
def get_qa_chain():
    embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    database = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME, embedding=embedding
    )
    retriever = database.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 질의응답 도우미입니다. "
         "아래 검색된 문맥만 사용해서 질문에 답하세요. "
         "모르면 모른다고 답하세요. "
         "한국어로, 최대 세 문장으로 간결하게 답하세요.\n\n"
         "문맥: {context}"),
        ("human", "{question}"),
    ])

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | get_llm(temperature=0.2, max_tokens=400)
        | StrOutputParser()
    )


def get_ai_message(user_message, history=None):
    rewritten = get_dictionary_chain().invoke(
        {"question": user_message, "history": history or []}
    )
    print("=== 재작성된 질문:", rewritten)
    return get_qa_chain().stream(rewritten)
