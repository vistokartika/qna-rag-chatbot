import os
from typing import Sequence

from dotenv import load_dotenv
from langchain_community.vectorstores import Milvus
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

### Statefully manage chat history ###
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

def retriever_vector_store(collection_name, embeddings, top_k=100):
    vector_store = Milvus(
        embeddings,
        connection_args={
            "uri": os.getenv('ZILLIZ_CLOUD_URI'),
            "user": os.getenv('ZILLIZ_CLOUD_USERNAME'),
            "password": os.getenv('ZILLIZ_CLOUD_PASSWORD'),
            # "token": ZILLIZ_CLOUD_API_KEY,  # API key, for serverless clusters which can be used as replacements for user and password
            "secure": True,
        },
        collection_name=collection_name
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    return retriever

def init_rag_chain_workflow(llm, retriever):
    ### Contextualize question ###
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
        "The context is Google user reviews for Spotify App."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt)

    ### Answer question ###
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know."
        "\n\n"
        "The given context is chat history and Google user reviews for Spotify App."
        "You will receive multiple user reviews for context and each will contains:"
        "1. review_text: The review text which describes the user's experience with the app."
        "2. rating_from_reviewer: The rating which is an integer between 1 and 5. The higher the rating, the more satisfied the user is."
        "3. review_likes: The number of likes the review has received from other users. Usually, the more likes, the more helpful and relevant the review is."
        "4. app_version: The version of the app the user is reviewing."
        "5. review_date: The date the review was posted. Please highly consider this date when answering questions that has period of time mention such as 'latest' or 'recent'."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def call_model(state: State):
        response = rag_chain.invoke(state)
        return {
            "chat_history": [
                HumanMessage(state["input"]),
                AIMessage(response["answer"]),
            ],
            "context": response["context"],
            "answer": response["answer"],
        }
    
    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app

