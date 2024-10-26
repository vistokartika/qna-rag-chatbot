import uuid

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.app.rag_chatbot import retriever_vector_store, init_rag_chain_workflow
from src.app.evaluator import helpfulness_evaluator

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.35)

retriever = retriever_vector_store("spotify_reviews", embeddings, 320)
app = init_rag_chain_workflow(llm, retriever)

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

st.title("💬 Chatbot")

example_questions = [
    "What are the specific features or aspects that users appreciate the most in our application?",
    "In comparison to our application, which music streaming platform are users most likely to compare ours with?",
    "What are the primary reasons users express dissatisfaction with Spotify?",
    "Can you identify emerging trends or patterns in recent user reviews that may impact our product strategy?",
    "What can we do as a company to solve it?",
]

st.sidebar.title("Example Questions")
for q in example_questions:
    st.sidebar.markdown(f"- {q}")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Call the model
    response = result = app.invoke(
        {"input": prompt},
        config=config,
    )
    msg = response["answer"]

    # Evaluate the response
    score = helpfulness_evaluator.evaluate_strings(
        prediction=msg,
        input=prompt,
    )
    msg += f"\n\nResponse score (Helpfulness): {float(score['score']) / 10.0}"
    
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)