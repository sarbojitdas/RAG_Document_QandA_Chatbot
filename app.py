import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# -------------------------------------------------
# ENV SETUP
# -------------------------------------------------
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# -------------------------------------------------
# LLM
# -------------------------------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.7
)

# -------------------------------------------------
# PROMPT
# -------------------------------------------------
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question strictly based on the provided context.
    If the answer is not present in the context, say "I don't know."

    <context>
    {context}
    </context>

    Question: {input}
    """
)

# -------------------------------------------------
# VECTOR CREATION
# -------------------------------------------------
def create_vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner("Creating vector database..."):
            # Embeddings
            st.session_state.embeddings = OllamaEmbeddings(
                model="nomic-embed-text"
            )

            # Load PDFs
            loader = PyPDFDirectoryLoader("researchpapers")
            docs = loader.load()

            # Split text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            final_documents = text_splitter.split_documents(docs)

            # FAISS
            st.session_state.vectors = FAISS.from_documents(
                final_documents,
                st.session_state.embeddings
            )

# -------------------------------------------------
# QA CHAIN
# -------------------------------------------------
def get_answer(user_query):
    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = st.session_state.vectors.as_retriever(
        search_kwargs={"k": 3}
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    response = retrieval_chain.invoke({"input": user_query})
    return response["answer"]

# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
st.set_page_config(page_title="RAG Q&A with GROQ and LLAMA", layout="wide")

st.title("📄 RAG Q&A with GROQ and LLAMA")
st.write("Ask questions from your uploaded research papers")

user_prompt = st.text_input(
    "Enter your question",
    placeholder="What is the main contribution of the paper?"
)

col1, col2 = st.columns([1, 3])

with col1:
    if st.button("📥 Document Embedding"):
        create_vector_embedding()
        st.success("Vector Database is ready!")

with col2:
    if user_prompt:
        if "vectors" not in st.session_state:
            st.warning("Please create the vector database first.")
        else:
            with st.spinner("Searching documents..."):
                answer = get_answer(user_prompt)

            st.markdown("### ✅ Answer")
            st.write(answer)
