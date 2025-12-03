import streamlit as st
import tempfile
import os

# 1. IMPORTS
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq  # Replaces ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings # Replaces OllamaEmbeddings
# Note: older/newer LangChain releases may not expose the
# `create_tool_calling_agent` / `AgentExecutor` APIs at the same location.
# To avoid import errors across versions, use a simple retriever + LLM
# prompt flow instead of the agent toolbox.

# --- PAGE CONFIG ---
st.set_page_config(page_title="RAG Agent", page_icon="🤖")
st.title("🤖 Cloud RAG Agent (Powered by Groq)")

# --- SIDEBAR: SETUP ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Groq API Key", type="password")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    st.markdown("Get your key at [Groq Console](https://console.groq.com)")

# --- CACHED RESOURCE FUNCTION ---
# We cache this so the DB isn't rebuilt on every interaction
@st.cache_resource
def setup_vector_db(file_path):
    st.info("Processing PDF... this runs on cloud CPU.")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Use a lightweight embedding model that fits in free cloud RAM (80MB)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="local_rag"
    )
    return vector_db

# --- MAIN APP LOGIC ---
if uploaded_file and api_key:
    # 1. Save uploaded file temporarily (Cloud apps can't access local C:/ drives)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # 2. Setup Vector DB
    try:
        vector_db = setup_vector_db(tmp_path)
        retriever = vector_db.as_retriever()
        st.success("PDF Loaded & Embedded!")
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        st.stop()

    # 3. Define a simple lookup function
    def lookup_policy(query: str) -> str:
        """Useful for finding specific information in the PDF."""
        # prefer the common retriever method name used by LangChain
        if hasattr(retriever, "get_relevant_documents"):
            results = retriever.get_relevant_documents(query)
        elif hasattr(retriever, "retrieve"):
            results = retriever.retrieve(query)
        else:
            # fallback: try calling retriever as a callable
            try:
                results = retriever(query)
            except Exception:
                results = []
        return "\n\n".join([doc.page_content for doc in results])

    # 4. Initialize LLM (Mistral via Groq)
    # Groq hosts the open-source Mistral model for you
    llm = ChatGroq(
        temperature=0, 
        model_name="mixtral-8x7b-32768", 
        groq_api_key=api_key
    )

    # 5. Helper to call the LLM robustly across LangChain versions
    def call_llm(model, prompt_text: str) -> str:
        """Try common call methods and return a text string."""
        try:
            # common high-level convenience method
            if hasattr(model, "predict"):
                return model.predict(prompt_text)

            # some LLMs implement __call__ to return text
            if callable(model):
                try:
                    result = model(prompt_text)
                    # sometimes the __call__ returns a dict-like object
                    if isinstance(result, str):
                        return result
                    try:
                        return str(result)
                    except Exception:
                        return repr(result)
                except Exception:
                    pass

            # try `generate` path used by LangChain LLM wrappers
            if hasattr(model, "generate"):
                try:
                    gen = model.generate([prompt_text])
                    # best-effort: extract text if present
                    if hasattr(gen, "generations"):
                        try:
                            return gen.generations[0][0].text
                        except Exception:
                            return str(gen)
                    return str(gen)
                except Exception:
                    pass

        except Exception as e:
            return f"Error calling LLM: {e}"
        return "(LLM did not return output)"

    # 6. Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask a question about your PDF..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # 1) retrieve relevant docs
                if hasattr(retriever, "get_relevant_documents"):
                    docs = retriever.get_relevant_documents(user_input)
                elif hasattr(retriever, "retrieve"):
                    docs = retriever.retrieve(user_input)
                else:
                    try:
                        docs = retriever(user_input)
                    except Exception:
                        docs = []

                context = "\n\n".join([d.page_content for d in docs[:5]])

                system_prompt = (
                    "You are a helpful assistant. Use the following context from the PDF to answer the question."
                )
                full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {user_input}\n\nAnswer:"

                response_text = call_llm(llm, full_prompt)
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

elif not uploaded_file:
    st.info("Please upload a PDF to begin.")