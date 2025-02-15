# app.py

from pathlib import Path

from cycler import V
import streamlit as st
# https://discuss.streamlit.io/t/message-error-about-torch/90886/5
import torch
torch.classes.__path__ = [] # add this line to manually set it to empty. 
#

import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.document_loaders.base import Document
import ollama
import ollamamanager
import tempfile


load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DOC_PATH = "./data/BOI.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./.chroma_db"


def pull_ollama_models(model:str=MODEL_NAME, embedding:str=EMBEDDING_MODEL) -> None:
    '''downloads necessary models'''
    logging.info(f'pulling {model = }...')
    ollama.pull(model)
    logging.info(f'pulling {embedding = }...')
    ollama.pull(embedding)


def load_pdf(doc_path:str) -> list[Document]:
    """Load PDF documents."""
    logging.info(f'checking {doc_path = }')
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        logging.info(f'loading {doc_path = }')
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        st.error("PDF file not found.")
        return None


def split_documents(documents:Document) -> list[Document]:
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


@st.cache_resource
def load_vector_db(doc_path:str, embedding_model:str, vector_store_name:str):
    """Load or create the vector database."""
    # Pull the embedding model if not already available
    # ollama.pull(EMBEDDING_MODEL)

    embedding = OllamaEmbeddings(model=embedding_model)

    # Load and process the PDF document
    data = load_pdf(doc_path)
    if data is None:
        return None

    # Split the documents into chunks
    chunks = split_documents(data)
    
    vector_db = None
    try:
        if os.path.exists(PERSIST_DIRECTORY):
            logging.info(f'loading existing vector db from {PERSIST_DIRECTORY}')
            vector_db = Chroma(
                embedding_function=embedding,
                collection_name=vector_store_name,
                persist_directory=PERSIST_DIRECTORY,
            )
            # wtf? doesn't he load the docs???
            logging.info(f'adding chunks to existing db')
            vector_db.add_documents(chunks)
            #
            logging.info("Loaded existing vector database.")
        else:
            
            logging.info('creating new vector db from chunks')
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embedding,
                collection_name=vector_store_name,
                # persist_directory=PERSIST_DIRECTORY,
            )
            # vector_db.persist()
            logging.info("Vector database created and persisted.")
        return vector_db
    except Exception as e:
        logging.error(e)
        logging.error(f'deleting vector_db variable...')
        del vector_db
        vector_db = None
        logging.error(f'deleted vector_db variable')
        raise


def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever


def create_chain(retriever, llm):
    """Create the chain with preserved syntax."""
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created with preserved syntax.")
    return chain


def rag(
        user_input          :str, 
        document_path       :str, 
        model_name          :str=MODEL_NAME, 
        embedding_model     :str=EMBEDDING_MODEL,
        vector_store_name   :str=VECTOR_STORE_NAME
) -> str:
    with ollamamanager.OllamaServerCtx():
        pull_ollama_models(model=model_name)
        # Initialize the language model
        llm = ChatOllama(model=model_name)
        try:
            # Load the vector database
            vector_db = load_vector_db(document_path, embedding_model, vector_store_name)
            if vector_db is None:
                st.error("Failed to load or create the vector database.")
                return

            # Create the retriever
            retriever = create_retriever(vector_db, llm)

            # Create the chain
            chain = create_chain(retriever, llm)

            # Get the response
            response = chain.invoke(input=user_input)
        finally:
            # fix An error occurred: Could not connect to tenant default_tenant. 
            # Are you sure it exists?
            if vector_db is not None:
                logging.info(f'RAG| deleting Chroma collection {vector_store_name}...')
                vector_db.delete_collection()
                logging.info(f'RAG| deleting vector db...')
                del vector_db
                logging.info(f'RAG| setting vector_db to None...')
                vector_db = None
                logging.info(f'RAG| deletion of Chroma completed')
            else:
                logging.info('RAG| vector_db is None, nothing to do')
        
        return response


def main():
    st.title("Document Assistant")
    pdf = st.file_uploader('pdf', 'pdf')
    # return
    if pdf:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_pdf_path = str(Path(tmpdir, pdf.name))
            logging.info(f'saving pdf contento to tmp path: {tmp_pdf_path}')
            with open(tmp_pdf_path, 'wb') as fp:
                fp.write(pdf.getvalue())

            # User input
            user_input = st.text_input("Enter your question:", "")

            if user_input:
                with st.spinner("Generating response..."):
                    try:
                        response = rag(user_input, tmp_pdf_path)
                        st.markdown("**Assistant:**")
                        st.write(response)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            else:
                st.info("Please enter a question to get started.")
    
    st.session_state


if __name__ == "__main__":
    main()
