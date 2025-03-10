## 1. Ingest PDF Files
# 2. Extract Text from PDF Files and split into small chunks
# 3. Send the chunks to the embedding model
# 4. Save the embeddings to a vector database
# 5. Perform similarity search on the vector database to find similar documents
# 6. retrieve the similar documents and present them to the user
## run pip install -r requirements.txt to install the required packages

# general purpose
from pathlib import Path
import typer
# pdf loading
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_core.document_loaders.base import Document
# store embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# retrieval
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
# ollama
import ollama, ollamamanager
import defaults


# defaults
DOC_PATH = 'data/BOI.pdf'
MODEL = 'llama3.2'
EMBEDDINGS_MODEL = 'nomic-embed-text'


def download_ollama_models(model:str=MODEL, embedding:str=EMBEDDINGS_MODEL):
    '''downloads necessary models'''
    print(f'pulling {model = }...')
    ollama.pull(model)
    print(f'pulling {embedding = }...')
    ollama.pull(embedding)


def load_pdf(path:str) -> list[Document]:
    '''loads the specified pdf file and returns it as Document'''
    if Path(path).exists():
        loader = UnstructuredPDFLoader(file_path=path)
        try:
            print(f'loading {path = }...')
            data = loader.load()
        except LookupError:
            # IMHO this should be installable via pip...
            # i cannot add punkt to the dependecies
            print('loading failed, trying to resolve...')
            import nltk
            print(f'downloading punkt_tab...')
            nltk.download('punkt_tab')
            print(f'downloading averaged_perceptron_tagger_eng...')
            nltk.download('averaged_perceptron_tagger_eng')
            print(f'loading {path}...')
            data = loader.load()
        
        return data
    else:
        print(f'file {DOC_PATH} not found')


def split_documents(data:Document) -> list[Document]:
    '''splits the document and returns the processed chunks'''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300
    )
    print('splitting document...')
    chunks = text_splitter.split_documents(data)
    return chunks


def create_vector_db(name:str, chunks:list[Document], embeddings_model:str) -> Chroma:
    '''creates the vector db from the chunks and embeddings llm'''
    print('creating vector db...')
    return Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=embeddings_model),
        collection_name=name
    )

def create_retriever(vector_db:Chroma, llm:BaseChatModel) -> MultiQueryRetriever:
    # a simple technique to generate multiple questions from a single question and then retrieve documents
    # based on those questions, getting the best of both worlds.
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template='''
            You are an AI language model assistant. Your task is to generate five
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}
        ''',
    )
    
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        prompt=QUERY_PROMPT
    )
    return retriever


def create_chain(retriever:MultiQueryRetriever, llm:BaseChatModel):
    '''creates a chat subimitting the question and returnig the answer'''
    # RAG prompt
    template = '''
        Answer the question based ONLY on the following context:
        {context}
        Question: {question}
    '''
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {'context': retriever, 'question': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # response = chain.invoke(input=(question, ))
    # return response
    return chain


# create typer app
app = typer.Typer(add_completion=False)

@app.command()
def main(
    question    :str, 
    doc_path    :str = DOC_PATH, 
    model       :str = MODEL, 
    embeddings  :str = EMBEDDINGS_MODEL,
    host        :str   = defaults.HOST,
    port        :int   = defaults.PORT,
    wait        :float = defaults.WAIT_SECONDS,
    attempts    :int   = defaults.ATTEMPTS,
    stop        :bool  = True
):
    '''typer command representig main'''
    with ollamamanager.OllamaServerCtx(host, port, wait, attempts, stop):
        download_ollama_models()
        
        document = load_pdf(doc_path)
        
        chunks = split_documents(document)
        
        vector_db = create_vector_db('simple-rag', chunks, embeddings)
    
        llm = ChatOllama(model=model)
        
        retriever = create_retriever(vector_db, llm)
        
        chain = create_chain(retriever, llm)
        
        print(f'invoking chain with question: {question}...')
        response = chain.invoke(input=question)
        print(f'\nanswer: {response}\n')
    
    print('done.')

if __name__ == "__main__":
    app()