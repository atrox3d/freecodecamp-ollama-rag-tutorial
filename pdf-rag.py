## 1. Ingest PDF Files
# 2. Extract Text from PDF Files and split into small chunks
# 3. Send the chunks to the embedding model
# 4. Save the embeddings to a vector database
# 5. Perform similarity search on the vector database to find similar documents
# 6. retrieve the similar documents and present them to the user
## run pip install -r requirements.txt to install the required packages

# general purpose
from pathlib import Path
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
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
# ollama
import ollama, ollamamanager


DOC_PATH = 'data/BOI.pdf'
MODEL = 'llama3.2'
EMBEDDINGS_MODEL = 'nomic-embed-text'


def setup_ollama_models(model:str=MODEL, embedding:str=EMBEDDINGS_MODEL):
    print(f'pulling {model = }...')
    ollama.pull(model)
    print(f'pulling {embedding = }...')
    ollama.pull(embedding)


def load_pdf(path:str) -> list[Document]:
    if Path(path).exists():
        loader = UnstructuredPDFLoader(file_path=path)
        try:
            print(f'loading {path}...')
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
        
        # print(data[0].page_content[:100])
        return data
    else:
        print(f'file {DOC_PATH} not found')


def split_text(data:Document) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300
    )
    print('splitting document...')
    chunks = text_splitter.split_documents(data)
    return chunks
    
    
def create_db(name:str, chunks:list[Document], model:str) -> Chroma:
    print('creating db...')
    return Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=model),
        collection_name=name
    )


def get_rag_answer(question:str, model:str, vector_db:Chroma):
    llm = ChatOllama(model=model)
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
    
    response = chain.invoke(input=(question, ))
    return response


if __name__ == "__main__":
    doc_path = DOC_PATH
    model = MODEL
    embeddings_model = EMBEDDINGS_MODEL
    
    with ollamamanager.OllamaCtx():
        setup_ollama_models()
        document = load_pdf(DOC_PATH)
        # print(document[0].page_content[:100])
        
        chunks = split_text(document)
        # for id, chunk in enumerate(chunks):
            # print(f'{id = }')
            # print('-' * 100)
            # print(f'{chunk.page_content}')
            # print('-' * 100)
        # print(len(chunks))
        vector_db = create_db('simple-rag', chunks, EMBEDDINGS_MODEL)
        question = 'what is this document about?'
        print(f'\nquestion: {question}')
        response = get_rag_answer(
            question,
            MODEL,
            vector_db
        )
        print(f'\nanswer: {response}')
