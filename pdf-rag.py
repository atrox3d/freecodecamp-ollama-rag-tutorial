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
# ollama
import ollama, ollamamanager


DOC_PATH = 'data/BOI.pdf'
MODEL = 'llama3.2'
EMBEDDINGS_MODEL = 'nomic-embed-text'


def setup_ollama(model:str=MODEL, embedding:str=EMBEDDINGS_MODEL):
    ollamamanager.start_ollama()
    print(f'pulling {model = }...')
    ollama.pull(model)
    print(f'pulling {embedding = }...')
    ollama.pull(embedding)


def teardown_ollama():
    ollamamanager.stop_ollama()


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


if __name__ == "__main__":
    setup_ollama()
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
    print(vector_db)
    teardown_ollama()
    