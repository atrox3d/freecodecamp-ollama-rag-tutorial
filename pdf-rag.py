## 1. Ingest PDF Files
# 2. Extract Text from PDF Files and split into small chunks
# 3. Send the chunks to the embedding model
# 4. Save the embeddings to a vector database
# 5. Perform similarity search on the vector database to find similar documents
# 6. retrieve the similar documents and present them to the user
## run pip install -r requirements.txt to install the required packages
from pathlib import Path
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader


DOC_PATH = 'data/BOI.pdf'
MODEL = 'llama3.2'

if Path(DOC_PATH).exists():
    loader = UnstructuredPDFLoader(file_path=DOC_PATH)
    try:
        data = loader.load()
    except LookupError:
        # IMHO this should be installable via pip...
        # i cannot add punkt to the dependecies
        import nltk
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger_eng')
        data = loader.load()
    print(data)
else:
    print(f'file {DOC_PATH} not found')