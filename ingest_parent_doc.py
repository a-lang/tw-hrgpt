from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
#from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from embedding_func import embed_hf
import os
import shutil

DATA_PATH = "docs/pdf"
DB_PATH = "vector_db"
DOC_NAME = "parent_documents"
DOC_FILE = "docstore.pkl"
DOC_KEY = "doc_id"

embed_model = embed_hf()

def load_pdfs(dir=DATA_PATH):
    loader = PyPDFDirectoryLoader(dir)
    return loader.load()

def split_text(docs: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        #chunk_size=1000, chunk_overlap=200, 
        chunk_size=500, chunk_overlap=125, 
        add_start_index=True
    )
    return text_splitter.split_documents(docs)

def save_to_pickle(obj, filename):
    import pickle
    with open(filename, "wb") as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

def load_from_pickle(filename):
    import pickle
    with open(filename, "rb") as file:
        return pickle.load(file)

def create_parent_chroma(docs):
    from langchain.retrievers import ParentDocumentRetriever
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.storage import InMemoryStore
    import uuid

    # This text splitter is used to create the child documents
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    # The vectorstore to use to index the child chunks
    db = Chroma(
        collection_name=DOC_NAME, embedding_function=embed_model, persist_directory=DB_PATH
    )

    # The storage layer for the parent documents
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=db,
        docstore=store,
        child_splitter=child_splitter,
        id_key=DOC_KEY,
    )

    docstore_path = os.path.join(DB_PATH, DOC_FILE)
    #doc_ids = [str(uuid.uuid4()) for _ in docs]
    retriever.add_documents(docs, ids=None)
    save_to_pickle(retriever.docstore.store, docstore_path)

    print(f"Ingested Documents: {len(retriever.docstore.store)}")

def clear_database(db_path=DB_PATH):
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

def main():
    docs = load_pdfs()
    clear_database()
    create_parent_chroma(docs)

if __name__ == "__main__":
    main()