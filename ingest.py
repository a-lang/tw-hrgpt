from langchain_community.document_loaders import UnstructuredMarkdownLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
from embedding_func import embed_hf, embed_cohere
import os
import shutil

DATA_PATH = "docs"
DB_PATH = "vector_db"
#embed_model = embed_cohere()
embed_model = embed_hf()

def load_files(dir=DATA_PATH):
    pages = []
    path = os.path.join(os.getcwd(), dir)

    for filename in os.listdir(path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path + '/' + filename)
            pages.extend(loader.load())
        elif filename.endswith(".txt"):
            loader = TextLoader(path + '/' + filename)
            pages.extend(loader.load())
        elif filename.endswith(".md"):
            loader = UnstructuredMarkdownLoader(path + '/' + filename)
            pages.extend(loader.load())
        else:
            print(f"Unsupported file type: {filename}")
        
    return pages

def load_files2(dir=DATA_PATH):
    docs = []
    path = os.path.join(os.getcwd(), dir)

    for filename in os.listdir(path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path + '/' + filename)
            pdf_pages = loader.load()
            doc = merge_pages_contenct(pdf_pages)
            docs.extend(doc)
        elif filename.endswith(".txt"):
            loader = TextLoader(path + '/' + filename)
            docs.extend(loader.load())
        elif filename.endswith(".md"):
            loader = UnstructuredMarkdownLoader(path + '/' + filename)
            docs.extend(loader.load())
        else:
            print(f"Unsupported file type: {filename}")
        
    return docs

def merge_pages_contenct(pages):
    merged_content = ""
    doc = []
    for page in pages:
        merged_content += page.page_content + "\n"

    doc = yield Document(page_content=merged_content, metadata={'source': page.metadata['source']})
    return doc

def split_text(docs: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        #chunk_size=1000, chunk_overlap=200, 
        chunk_size=500, chunk_overlap=125, 
        add_start_index=True,
        separators=[
            "第 .+ 章 .+\n\n",
            "第 [123456789-]+ 條\n\n",
            "\uff0e",  # Fullwidth full stop "．"
            "\u3002",  # Ideographic full stop "。"
            "\uff0c",  # Fullwidth comma "，"
        ],
        is_separator_regex=True,
    )
    return text_splitter.split_documents(docs)

def save_to_chroma(chunks: list[Document]):
    persist_directory = DB_PATH
    clear_database(persist_directory)
    db = Chroma.from_documents(collection_name="full_docs", documents=chunks, embedding=embed_model, persist_directory=persist_directory)

    # Calculate Page IDs.
    # Added the Title into the meta-data
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks: list[Document]):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

        # Add Title into the meta-data
        #title=os.path.splitext(os.path.basename(source))[0]
        #chunk.metadata["title"] = title

    return chunks

def clear_database(db_path):
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

def main():
    #pages = load_files()
    #chunks = split_text(pages)
    #print(f"Loaded Pages: {len(pages)}")
    docs = load_files2()
    chunks = split_text(docs)
    print(f"Loaded Docs: {len(docs)}")

    print(f"Loaded Chunks: {len(chunks)}")
    save_to_chroma(chunks)


if __name__ == "__main__":
    main()