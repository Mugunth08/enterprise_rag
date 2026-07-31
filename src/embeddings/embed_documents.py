from functools import partial

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import get_embeddings_from_gcp


CHROMA_DB_PATH = "./data/chroma_db"

loader = DirectoryLoader(
    path="./data/processed/text",
    glob="*.txt",
    loader_cls=partial(
        TextLoader,
        encoding="utf-8",
    ),
)

documents = loader.load()

print(f"Loaded {len(documents)} documents")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# Create unique IDs
ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]


embedding_model = get_embeddings_from_gcp()


vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    ids=ids,
    persist_directory=CHROMA_DB_PATH,
)

print(f"Stored {len(chunks)} vectors in ChromaDB")
print("Embedding completed successfully!")
print(f"Vector Count: {vector_store._collection.count()}")