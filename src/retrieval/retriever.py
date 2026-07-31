from langchain_chroma import Chroma

from utils import get_embeddings_from_gcp


CHROMA_DB_PATH = "./data/chroma_db"

embedding_model = get_embeddings_from_gcp()

vector_store = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embedding_model,
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

