from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

DOCUMENT_PATH = "../documents/database_internals.pdf"

# create loader
loader = PyPDFLoader(DOCUMENT_PATH)

# split document
pages = loader.load_and_split()

# embedding function
embedding_func = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# create vector store
vectordb = Chroma.from_documents(
    documents=pages,
    embedding=embedding_func,
    persist_directory=f"../vector_db",
    collection_name="database_internals")

# make vector store persistant
vectordb.persist()