import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def ingest_data(filepath: str):
    loader = TextLoader(filepath)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=20, chunk_overlap=0) # Small chunk size forcefully to split each row is a chunk
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_db = Milvus.from_documents(
        docs,
        embeddings,
        collection_name="spotify_reviews",
        connection_args={
            "uri": os.getenv('ZILLIZ_CLOUD_URI'),
            "user": os.getenv('ZILLIZ_CLOUD_USERNAME'),
            "password": os.getenv('ZILLIZ_CLOUD_PASSWORD'),
            # "token": ZILLIZ_CLOUD_API_KEY,  # API key, for serverless clusters which can be used as replacements for user and password
            "secure": True,
        },
    )

    return

if __name__ == '__main__':
    ingest_data('dataset/SPOTIFY_REVIEWS_CLEANED.txt')
