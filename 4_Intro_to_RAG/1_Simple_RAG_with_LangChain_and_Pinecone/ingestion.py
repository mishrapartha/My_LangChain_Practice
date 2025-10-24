# Importing Required Libraries
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


# Loading Environment Variables
load_dotenv()

if __name__ == '__main__':
    print('Ingesting/Loading the documents ...')
    loader = TextLoader("sample_mediumblog.txt", encoding='utf8')
    document = loader.load()

    print('Splitting into Chunks...')
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f'Created {len(texts)} chunks')

    # Creating the OpenAI Embeddings
    embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))

    print('Ingesting the Chunks...')
    PineconeVectorStore.from_documents(
        texts,embeddings,index_name=os.environ['INDEX_NAME']
    )
    print('Ingestion Process Finished and saved under the index in Pinecone')


