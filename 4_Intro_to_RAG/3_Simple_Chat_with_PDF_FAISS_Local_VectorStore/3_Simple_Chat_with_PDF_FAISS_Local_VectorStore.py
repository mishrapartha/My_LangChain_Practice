# Importing Required Libraries
import os

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# Loading Environment Variables
load_dotenv()

if __name__ == "__main__":
    print("Chat with PDF...")
    pdf_path = "react.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()

    # Chunking the data and storing it as documents
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)

    # Defining the Embeddings
    embeddings = OpenAIEmbeddings()

    # Defining a VectorStore and saving a copy in the local machine
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    vectorstore.save_local("faiss_index_react")

    # Loading from the local vectorstore
    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings,
        allow_dangerous_deserialization=True
    )  # 'allow_dangerous_deserialization=True' this is not recommended in production due to security issues

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        OpenAI(), retrieval_qa_chat_prompt
    )

    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(),combine_docs_chain
    )

    res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})

    print(f"Given Question is :\t {res['input']}\n")
    print(f"Answer: {res['answer']}")





