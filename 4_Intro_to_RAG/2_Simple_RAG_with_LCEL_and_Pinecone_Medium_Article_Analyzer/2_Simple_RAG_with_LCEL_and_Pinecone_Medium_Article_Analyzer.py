# Importing Required Libraries
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

# Loading Environment Variables
load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__=='__main__':
    print('Retrieving Relevant Chunks...')

    # Defining the Embeddings and LLM
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    query = 'What is Pinecone in Machine Learning?'
    chain = PromptTemplate.from_template(template=query) | llm

    # result = chain.invoke(input={})
    # print(result.content)

    # Defining a VectorStore
    vectorstore = PineconeVectorStore(
        index_name=os.environ['INDEX_NAME'], embedding=embeddings
    )

    # Creating custom Template
    template = """
                Use the following pieces of context to answer the question at the end. 
                If you don't know the answer, just say you don't know, don't try to make up an answer.
                Use three sentences maximum and keep the answer as concise as possible.
                Always say "thank you for asking!" at the end of the answer.
                
                {context}
                
                Question : {question}
                
                Helpful Answer: 
    """

    custom_rag_prompt = PromptTemplate.from_template(template)

    # Creating the RAG chain using LCEL
    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs,
         "question":RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    res = rag_chain.invoke(query)
    print(res.content)