import streamlit as st
import time
import openai
from openai import OpenAI
import langchain
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.chat import (SystemMessagePromptTemplate, HumanMessagePromptTemplate,
                                    ChatPromptTemplate)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import os

# Step 1: Load environment variables

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.7,max_completion_tokens=100)

system_template = """Answer the questions based on the provided context only.
                    Please provide the most accurate respone based on the question"""

human_template = """<context>
                    {context}
                    <context>
                    Question:{input}"""

system_message_template = SystemMessagePromptTemplate.from_template(system_template)
human_messgae_template = HumanMessagePromptTemplate.from_template(human_template)
prompt = ChatPromptTemplate.from_messages([system_message_template,human_messgae_template])

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceBgeEmbeddings(model_name='all-MiniLM-L6-v2')
        st.session_state.loader = PyPDFDirectoryLoader("material")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = Chroma.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings,
            persist_directory="chroma_db2"
        )
        st.session_state.vectors.persist()

st.title("RAG Document Q&A With OpenAi And Hugging Face Embedding")

user_prompt=st.text_input("Enter your Questions")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")


if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever(search_type = 'mmr',
                                    search_Kwargs = {'k' : 3,
                                                     'lambda_mult' : 0.7})
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retrieval_chain.invoke({'input':user_prompt})
    print(f"Response time :{time.process_time()-start}")

    st.write(response['answer'])

    ## With a streamlit expander
    with st.expander("Document similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')
