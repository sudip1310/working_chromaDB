from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
import streamlit as st
 
os.environ["OPENAI_API_KEY"] = "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"
# set up openai api key
openai_api_key = os.environ.get('OPENAI_API_KEY')

persist_directory = 'db'
embedding = OpenAIEmbeddings()

vectordb2 = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding,
                   )

retriever = vectordb2.as_retriever(search_kwargs={"k": 2})
turbo_llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo'
)
# Create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True,
                                  verbose=True)
# Cite sources
def process_llm_response(llm_response):
    print(llm_response['result'])
    st.write(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
# Question

#query="What is the age of the patient?"
query=""
query = st.text_input("Enter Your Query")
if query!="":
    llm_response = qa_chain(query)
    process_llm_response(llm_response)