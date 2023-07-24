import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
import streamlit as st

def convert_pdf_to_text(pdf_path):
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text()
        return text_content

def save_text_file(pdf_path, text_content):
    text_filename = os.path.splitext(os.path.basename(pdf_path))[0] + ".txt"
    with open(text_filename, 'w', encoding='utf-8') as text_file:
        text_file.write(text_content)

def convert_and_save_all_pdfs_in_directory(directory_path):
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.pdf'):
            pdf_path = os.path.join(directory_path, file_name)
            text_content = convert_pdf_to_text(pdf_path)
            save_text_file(pdf_path, text_content)

# Usage
input_directory = 'data'
convert_and_save_all_pdfs_in_directory(input_directory)

loader = DirectoryLoader('', glob="./*.txt")
doc = loader.load ( )

from langchain.text_splitter import CharacterTextSplitter
def get_chunk_lst(pdf_text):
    splitter = CharacterTextSplitter(
                separator = ".",
                chunk_size = 200,
                chunk_overlap = 100,
                length_function = len
            )

    chunk_lst = splitter.split_text(pdf_text)
    print(chunk_lst)
    return chunk_lst

text_splitter = RecursiveCharacterTextSplitter (chunk_size=200, chunk_overlap=100)
texts = text_splitter.split_documents(doc)

# Create a new openai api key
os.environ["OPENAI_API_KEY"] = "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"
# set up openai api key
openai_api_key = os.environ.get('OPENAI_API_KEY')

persist_directory = 'son_db'

# OpenAI embeddings
embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory)

vectordb.persist()