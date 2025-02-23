from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

embeddings = OllamaEmbeddings(model='nomic-embed-text')
pdfloader = PyPDFLoader(file_path='genai-coding-task-overview.pdf')
pdf_document = pdfloader.load()


vstore = FAISS.from_documents(documents=pdf_document, embedding=embeddings)
print(pdf_document)
print(vstore)