from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

embeddings = OllamaEmbeddings(model='nomic-embed-text')
pdfloader = PyPDFLoader(file_path='genai-coding-task-overview.pdf')
pdf_document = pdfloader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20,separators=["\n"])
splitted_docs = splitter.split_documents(pdf_document)

vstore = FAISS.from_documents(documents=splitted_docs, embedding=embeddings)
retriever = vstore.as_retriever(search_kwargs={"k": 10})
#print(splitted_docs)
#print(vstore)
print(retriever.invoke("What is the title"))