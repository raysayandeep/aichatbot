from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOllama(model='llama3.2')
embeddings = OllamaEmbeddings(model='nomic-embed-text')
pdfloader = PyPDFLoader(file_path='genai-coding-task-overview.pdf')
pdf_document = pdfloader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50,separators=["\n"])
splitted_docs = splitter.split_documents(pdf_document)

vstore = FAISS.from_documents(documents=splitted_docs, embedding=embeddings)
retriever = vstore.as_retriever(search_type='similarity', search_kwargs={"k": 10})
#prompt = hub.pull("rlm/rag-prompt")
#print(splitted_docs)
#print(vstore)
message = '''
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context only to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
question: {question}
context: {context}
'''
prompt = ChatPromptTemplate([('human',message)])

#print(prompt)
#print(prompt1)
while True:
    query = input("Question: ")
    retrieved_docs = retriever.invoke(query)
    
    messages = prompt.invoke({"question": query, "context": retrieved_docs})
    response = model.invoke(messages)
    print("AI Assistant: ",response.content)