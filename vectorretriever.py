from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

embeddings = OllamaEmbeddings(model='nomic-embed-text')
pdfloader = PyPDFLoader(file_path='genai-coding-task-overview.pdf')
pdf_document = pdfloader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50,separators=["\n"])
splitted_docs = splitter.split_documents(pdf_document)

vstore = FAISS.from_documents(documents=splitted_docs, embedding=embeddings)
retriever = vstore.as_retriever(search_kwargs={"k": 10})
#print(splitted_docs)
#print(vstore)
#print(retriever.invoke("What is the title"))
query="What is Gen AI"
emb_query = embeddings.embed_query(query)
sub_docs = vstore.similarity_search_with_score(query,k=2)
sub_docs1 = vstore.similarity_search_with_score_by_vector(emb_query,k=2)
print(sub_docs)
print(sub_docs1)