from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

embeddings = OllamaEmbeddings(model='nomic-embed-text')

document_list =[]
sampletext  = 'This is a smple text'
langhchain_document = Document(
    page_content=sampletext,
    metadata={'source':'text'}
)
document_list.append(langhchain_document)
result = embeddings.embed_query(sampletext)
#result = embeddings.embed_documents(sampletext)


print(result[:5])
#print(document_list)