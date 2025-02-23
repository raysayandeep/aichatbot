#export GROQ_API_KEY=<groq-api-key> linux
#set GROQ_API_KEY=<groq-api-key> Windows

from langchain_groq import ChatGroq

model = ChatGroq(model="llama-3.1-8b-instant")