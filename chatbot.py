from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

model = ChatOllama(model='llama3.2')
#model = ChatOllama(model='deepseek-r1')

message_store = []

while True:
    if len(message_store) > 10: #define chat history limit
        message_store = []
    user_message = input('User: ')
    if user_message.lower() in ["quit","q"]:
        print("AI Assistant: Good Bye")
        break
    message_store.append(HumanMessage(user_message))
    response = model.invoke(message_store)
    message_store.append(AIMessage(response.content))
    print('AI Assistant: ',response.content)