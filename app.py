import os
from config import OPENAI_API_KEY
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from warnings import filterwarnings

filterwarnings("ignore", category=DeprecationWarning)
from fastapi import FastAPI

app = FastAPI()


txt_file_path = "cb.txt"
loader = TextLoader(file_path=txt_file_path, encoding="utf-8")
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
data = text_splitter.split_documents(data)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
print("OPENAI_API_KEY has been set!")


embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(data, embedding=embeddings)


llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(), memory=memory
)

vectorstore_path = "vectorstore_index"

if os.path.exists(vectorstore_path):
    print("Loading existing vectorstore...")
    vectorstore = FAISS.load_local(
        folder_path=vectorstore_path,
        embeddings=OpenAIEmbeddings(),
        index_name="faiss_index",
        allow_dangerous_deserialization=True,
    )

else:
    print("Creating new vectorstore and saving to disk...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(data, embedding=embeddings)
    vectorstore.save_local(vectorstore_path, index_name="faiss_index")


@app.post("/api-llm_rag/")
def llm_rag(query):
    result = conversation_chain({"question": query})
    return {"answer": result["answer"]}


# query = "What is the topic of the document?"
# docs = vectorstore.similarity_search(query, k=3)  # k = number of top matches
# for doc in docs:
#     print(doc.page_content)
