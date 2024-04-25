import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

#title
st.title(" RAG System : A context understand on Leave No Context Behind")
st.header("AI For Question & Answers ")

f = open(r"C:\Users\Abinay Rachakonda\Desktop\RAGS\KEY\geminiai_key.txt")
KEY = f.read()

chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a Helpful AI Bot. 
    You take the context and question from user. Your answer should be based on the specific context."""),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Aswer the question based on the given context.
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
])

from langchain_google_genai import ChatGoogleGenerativeAI

chat_model = ChatGoogleGenerativeAI(google_api_key=KEY, 
                                   model="gemini-1.5-pro-latest")

from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
chain = chat_template | chat_model | output_parser
from langchain_community.document_loaders import PDFMinerLoader
dat = PDFMinerLoader(r"C:\Users\Abinay Rachakonda\Desktop\RAGS\2404.07143.pdf")
dat_nik =dat.load()
# Split the document into chunks

from langchain_text_splitters import NLTKTextSplitter

text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)

chunks = text_splitter.split_documents(dat_nik)
# Creating Chunks Embedding
# We are just loading OpenAIEmbeddings

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=KEY, 
                                               model="models/embedding-001")

# vectors = embeddings.embed_documents(chunks)
# Store the chunks in vector store
from langchain_community.vectorstores import Chroma

# Embed each chunk and load it into the vector store
db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")

# Persist the database on drive
db.persist()
# Setting a Connection with the ChromaDB
db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)
# Converting CHROMA db_connection to Retriever Object
retriever = db_connection.as_retriever(search_kwargs={"k": 5})


from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

user_input = st.text_area("Ask Questions to AI")
if st.button("click_for_Answers"):
    st.subheader("Question")
    st.title(user_input)
    response = rag_chain.invoke(user_input)
    st.subheader("Answer:-")
    st.write(response)


