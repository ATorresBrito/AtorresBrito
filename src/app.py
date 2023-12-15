import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, CohereEmbeddings
from langchain.llms.cohere import Cohere
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
import transformers
import torch









def get_pdf_text():
    loader = PyPDFLoader("data\SRD_CC_v5.1_ES.pdf")
    data = loader.load()
    print("get_pdf_text data: " + data[4].page_content[:300])
    print("get_pdf_text data: " + str(type(data)))
    print("get_pdf_text data: " + str(len(data)))
    print("get_pdf_text data: " + str(data[4].metadata))
    
    return data

def get_text_chunks(data):
    text_splitter = RecursiveCharacterTextSplitter(
        #OpenAI chunk_size = 1000, chunk_overlap = 200,
        #Huggenface chunk_size = 700, chunk_overlap = 200,
        #Cohere chunk_size = 600, chunk_overlap = 65,
   
        chunk_size = 1000, #Size of the chunk based on the length function (len() in this case) 1000 600 700
        chunk_overlap = 200, #The "connection" between chunks, 200 character from the last chunk are present in the new one 200 65 100
        length_function = len
        )
    chunks = text_splitter.split_documents(data)     
    print("get_text_chunks chunks: " + str(chunks))
    return chunks


def get_embeddings(embedding_model, clave_api_embe):
    
    if embedding_model == "OpenAI":
        os.environ["OPENAI_API_KEY"] = clave_api_embe
        embeddings = OpenAIEmbeddings()
        print("embedding_model : " +embedding_model + "clave_api_embe: " + clave_api_embe)
        
    elif embedding_model == "HuggingFace":
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = clave_api_embe
        #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", model_kwargs = {"device" : "cuda"})
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large", model_kwargs = {"device" : "cuda"})
        print("embedding_model : " +embedding_model + "clave_api_embe: " + clave_api_embe)                                   

    elif embedding_model == "Cohere":
        os.environ["COHERE_API_KEY"] = clave_api_embe
        embeddings = CohereEmbeddings(model="multilingual-22-12")
        #embeddings = CohereEmbeddings(model = "embed-multilingual-v3.0")
        print("embedding_model : " +embedding_model + "clave_api_embe: " + clave_api_embe)
    else:
        raise ValueError("Elección del modelo del lenguaje no válida. Opciones admitidas: 'OpenAI','HuggingFace' y 'Cohere'.")

    print("get_embeddings embedding_model: " + str(embedding_model))
    print("get_embeddings embeddings: " + str(embeddings))
    return embeddings


def get_vectorstore(text_chunks, embeddings):  

    vectorstore=Chroma.from_documents(text_chunks, embeddings)

    print("get_vectorstore text_chunks: " + str(text_chunks))
    print("get_vectorstore embeddings: " + str(embeddings))
    print("get_vectorstore embeddings: " + str(vectorstore))
    return vectorstore


def get_conversation_chain(knowledge_base, conversational_model, model_temperature, clave_api_con):

    if conversational_model == "OpenAI":
        os.environ["OPENAI_API_KEY"] = clave_api_con      
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature= model_temperature)
        print("el valor de temperature es: " + str(model_temperature) )

    elif conversational_model == "HuggingFace":
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = clave_api_con       
        print("Finalizado tokinizer")
        #model=AutoModelForCausalLM.from_pretrained(model_id)
        print("Finalizado model")
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":model_temperature, "max_length":512})                        

    elif conversational_model == "Cohere":
         os.environ["COHERE_API_KEY"] = clave_api_con   
         llm=Cohere(model="command", temperature=model_temperature)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=knowledge_base.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
    retriever=knowledge_base.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents("")

    print("get_conversation_chain knowledge_base: " + str(knowledge_base))
    print("get_conversation_chain conversational_model: " + str(conversational_model))
    print("get_conversation_chain model_temperature: " + str(model_temperature) )
    print("get_conversation_chain conversation_chain: " + str(conversation_chain))
    print("get_conversation_chain retriever: " + str(knowledge_base.as_retriever(search_kwargs={"k": 3})))
    
    print("get_conversation_chain docs_retriever: " + str(docs))
    return conversation_chain



def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    print("handle_userinput response: " + str(response))
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            print("handle_userinput message: " + str(message))
            
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            print("handle_userinput message: " + str(message))



def main():

    st.set_page_config('preguntaDMBOT')
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.image('image\dnd_logo.jpg', width=300)
   
    user_question = st.text_input("Realiza una pregunta al Dungeon Master Bot:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        
        st.subheader("Dungeon Master Bot")

        st.write("Tutorial: ")
        st.write("1) Seleccione Incrustación.\n2) Seleccione Conversación.\n3) Añadir clave Conversación\n4) Botón Ejecutar")
     
        
        # Choose the embedding model
        embedding_model = st.radio("Elija el modelo de incrustación (Embedding)", ("OpenAI", "HuggingFace", "Cohere"))

        if embedding_model == "OpenAI":
            clave_api_embe = st.text_input('OpenAI API Key embedding', type='password')
            print("clave_api_embe : " +clave_api_embe + "embedding_model: " + embedding_model)
       
        elif embedding_model == "HuggingFace":
            clave_api_embe = st.text_input('HuggingFaceHub API Token enbedding', type='password')
            print("clave_api_embe : " +clave_api_embe + "embedding_model: " + embedding_model)
        
        elif embedding_model == "Cohere":
            clave_api_embe = st.text_input('Cohere API Token embedding', type='password')
            print("clave_api_embe : " +clave_api_embe + "embedding_model: " + embedding_model)

        # Choose the conversational model
        conversational_model = st.radio("Elija el modelo conversacional", ("OpenAI", "HuggingFace", "Cohere"))

        if conversational_model == "OpenAI":
            clave_api_con = st.text_input('OpenAI API Key conversacional', type='password')
            print("clave_api_con : " +clave_api_con + "conversational_model: " + conversational_model)
       
        elif conversational_model == "HuggingFace":
            clave_api_con = st.text_input('HuggingFaceHub API Token conversacional', type='password')
            print("clave_api_con : " +clave_api_con + "conversational_model: " + conversational_model)
        
        elif conversational_model == "Cohere":
            clave_api_con = st.text_input('Cohere API Token conversacional', type='password')
            print("clave_api_con : " +clave_api_con + "conversational_model: " + conversational_model)
        

        # Choose the model temperature
        model_temperature = st.slider("Elija la temperatura del modelo", 0.0, 1.0, 0.5, 0.1)

        if st.button("Ejecutar"):
            with st.spinner(text="En proceso..."):
                
                # Obtener PDF            
                raw_text = get_pdf_text()

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # get embeddings
                embedding = get_embeddings(embedding_model, clave_api_embe)

                # Carga a la base de datos vectorial Chroma
                knowledge_base =  get_vectorstore(text_chunks, embedding) 

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    knowledge_base, conversational_model, model_temperature, clave_api_con)
                
            

if __name__ == '__main__':
    main()