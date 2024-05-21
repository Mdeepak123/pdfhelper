import os
import streamlit as st
from pinecone import Pinecone
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex, SimpleDirectoryReader, download_loader, ServiceContext, StorageContext
from dotenv import load_dotenv
from llama_index.chat_engine.types import ChatMode
import openai
from llama_index.indices.postprocessor import SentenceEmbeddingOptimizer
from llama_index.vector_stores import PineconeVectorStore 
from llama_index.callbacks import LlamaDebugHandler, CallbackManager
from node_postprocessors.dupilicate_postprocessors import DuplicateRemoverNodePostprocessor

load_dotenv()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ['PINECONE_ENVIRONMENT'] )
openai.api_key =os.getenv("OPENAI_API_KEY")

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager(handlers=[llama_debug])
service_context = ServiceContext.from_defaults(callback_manager=callback_manager)



def load_data():
    #initialize pinecone
        pinecone_index = pc.Index("vector-db")
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store, service_context=service_context
        )


index = load_data()
#initialize chat engine
if "chat-engine" not in st.session_state.keys():
    postprocessor = SentenceEmbeddingOptimizer(
        embed_model=service_context.embed_model,
        percentile_cutoff=0.5, #keep top 50% of related chunks
        threshold_cutoff=0.7, #leave sentances with similarity score above .7
    )

    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT, verbose=True,
        node_postprocessors=[postprocessor, DuplicateRemoverNodePostprocessor()],
    )

st.set_page_config(
    page_title="Chat with Your PDF",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("Chat with Your PDF 💬")

#adding the past messages in a session state

if "messages" not in st.session_state.keys():
    st.session_state['messages'] = [
        {
            "role": "assistant",
            "content": "Ask me a question about what is on your pdf?",
        }
    ]

#user input
if prompt := st.chat_input("Your Question: "):
    st.session_state['messages'].append({"role": "user", "content": prompt})
    

#display past msgs
for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
        st.write(message["content"])

#get response from ai
if st.session_state['messages'][-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(message=prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state['messages'].append(message)

