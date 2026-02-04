import streamlit as st 
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper , WikipediaAPIWrapper
from langchain.agents import initialize_agent , AgentType
from langchain_community.tools import ArxivQueryRun ,WikipediaQueryRun , DuckDuckGoSearchRun
from langchain.callbacks import StreamlitCallbackHandler
import os 
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
arxiv_wrapper = ArxivAPIWrapper(top_k_results = 1, doc_content_chars_max = 200 )
arxiv = ArxivQueryRun(api_wrapper = arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results = 1, doc_content_chars_max = 200 )
wiki = WikipediaQueryRun(api_wrapper = api_wrapper)

search = DuckDuckGoSearchRun(name= 'Search')


st.title('Langchain - Chat with Search')

st.sidebar.title('Settings')
api_key = st.sidebar.text_input('Enter your Groq API Key ',type ='password')

if "messages" not in st.session_state:
    st.session_state['messages']= [
        {"role":"assistant ","content" : "Hi ,I am a Chatbot who can search the web . How can I Help You ?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:=st.chat_input(placeholder="What is Machine Learning ?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message('user').write(prompt)

    llm = ChatGroq( groq_api_key=api_key , model_name = "llama-3.1-8b-instant",streaming = True)
    tools = [arxiv , wiki , search ]
    search_agent = initialize_agent(tools , llm , agent  = AgentType.ZERO_SHOT_REACT_DESCRIPTION , handling_parsing_error =True) 
    
    with st.chat_message('assistant'):
        st_cb=StreamlitCallbackHandler(st.container()  ,expand_new_thoughts = False)
        response = search_agent.invoke(
        {"input": st.session_state.messages},
        config={"callbacks": [st_cb]}
        )
        output = response["output"]

        st.session_state.messages.append(
        {"role": "assistant", "content": output}
        )

        st.write(output)