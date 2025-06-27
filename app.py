import streamlit as st  # Imports Streamlit for creating the web UI
from langchain_groq import ChatGroq  # Imports ChatGroq LLM wrapper from LangChain-Groq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper  # Imports wrappers for Arxiv and Wikipedia
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun  # Imports query tools
from langchain.agents import initialize_agent, AgentType  # Imports agent initializer and agent type
from langchain.callbacks import StreamlitCallbackHandler  # Callback handler for displaying intermediate steps in Streamlit
import os  # OS module for environment access (not used in current script)
from dotenv import load_dotenv  # Loads environment variables from a .env file

## Arxiv and Wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)  # Initialize Arxiv wrapper with top 1 result, 200 char max
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)  # Creates Arxiv query tool using the wrapper

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)  # Initialize Wikipedia wrapper similarly
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)  # Creates Wikipedia query tool

search = DuckDuckGoSearchRun(name="Search")  # Creates DuckDuckGo search tool

st.title("🔎 LangChain - Chat with search")  # Sets the Streamlit app title
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""  # Description shown below the title

## Sidebar for settings
st.sidebar.title("Settings")  # Title in the sidebar
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")  # API key input (password hidden)

if "messages" not in st.session_state:  # If chat history isn't in session state
    st.session_state["messages"] = [  # Initialize chat history with a welcome message
        {"role": "assisstant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:  # Loop through each message in session
    st.chat_message(msg["role"]).write(msg['content'])  # Display chat messages by role

if prompt := st.chat_input(placeholder="What is machine learning?"):  # If the user types a message
    st.session_state.messages.append({"role": "user", "content": prompt})  # Add user message to chat history
    st.chat_message("user").write(prompt)  # Display user message

    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)  # Initialize the LLM with API key
    tools = [search, arxiv, wiki]  # Define list of tools the agent can use

    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)  # Initialize the agent with tools and LLM

    with st.chat_message("assistant"):  # Display assistant response
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)  # Setup callback to show intermediate steps
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])  # Run agent on chat history and capture response
        st.session_state.messages.append({'role': 'assistant', "content": response})  # Save assistant response to chat history
        st.write(response)  # Display the assistant's response
