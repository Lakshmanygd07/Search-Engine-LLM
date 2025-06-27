import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import tool
from typing import Optional
from dotenv import load_dotenv

# Optional: load_dotenv() if using .env file locally
# load_dotenv()

## --- Tool Definitions ---

# Arxiv
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# Wikipedia
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# Safe DuckDuckGo Search Tool with Error Handling
@tool
def safe_search(query: str) -> Optional[str]:
    """
    Search the web using DuckDuckGo. Returns search results or error message.
    """
    try:
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query)
    except Exception as e:
        return f"Search failed: {str(e)}"

search = safe_search

## --- Streamlit UI Setup ---

st.title("🔎 LangChain - Chat with Search")

"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

# Sidebar for API Key input
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Session State for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle User Input
if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize LLM and Tools
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        except Exception as e:
            response = f"Agent failed: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
