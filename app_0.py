import os

# os.system("pip install langchain")
# os.system("pip install langchain-openai")
# os.system("pip install -qU langchain-text-splitters")
# os.system("pip install langchain-chroma")
os.environ["OPENAI_API_KEY"] = "sk-proj-7aIVmW3EsY8DWqCUF04MT3BlbkFJSscfejAgB1Xlpe03UuIz"

import streamlit as st # Import python packages
# from snowflake.snowpark.context import get_active_session
# session = get_active_session() # Get the current credentials
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
import pandas as pd

pd.set_option("max_colwidth",None)

### Default Values
#model_name = 'mistral-7b' #Default but we allow user to select one
num_chunks = 10 # Num-chunks provided as context. Play with this to check how it affects your accuracy
slide_window = 10 # how many last conversations to remember. This is the slide window.
debug = 1 #Set this to 1 if you want to see what is the text created as summary and sent to get chunks
# use_chat_history = 0 #Use the chat history by default

### Functions
def main():
    st.title(f":speech_balloon: Biology Chat Assistant (Synkrino)")
    urls_input = ask_urls()

    button_setup()
    
    if st.session_state.button_pressed:
        urls_input = urls_input.split("\n")
        urls_input = [i for i in urls_input if "http" in i]

        st.write("List of documents:")

        config_options()
        init_messages()

        if "initialized" not in st.session_state:
                initialize(urls_input)
                st.session_state.initialized = True

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if question := st.chat_input("Prompt here..."):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()

                question = question.replace("'","")

                with st.spinner(f"Synkrino bot thinking..."):
                    response = complete(question)
                    message_placeholder.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

def ask_urls():
    global valid
    valid = False
    urls_input = st.text_area("URLs to use:",
"""https://grants.nih.gov/grants/guide/pa-files/par-19-357.html
https://grants.nih.gov/grants/guide/pa-files/PAR-21-135.html
https://grants.nih.gov/grants/guide/rfa-files/RFA-AG-15-014.html
https://grants.nih.gov/grants/guide/pa-files/PAR-21-130.html
https://grants.nih.gov/grants/guide/pa-files/PAR-18-026.html"""
                  )
    return urls_input

def button_setup():
    if "button_pressed" not in st.session_state:
        st.session_state.button_pressed = False

    if st.button("Continue"):
        st.session_state.button_pressed = True

def initialize(urls_input=None):
    global urls # Should change this later

    # List of URLs
    if urls_input is None:
        urls = [
            "https://grants.nih.gov/grants/guide/pa-files/par-19-357.html",
            "https://grants.nih.gov/grants/guide/pa-files/PAR-21-135.html",
            "https://grants.nih.gov/grants/guide/rfa-files/RFA-AG-15-014.html",
            "https://grants.nih.gov/grants/guide/pa-files/PAR-21-130.html",
            "https://grants.nih.gov/grants/guide/pa-files/PAR-18-026.html",
        ]
    else:
        urls = urls_input

    st.dataframe(urls)

    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
        ("h4", "Header 4"),
    ]

    # Initialize HTML header text splitter
    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # Splitting parameters
    chunk_size = 500
    chunk_overlap = 30

    # Initialize recursive character text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    with st.spinner(f"Fetching URLs and splitting HTML text..."):
        # Create splits
        all_splits = []
        for url in urls:
            html_header_splits = html_splitter.split_text_from_url(url)
            splits = text_splitter.split_documents(html_header_splits)
            all_splits.extend(splits)

    with st.spinner(f"Creating vector store and LLM..."):
        # Create vector store and LLM
        st.session_state.db = Chroma.from_documents(all_splits, OpenAIEmbeddings())
        st.session_state.llm = ChatOpenAI(temperature=0.5, model_name=st.session_state.model_name, max_tokens=1000)

def config_options():
    st.sidebar.selectbox('Select your model:',(
                                    # 'mixtral-8x7b',
                                    # 'snowflake-arctic',
                                    # 'mistral-large',
                                    # 'llama3-8b',
                                    # 'llama3-70b',
                                    # 'reka-flash',
                                    #  'mistral-7b',
                                    #  'llama2-70b-chat',
                                    #  'gemma-7b'
                                    'gpt-3.5-turbo',
                                    'gpt-4o',
                                     ), key="model_name")

    # For educational purposes. Users can chech the difference when using memory or not
    st.sidebar.checkbox('Do you want that I remember the chat history?', key="use_chat_history", value = True)

    st.sidebar.checkbox('Debug: Click to see summary generated of previous conversation', key="debug", value = True)
    st.sidebar.button("Start Over", key="clear_conversation")
    st.sidebar.expander("Session State").write(st.session_state)


def init_messages():
    if st.session_state.clear_conversation or "messages" not in st.session_state:
        st.session_state.messages = []

def get_similar_chunks (question):
    # similar_chunks = st.session_state.db.similarity_search(question, k=num_chunks)
    similar_chunks = st.session_state.db.similarity_search_by_vector(OpenAIEmbeddings().embed_query(question), k=num_chunks)
    if st.session_state.debug:
        st.sidebar.text("Similar chunks found: ")
        st.sidebar.caption(similar_chunks)
    return similar_chunks

def get_chat_history():
    chat_history = []
    start_index = max(0, len(st.session_state.messages) - slide_window)
    for i in range (start_index , len(st.session_state.messages) -1):
         chat_history.append(st.session_state.messages[i])
    return chat_history

def summarize_question_with_history(chat_history, question):
    prompt = f"""
        Based on the chat history below and the question, generate a query that extend the question
        with the chat history provided. The query should be in natual language.
        Answer with only the query. Do not add any explanation.

        <chat_history>
        {chat_history}
        </chat_history>
        <question>
        {question}
        </question>
        """
    result = complete_llm(prompt)
    if st.session_state.debug:
        st.sidebar.text("Summary to be used to find similar chunks in the docs:")
        st.sidebar.caption(result)
    return result

def create_prompt (myquestion):

    if st.session_state.use_chat_history:
        chat_history = get_chat_history()

        if chat_history != []: #There is chat_history, so not first question
            question_summary = summarize_question_with_history(chat_history, myquestion)
            prompt_context =  get_similar_chunks(question_summary)
        else:
            prompt_context = get_similar_chunks(myquestion) #First question when using history
    else:
        prompt_context = get_similar_chunks(myquestion)
        chat_history = ""

    prompt = f"""
           You are an expert chat assistance that extracs information from the CONTEXT provided
           between <context> and </context> tags.
           You offer a chat experience considering the information included in the CHAT HISTORY
           provided between <chat_history> and </chat_history> tags..
           When ansering the question contained between <question> and </question> tags
           be concise and do not hallucinate.
           If you don´t have the information just say so.

           Do not mention the CONTEXT used in your answer.
           Do not mention the CHAT HISTORY used in your asnwer.

           <chat_history>
           {chat_history}
           </chat_history>
           <context>
           {prompt_context}
           </context>
           <question>
           {myquestion}
           </question>
           Answer:
           """

    return prompt

def complete(myquestion):
    prompt = create_prompt(myquestion)
    result = complete_llm(prompt)
    return result

def complete_llm(prompt, system_prompt=None):
    if system_prompt is None:
        system_prompt = "You are a helpful assistant that gives information about biology."
    messages = [
        ("system", system_prompt),
        ("human", prompt),
    ]
    result = st.session_state.llm.invoke(messages).content
    result = result.replace("'", "")
    return result

if __name__ == "__main__":
    main()