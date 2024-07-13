import os

# os.system("pip install langchain")
# os.system("pip install langchain-openai")
# os.system("pip install -qU langchain-text-splitters")
# os.system("pip install langchain-chroma")
os.environ["OPENAI_API_KEY"] = "apikey"

import streamlit as st # Import python packages
# from snowflake.snowpark.context import get_active_session
# session = get_active_session() # Get the current credentials
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
import pandas as pd
from googlesearch import search
import requests
from bs4 import BeautifulSoup

pd.set_option("max_colwidth",None)

### Default Values
#model_name = 'mistral-7b' #Default but we allow user to select one
num_chunks = 10 # Num-chunks provided as context. Play with this to check how it affects your accuracy
slide_window = 10 # how many last conversations to remember. This is the slide window.
debug = 1 #Set this to 1 if you want to see what is the text created as summary and sent to get chunks
# use_chat_history = 0 #Use the chat history by default

### Functions
def main():
    global llm
    st.title(f":speech_balloon: Biology Chat Assistant (Synkrino)")
    config_options()
    init_messages()
    llm = ChatOpenAI(temperature=0.5, model_name=st.session_state.model_name, max_tokens=1000)
    urls_input = ask_urls()

    if st.session_state.button_pressed_init:
        boxes = []
        for search_result in st.session_state.search_results:
            boxes.append(st.checkbox(f"[{search_result['title']}]({search_result['url']})"))
        
        button_setup()
        
        if st.session_state.button_pressed:
            st.session_state.urls_input = []
            for i in range(len(boxes)):
                if boxes[i]:
                    st.session_state.urls_input.append(st.session_state.search_results[i])

            st.write("List of documents:")

            if "initialized" not in st.session_state:
                initialize(st.session_state.urls_input)
                st.session_state.initialized = True
            
            if st.session_state.initialized:
                st.dataframe(st.session_state.urls_input)

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
#     urls_input = st.text_area("URLs to use:",
# """https://grants.nih.gov/grants/guide/pa-files/par-19-357.html
# https://grants.nih.gov/grants/guide/pa-files/PAR-21-135.html
# https://grants.nih.gov/grants/guide/rfa-files/RFA-AG-15-014.html
# https://grants.nih.gov/grants/guide/pa-files/PAR-21-130.html
# https://grants.nih.gov/grants/guide/pa-files/PAR-18-026.html"""
#                   )

    if "button_pressed_init" not in st.session_state:
        st.session_state.prompt_init = ""
        st.session_state.button_pressed_init = False
    st.session_state.prompt_init = st.text_area("What are you searching for?", st.session_state.prompt_init)

    if st.button("Search"):
        st.session_state.button_pressed_init = True
    
    
    if st.session_state.button_pressed_init:
        # query = complete_llm("I want sites that talk about Mycobacterium tuberculosis that start with `https://grants.nih.gov/grants/guide/pa-files/`.", 
        #                     "Give a Google search query for the topic specified. Use a Google Dork if needed.")
        if "search_results" not in st.session_state:
            st.session_state.query_init = complete_llm(st.session_state.prompt_init, 
                                "Give a Google search query for the topic specified. Use a Google Dork if needed. Generate nothing else.")
        
        st.write(f"Google Search Query: ```{st.session_state.query_init}```")

        # 
        # st.session_state.search_results = []
        # for url in search(query, num=10, stop=10, pause=2):
        #     try:
        #         response = requests.get(url)
        #         soup = BeautifulSoup(response.content, 'html.parser')
        #         title = soup.title.string if soup.title else 'No Title'
        #         st.session_state.search_results.append({'title': title, 'url': url})
        #     except Exception as e:
        #         print(f"Failed to retrieve {url}: {e}")
        st.session_state.search_results = [
            {'title': "Myeloid-Derived Suppressor Cells (MDSCs) as Potential Therapeutic Targets in TB/HIV (R01 Clinical Trial Not Allowed)", 'url': "https://grants.nih.gov/grants/guide/pa-files/par-19-357.html"},
            {'title': "Development of Psychosocial Therapeutic and Preventive Interventions for Mental Disorders (R61/R33 Clinical Trial Required)", 'url': "https://grants.nih.gov/grants/guide/pa-files/PAR-21-135.html"},
            {'title': "Research Planning Infrastructure to Develop Therapeutic Target-ID Strategies Based on Favorable Genetic Variants of Human Longevity or Health Span (U24)", 'url': "https://grants.nih.gov/grants/guide/rfa-files/RFA-AG-15-014.html"},
            {'title': "Clinical Trials to Test the Effectiveness of Treatment, Preventive, and Services Interventions (R01 Clinical Trial Required)", 'url': "https://grants.nih.gov/grants/guide/pa-files/PAR-21-130.html"},
            {'title': "Phenotypic and Functional Studies on FOXO3 Human Longevity Variants to Inform Potential Therapeutic Target Identification Research (R01 Clinical Trial Optional)", 'url': "https://grants.nih.gov/grants/guide/pa-files/PAR-18-026.html"},
        ]

def button_setup():
    if "button_pressed" not in st.session_state:
        st.session_state.button_pressed = False

    if st.button("Continue"):
        st.session_state.button_pressed = True

def initialize(urls_input=None):
    # global urls, db # Should change this later

    # List of URLs
    if urls_input is None:
        st.session_state.urls = [
            "https://grants.nih.gov/grants/guide/pa-files/par-19-357.html",
            "https://grants.nih.gov/grants/guide/pa-files/PAR-21-135.html",
            "https://grants.nih.gov/grants/guide/rfa-files/RFA-AG-15-014.html",
            "https://grants.nih.gov/grants/guide/pa-files/PAR-21-130.html",
            "https://grants.nih.gov/grants/guide/pa-files/PAR-18-026.html",
        ]
        # st.dataframe(st.session_state.urls)
    else:
        # st.dataframe(urls_input)
        urls_input = [i["url"] for i in urls_input]
        st.session_state.urls = urls_input

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
        for url in st.session_state.urls:
            html_header_splits = html_splitter.split_text_from_url(url)
            splits = text_splitter.split_documents(html_header_splits)
            all_splits.extend(splits)

    with st.spinner(f"Creating vector store..."):
        # Create vector store
        st.session_state.db = Chroma.from_documents(all_splits, OpenAIEmbeddings())

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
    result = llm.invoke(messages).content
    return result

if __name__ == "__main__":
    main()