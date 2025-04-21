import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import *
from groq import Groq
from pypdf import PdfReader

session_options = ['college_name', 'company_names', 'degree', 'designation',
 'email','mobile_number','name','no_of_pages','skills','total_experience','count']
for i in session_options:
    if i not in st.session_state:
        st.session_state[i] = None

if "session_dict" not in st.session_state:
    st.session_state["session_dict"] = {}

def check_status():
    return st.session_state.get('count', 0) == 1

def parsed_string(file):
    reader=PdfReader(file)
    page_string=""
    for i in range(len(reader.pages)):
        page_string=page_string+"\n"+reader.pages[i]
    return page_string
def call_llm(parsed_string,query):
    query=",".join(query.remove("count"))
    final_query="""You are given with a list of strings which are comma separarted.
    Also you are given witha a parsed string, just extract information for a list of comma sepearated lists from the parsed strings
    if any information not found then use can use 'Information Not Found' as an option to return.
    Also the returened string should be only the values which are comma sepearetd ex: value1,value2,
    So just retun a string which contains corresponding and accurate values which are comma seperated.
    """
    client=Groq(api_key="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")
    chat_completion=client.chat.completions.create(
        messages=[
            {"role":"system",
            "content":"""You are a very Good Resume Parser Software. You have to return the parsed content from the parsed string.
            Note : you are given with a parsed string from the documents and query. You have to extract the information related to the query
            """
            },
            {
                "role":"user",
                "content":f"{final_query} \n Parsed String :{parsed_string} \n Information That Need To Extarct : {query}"
            }
        ],
        model="llama-3.3-70b-versatile"
    )
    return chat_completion.result[0].message.content
def primary_info(file,session_options):
    parsed_string=parseDoc(file)
    try:
        value=call_llm(parsed_string,session_options)
        value=value.split(',')
        for i in range(len(session_options)-1):
            if session_options[i] != 'count':
                st.session_state[session_options[i]]=value[i]
        st.
    except Exception as e:
        st.error(f"You got the following error: {e}")

def insights():
    pass

# File uploader
fileUploader = st.sidebar.file_uploader("Upload Files Of Type PDF, DOCX", type=['pdf', 'docx', 'doc'])

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["About The App", "Primary Info", "Key Insights"],
        icons=["info-circle", "person-badge", "lightbulb"],
        menu_icon="cast",
        default_index=0
    )

# Content handling
if selected == "About The App":
    pass  # Placeholder for future content
elif selected == "Primary Info":
    primary_info(fileUploader)
elif selected == "Key Insights":
    insights()