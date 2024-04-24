import os
import re
import json
from bs4 import BeautifulSoup
import streamlit as st 
from streamlit_pills import pills
from operator import itemgetter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain.schema.output_parser import StrOutputParser


# from prompts import modules  # Importing necessary modules and packages


def reset():
    """Reset session state variables."""
    # Resetting session state variables if they exist
    if 'responses' in st.session_state:
        del st.session_state['responses']
    if 'requests' in st.session_state:
        del st.session_state['requests']
    if 'drug_name' in st.session_state:
        del st.session_state['drug_name']
    if 'req_mod' in st.session_state:
        del st.session_state['req_mod']
    if 'vector_store' in st.session_state:
        del st.session_state['vector_store']
    if 'load_data' in st.session_state:
        del st.session_state['load_data']
    if 'refined_query' in st.session_state:
        del st.session_state['refined_query']
    if 'subparts' in st.session_state:
        del st.session_state['subparts']
    if 'flag' in st.session_state:
        del st.session_state.flag

def set_load_true():
    """Set 'load_data' session state variable to True."""
    st.session_state.load_data = True

def split_text_into_chunks(data, chunk_size=256, chunk_overlap=20):
    """
    Split input text into smaller chunks for processing.

    Args:
        data (str): The input text data to be split.
        chunk_size (int, optional): The size of each chunk. Defaults to 256.
        chunk_overlap (int, optional): The overlap between chunks. Defaults to 20.

    Returns:
        list: A list of text chunks.
    """
    # Initialize text splitter with provided parameters
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # Split text into chunks using the text splitter
    chunks = text_splitter.split_text(data)
    return chunks

def create_embeddings(chunks):
    """
    Create embeddings for text chunks and generate a vector store.

    Args:
        chunks (list): The list of text chunks.

    Returns:
        Chroma: A vector store containing embeddings of text chunks.
    """
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    # Generate embeddings for each chunk and create a vector store
    vector_store = Chroma.from_texts(chunks, embeddings)
    return vector_store

def get_conversation_vector(data):
    """
    Generate a conversation vector from input text data.

    Args:
        data (str): The input text data representing a conversation.

    Returns:
        Chroma: A vector store containing embeddings of the conversation.
    """
    # Split the input text into manageable chunks
    chunks = split_text_into_chunks(data)
    # Create embeddings and generate a conversation vector
    vector_store = create_embeddings(chunks)
    return vector_store

@st.cache_resource(show_spinner=False)
def get_elastic_vector():
    embeddings = OpenAIEmbeddings()
    vector_store = ElasticsearchStore(
        embedding=embeddings,
        index_name="ctd-prompts-index",
        es_cloud_id="CTD:YXAtc291dGgtMS5hd3MuZWxhc3RpYy1jbG91ZC5jb20kNmI2YmRhMzIxMjZlNDYyZmI3MzE0YzlhOTRiODYwOGEkZGYwY2E5MzhhMTkwNDJiN2IwYzcyYzM2ZmIzMDY3ZDE=",
        es_user = "elastic",
        es_password = "h3WtQii3kldV1y6aCvExdknO"
    )
    return vector_store

@st.cache_resource(show_spinner=False)
def get_qdrant_vector():
    embeddings = OpenAIEmbeddings()
        
    # Provide qdrant url and api key",
    url = "https://b7cbaa0e-63c3-4095-bf41-2e378eca4c9d.us-east4-0.gcp.cloud.qdrant.io:6333"
    qdrant_api_key = "i8M8w3dQe_aZ5tH1x85EfHn9mO04LgwnADr12b48h2vnSR4CLUclfw"
    
    # Define qdrant client",
    client = QdrantClient(
        url=url,
        api_key=qdrant_api_key,
    )
    # Connect with qdrant collection\n",
    vector_store = Qdrant(client, collection_name="ctd-prompts-index", embeddings=embeddings)
    return vector_store

def get_refined_query(query, context):
    """
    Refine the user query based on context.

    This function analyzes the user query and context to generate prompts
    that are relevant and detailed to create the required subparts.

    Args:
        query (str): The user query.
        context (list): The context representing subparts of a section.

    Returns:
        dict: A dictionary mapping generated prompts to their respective subparts.
    """
    # Template for refining query
    temp = f''' 
            Generate refined prompts for creating sections of a CTD (Common Technical Document) report for a drug. The input 
            will be a basic prompt provided by the user, which needs optimization and expansion to generate effective prompts for various subsections of the CTD report.

            The subsections required for the CTD report section will be provided as context, listing all the necessary subsections. 
            The goal is to optimize the user-provided prompt and generate 3-5 detailed prompts relevant to the provided subsections.

            Consider the context of subsections provided, ensuring that the prompts cover each subsection comprehensively. Optimize 
            the prompts to include specific details such as study designs, patient demographics, treatment protocols, efficacy endpoints, 
            safety profiles, and any relevant regulatory requirements.

            Provide prompts that encourage detailed and informative responses, guiding the generation of content suitable for inclusion in 
            the respective subsections of the CTD report.

            Additionally, ensure that the generated prompts adhere to the guidelines and standards for CTD report documentation, maintaining clarity, 
            accuracy, and compliance with regulatory requirements.
            
            Given the following user query and CTD section Subparts. 

            Query provided by user: {query}
            
            Subparts: {str(context)}''' + '''\n

            Your task is quite complex, Follow the given steps to provide the expected output in json format.
            1. Analyse the provided query and Generate few prompts which are most relevant and detailed to create the required subparts.
            2. Now from these generated prompts, find out which subpart each prompt will generate.
            3. Return this in json format with generated prompts as the keys and the respective subpart as it's value.

            Output format:
            {"New generated prompt" : "respective subpart"}
            Strictly follow this output format, do not provide anything else in the output.

            if the provided query is not relevant to any of the given subparts, Return empty dictionary, i.e. {}
            '''
    # Retrieving vector store from session state
    vector_store = st.session_state.vector_store
    # Initializing RetrievalQA interface
    qa_interface = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model= "gpt-3.5-turbo-16k", temperature=0),
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=False,
    )
    # Invoking query refinement
    refined_query = qa_interface.invoke(temp)
    output_dict = {}
    d = json.loads(refined_query["result"])
    if type(d)==dict:
        # Filtering prompts based on relevance to context
        for key in d:
            if "prompt" in key:
                return get_refined_query(query, context)
            if d[key] in context:
                output_dict[key] = d[key]
    else:
        # Recursive call if invalid output format received
        return get_refined_query(query, context)
    return output_dict

def add_context(query: str):
    #### **** ELASTICSEARCH **** ####

    # es_vector_store = get_elastic_vector()
    # r = es_vector_store.similarity_search(query, k=6, filter=[{"term": {"metadata.section.keyword": req_mod}}])

    #### **** QDRANT **** ####
    qd_vector_store = get_qdrant_vector()
    r = qd_vector_store.similarity_search(query, k=6, filter= {"section": req_mod})
    
    context = "\n".join(x.page_content for x in r)
    
    return context

def get_refined_query_es(query, subparts, req_mod):
    """
    Refine the user query based on context.

    This function analyzes the user query and context to generate prompts
    that are relevant and detailed to create the required subparts.

    Args:
        query (str): The user query.
        context (list): The context representing subparts of a section.

    Returns:
        dict: A dictionary mapping generated prompts to their respective subparts.
    """
    # Template for refining query
    
    temp = '''Find and convert the similar prompts into the required json format. These prompts should be similar to the provided query.
            
            Given the following contet, user query and CTD section Subparts. 

            Context: {context}

            Query provided by user: {query}
            ''' + f'''
            Subparts: {subparts}

            Your task is quite complex, Follow the given steps to provide the expected output in json format.
            1. Analyse the provided query and Find or generate few prompts which are most relevant and detailed to create the required subparts.
            2. Now from these generated prompts, find out which subpart each prompt will generate.
            3. Return this in json format with generated prompts as the keys and the respective subpart as it's value.

            Output format:
            ("generated prompt" : "respective subpart")
            Strictly follow this output format, do not provide anything else in the output.

            if the provided query is not relevant to any of the given subparts, Return empty dictionary.
            '''
    prompt = ChatPromptTemplate.from_template(temp)

    es_chain = (
        {"context": RunnableLambda(add_context), "query": RunnablePassthrough()}
        | prompt
        | ChatOpenAI()
        | StrOutputParser()
    )
    
    refined_query = es_chain.invoke(query)
    output_dict = {}
    d = json.loads(refined_query)
    if type(d)==dict:
        # Filtering prompts based on relevance to context
        for key in d:
            if "prompt" in key:
                return get_refined_query_es(query, subparts)
            if d[key] in subparts:
                output_dict[key] = d[key]
    else:
        # Recursive call if invalid output format received
        return get_refined_query_es(query, subparts)
    return output_dict

def response_to_soup(response):
    # Extracting only the HTML code
    html_code = re.search(r'```html(.*?)```', response, re.DOTALL).group(1).strip()

    # Converting into BeautifulSoup object
    soup = BeautifulSoup(html_code, 'html.parser')

    return soup

@st.cache_resource(show_spinner=False)
def load_chain_and_memory(drug_name, req_mod):
    """
    Load chat chain and memory for generating drug efficacy reports.

    Args:
        drug_name (str): The name of the drug.
        req_mod (str): The requested module.

    Returns:
        tuple: A tuple containing the chat chain, memory, and subparts context.
    """
    # Initializing initial prompt
    init_prompt = f"Drug name: {drug_name}\n\n" + PROMPTS[req_mod]["template"]
    model = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.1)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", '''You are an expert medical CTD drug report generator. The user wants you to create an specific section of the drug efficacy report. 
                He will first provide the details that need to be maintained in that particular section along with the subparts that needs to be present in it.
                Generate the result in more than 2500 words.
                Provide the generated contents in html format quoted in this- ```html\n<h6>(subpart name)</h6>(contents)```'''),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    # Initializing memory
    memory = ConversationBufferMemory(return_messages=True)
    memory.load_memory_variables({})

    # Creating chat chain
    chain = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt
        | model
    )
    # Invoking initial prompt
    input1 = {"input" : init_prompt}
    r = chain.invoke(input1)
    modified_r = r.content.split('Now')[0].split('```')[0]

    ##### Write code for filtering non-html########################

    st.session_state.responses.append(modified_r)
    memory.save_context(input1, {"output": modified_r})
    memory.load_memory_variables({})

    # Generating subparts list
    all_subparts = PROMPTS[req_mod]["subparts"]
    subparts = {}
    for part in all_subparts:
        subparts[part] = []
    
    return chain, memory, subparts

if __name__ == '__main__':
    
    st.set_page_config(page_title="CTD Report Generator", layout="wide")
    # Displaying header for the main page
    st.subheader('Drug Efficacy Reports')
    
    # Initializing session state variables if not present
    if 'responses' not in st.session_state:
        st.session_state['responses'] = []

    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    # Initializing query session state variables
    if "curr_query" not in st.session_state:
        st.session_state.curr_query = ""
        
    if "prev_query" not in st.session_state:
        st.session_state.prev_query = ""

    # Initializing flag session state variable
    if "flag" not in st.session_state:
        st.session_state.flag = True
        
    if "all_sections" not in st.session_state:
        st.session_state.all_sections = {}

    # Load the prompts
    file_path = 'prompts.json'
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        PROMPTS = data['PROMPTS']
        modules = data['modules']

    #############################################################################################################

    # Creating containers for response display and text input
    response_container = st.container(height=500, border=True)
    textcontainer = st.container()
    
    ####################################################################################################################
    # Sidebar for user input
    with st.sidebar:
        st.button("Reset", on_click=reset)  # Button to reset session state variables
        api_key = st.text_input('OpenAI API Key', type='password')  # Input for OpenAI API key
        drug_name = st.text_input('Please enter the drug name')  # Input for drug name
        
        # Select boxes for choosing required module
        level1 = st.selectbox("Select the required module.", modules.keys(), index=None)
        if not level1:
            st.warning("Please choose the required module.")
            st.stop()
        if type(modules[level1]) == dict and len(modules[level1]) > 0:
            level2 = st.selectbox("", modules[level1].keys(), index=None)
            if not level2:
                st.stop()
            if type(modules[level1][level2]) == dict and len(modules[level1][level2]) > 0:
                level3 = st.selectbox("", modules[level1][level2].keys(), index=None) 
                if not level3:
                    st.stop()
                if type(modules[level1][level2][level3]) == dict and len(modules[level1][level2][level3]) > 0:
                    level4 = st.selectbox("", modules[level1][level2][level3].keys(), index=None)
                    if not level4:
                        st.stop()
                    if type(modules[level1][level2][level3][level4]) == dict and len(modules[level1][level2][level3][level4]) > 0:
                        level5 = st.selectbox("", modules[level1][level2][level3][level4].keys(), index=None)
                        if not level5:
                            st.stop()
                        if type(modules[level1][level2][level3][level4][level5]) == dict and len(modules[level1][level2][level3][level4][level5]) > 0:
                            level6 = st.selectbox("", modules[level1][level2][level3][level4][level5].keys(), index=None)
                            if not level6:
                                st.stop()
                            else:
                                req_mod = level6
                        else:
                            req_mod = level5 
                    else:
                        req_mod = level4
                else:
                    req_mod = level3
            else:
                req_mod = level2
        else:
            req_mod = level1
            
        # Validating user inputs
        if not drug_name:
            st.warning("Please provide valid drug name.")
            st.stop()
        if not api_key:
            st.warning("Please provide valid OpenAI api key.")
            st.stop()
        if not req_mod:
            st.warning("Please choose the section to be generated.")
        if 'drug_name' not in st.session_state:
            st.session_state.drug_name = drug_name
            
        st.session_state.req_mod = req_mod
        
        # Button to start loading data
        if "load_data" not in st.session_state:
            st.session_state.load_data = False
        load_data = st.button('Start', on_click=set_load_true)

        # Fetching data if all required inputs are provided
        if drug_name and st.session_state.load_data and api_key and req_mod:
            with st.spinner('Fetching data.....'):
                os.environ['OPENAI_API_KEY'] = api_key
                chain, memory, subparts = load_chain_and_memory(drug_name, req_mod)
                st.session_state.all_sections[st.session_state.req_mod] = subparts
                # st.session_state.flag = True
                data = str(memory.chat_memory.messages)
                st.session_state.vector_store = get_conversation_vector(data)
                st.success('Data fetched successfully!')
        else:
            st.write('Please provide all the required information')
    
    ##########################################################################################################################

    with textcontainer:
        if "all_sections" in st.session_state and st.session_state.load_data:
            selected = pills("Select the subpart you want to be Generated", list(st.session_state.all_sections[st.session_state.req_mod].keys()), index=None, clearable=True)
            if selected:
                st.write("Selected query:", selected)
                st.session_state.curr_query = selected
        new_query = st.chat_input("How can I assist you?")
        if new_query:
            st.session_state.curr_query = new_query

    ############################################################################################################################

    # Processing user query and displaying refined query if available
    if st.session_state.flag and st.session_state.curr_query and ("drug_name" in st.session_state) and ("req_mod" in st.session_state):
        st.session_state.flag = False
        
        # refined_queries are coming in the form of a dictionary with prompts as key and respective subpart as it's value.
        with st.spinner("Finding refined queries....."):
            #st.write(st.session_state.curr_query)
            es_queries_dict = get_refined_query_es(st.session_state.curr_query, list(st.session_state.all_sections[st.session_state.req_mod].keys()), req_mod)
            st.write(es_queries_dict)
            
        try:
            if type(es_queries_dict)==dict:
                st.session_state.es_queries_dict = es_queries_dict
        except:
            st.warning("ERROR")
            st.write("Queries: \n" + es_queries_dict)
            
    if "es_queries_dict" in st.session_state and st.session_state.req_mod in st.session_state.all_sections and not st.session_state.flag:
        # st.session_state.flag = True
        if len(st.session_state.es_queries_dict)==0:
            st.warning("Query Irrelevant: Please choose from above subparts or provide another query")
        else:
            selected_query = pills("Choose any prompt:", list(st.session_state.es_queries_dict.keys()) + ['Other'], index=None, clearable=True)
            if selected_query == 'Other':
                # st.session_state.flag = False
                with st.spinner('Generating new refined queries'):
                    queries_dict = get_refined_query(st.session_state.curr_query, list(st.session_state.all_sections[st.session_state.req_mod].keys()))
                    st.session_state.queries_dict = queries_dict
            elif selected_query:
                st.success(selected_query)
                st.session_state.flag = True
                with st.spinner("Generating query output....."):
                    chain2, memory, subparts = load_chain_and_memory(st.session_state.drug_name, st.session_state.req_mod)
                    
                    q = {"input" : 'Query: \n' + selected_query + f'\n Subpart:\n{st.session_state.es_queries_dict[selected_query]} \nCreate this subpart in more than 2500 words. Return only the html code. "with code block"'}
                    count = 0
                    while True:
                        count += 1
                        res = chain2.invoke(q)
                        try:
                            html_content = response_to_soup(res.content)
                            break
                        except:
                            if count >= 6:
                                st.warning("Provide another query.")
                                st.stop()
                            continue                  
                    memory.save_context(q, {"output" : res.content})
                    memory.load_memory_variables({})
                    st.session_state.all_sections[st.session_state.req_mod][st.session_state.es_queries_dict[selected_query]].append(html_content)
                    # st.session_state.subparts[st.session_state.es_queries_dict[selected_query]].append(html_content)
                    
                    st.session_state.requests.append(selected_query)
                    st.session_state.responses.append(html_content.get_text())
                st.success("Generated successfully!")
        # del st.session_state.queries_dict
    
    if "queries_dict" in st.session_state and st.session_state.req_mod in st.session_state.all_sections and not st.session_state.flag:
        st.session_state.flag = True
        if len(st.session_state.queries_dict)==0:
            st.warning("Query Irrelevant: Please choose from above subparts or provide another query")
        else:
            selected_query = pills("Choose any prompt:", list(st.session_state.queries_dict.keys()), index=None, clearable=True)
            st.success(selected_query)
            if selected_query:
                with st.spinner("Generating query output....."):
                    chain2, memory, subparts = load_chain_and_memory(st.session_state.drug_name, st.session_state.req_mod)
                    q = {"input" : 'Query: \n' + selected_query + f'\n Subpart:\n{st.session_state.es_queries_dict[selected_query]} \nCreate this subpart in more than 2500 words. Return only the html code. "with code block"'}
                    count = 0
                    while True:
                        count += 1
                        res = chain2.invoke(q)
                        try:
                            html_content = response_to_soup(res.content)
                            break
                        except:
                            if count >= 6:
                                st.warning("Provide another query.")
                                st.stop()
                            continue            
                    memory.save_context(q, {"output" : res.content})
                    memory.load_memory_variables({})
                    st.session_state.all_sections[st.session_state.req_mod][st.session_state.queries_dict[selected_query]].append(html_content)
                    # st.session_state.subparts[st.session_state.queries_dict[selected_query]].append(html_content)
                    
                    st.session_state.requests.append(selected_query)
                    st.session_state.responses.append(html_content.get_text())
                st.success("Generated successfully!")

    ###############################################################################################################
    
    # Displaying conversation history
    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['requests'])):
                st.chat_message("assistant").write(st.session_state['responses'][i])
                if i < len(st.session_state['requests']):
                    st.chat_message("user").write(st.session_state["requests"][i])
            st.chat_message("assistant").write(st.session_state['responses'][-1])
