import base64
import os
from PIL import Image
import streamlit as st
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Page Configuration
st.set_page_config(
    page_title='TrustLens AI - Digital Safety & Scam Prevention',
    page_icon='🛡️',
    layout='centered',
)

st.title('🛡️ TrustLens AI')
st.subheader('Digital Safety & Scam Prevention Assistant')
st.write(
    'Analyze suspicious messages, fake job offers, and phishing attempts using'
    ' official cybersecurity guides.'
)


# 2. Load RAG Pipeline (Cached)
@st.cache_resource
def init_rag_pipeline():
  loader = DirectoryLoader('./data', glob='./*.pdf', loader_cls=PyPDFLoader)
  docs = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=500, chunk_overlap=50
  )
  chunks = text_splitter.split_documents(docs)

  embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
  vector_store = FAISS.from_documents(chunks, embeddings)
  retriever = vector_store.as_retriever(search_kwargs={'k': 3})

  llm = ChatGroq(
      model='openai/gpt-oss-120b',
      temperature=0,
      groq_api_key=st.secrets['GROQ_API_KEY'],
  )

  system_prompt = (
      'You are TrustLens AI, an expert cybersecurity and digital safety'
      ' assistant. Your goal is to help users spot scams, phishing, fake job'
      ' offers, secure their digital lives, and respond effectively to financial'
      ' fraud.\n\n'
      'Behavior Guidelines:\n'
      '1. **General Questions / Chat**: Respond naturally, intelligently, and'
      ' conversationally as an expert.\n'
      '2. **Document / Evidence Analysis**: When a user uploads a new image or'
      ' document, analyze it dynamically based on the current file name and'
      ' details provided. Identify the company (e.g., Indigo Airlines /'
      ' InterGlobe Aviation), check for red flags, and provide a clear'
      ' verdict of **SCAM (Fake)** or **Genuine**.\n'
      '3. **Scam Incident Reporting**: Provide concise structured steps using'
      ' plain markdown headings (###) for immediate actions, cybercrime portal'
      ' (https://cybercrime.gov.in/), and helpline (1930).\n\n'
      'Retrieved Context from Official Guides:\n{context}'
  )

  prompt = ChatPromptTemplate.from_messages(
      [('system', system_prompt), ('human', '{input}')]
  )

  question_answer_chain = create_stuff_documents_chain(llm, prompt)
  return create_retrieval_chain(retriever, question_answer_chain)


with st.spinner(
    'Loading RAG knowledge base and indexing PDFs... Please wait.'
):
  rag_chain = init_rag_pipeline()

# Initialize session state
if 'messages' not in st.session_state:
  st.session_state['messages'] = [
      {
          'role': 'assistant',
          'content': (
              'Hello! I am TrustLens AI. Upload your offer letter or'
              ' screenshot in the sidebar, and ask me anything below!'
          ),
      }
  ]

if 'current_file_name' not in st.session_state:
  st.session_state['current_file_name'] = None

# Sidebar: File & Image Uploader
with st.sidebar:
  st.header('📁 Evidence Upload')
  uploaded_file = st.file_uploader(
      'Upload Offer Letter or Screenshot:', type=['png', 'jpg', 'jpeg', 'pdf']
  )

  if uploaded_file is not None:
    # Check if the user uploaded a new file
    if st.session_state['current_file_name'] != uploaded_file.name:
      st.session_state['current_file_name'] = uploaded_file.name
      st.success(f'Loaded new file: {uploaded_file.name}')

    if uploaded_file.type == 'application/pdf':
      import pypdf

      reader = pypdf.PdfReader(uploaded_file)
      pdf_text = ''.join([page.extract_text() or '' for page in reader.pages])
      st.session_state['file_content_context'] = pdf_text
    else:
      image = Image.open(uploaded_file)
      st.image(image, caption='Active Uploaded Evidence', use_container_width=True)
      st.session_state['file_content_context'] = (
          f'User uploaded image file named: {uploaded_file.name}.'
      )
  else:
    st.session_state['current_file_name'] = None
    st.session_state['file_content_context'] = None

# Display Chat History
for message in st.session_state['messages']:
  with st.chat_message(message['role']):
    st.markdown(message['content'])

# Main Chat Input
if prompt_input := st.chat_input('Ask about the newly uploaded document...'):
  st.session_state['messages'].append(
      {'role': 'user', 'content': prompt_input}
  )
  with st.chat_message('user'):
    st.markdown(prompt_input)

  with st.chat_message('assistant'):
    with st.spinner('Analyzing current evidence...'):
      evidence_context = ''
      if st.session_state.get('file_content_context'):
        evidence_context = (
            f'[Currently Active Uploaded Evidence File Name:'
            f' {st.session_state["current_file_name"]}]\n'
            f'{st.session_state["file_content_context"]}\n\n'
        )
      else:
        evidence_context = '[No active evidence file currently uploaded.]\n\n'

      full_input = (
          f'{evidence_context}User Question: {prompt_input}. (Analyze the'
          ' newly active evidence file name and contents above. Do NOT use or'
          ' reference previous old uploads. Answer accurately based solely on'
          ' the currently active file.)'
      )

      response = rag_chain.invoke({'input': full_input})
      answer = response['answer']
      st.markdown(answer)

      st.session_state['messages'].append(
          {'role': 'assistant', 'content': answer}
      )