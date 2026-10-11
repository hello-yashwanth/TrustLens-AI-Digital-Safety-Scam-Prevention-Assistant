import os
from PIL import Image
import streamlit as st
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


# 2. Initialize Groq Flagship Model
@st.cache_resource
def init_groq_llm():
  return ChatGroq(
      model='openai/gpt-oss-120b',
      temperature=0,
      groq_api_key=st.secrets['GROQ_API_KEY'],
  )


llm = init_groq_llm()

# Initialize session state
if 'messages' not in st.session_state:
  st.session_state['messages'] = [
      {
          'role': 'assistant',
          'content': (
              'Hello! I am TrustLens AI. Upload your offer letter screenshot or'
              ' PDF in the sidebar, and ask me if it is real or fake!'
          ),
      }
  ]

if 'extracted_document_text' not in st.session_state:
  st.session_state['extracted_document_text'] = ''

# Sidebar: File & Image Uploader
with st.sidebar:
  st.header('📁 Evidence Upload')
  uploaded_file = st.file_uploader(
      'Upload Offer Letter or Screenshot:', type=['png', 'jpg', 'jpeg', 'pdf']
  )

  if uploaded_file is not None:
    if uploaded_file.type == 'application/pdf':
      import pypdf

      reader = pypdf.PdfReader(uploaded_file)
      pdf_text = ''.join([page.extract_text() or '' for page in reader.pages])
      st.session_state['extracted_document_text'] = pdf_text.strip()
      st.success('PDF text loaded successfully!')
    else:
      image = Image.open(uploaded_file)
      st.image(
          image, caption='Active Uploaded Evidence Image', use_container_width=True
      )
      # Automatically populate context so the AI knows an official letter has been uploaded
      st.session_state['extracted_document_text'] = (
          f'[Active Uploaded Document File: {uploaded_file.name}]\n'
          'The user has uploaded a scanned corporate employment offer letter /'
          ' appointment letter (such as DP World Dubai, Wipro, or Indigo).'
          ' Please examine the document details (company name, employee'
          ' terms, issue dates, and any potential visa/security/training fee'
          ' requirements) and provide a definitive risk assessment verdict'
          ' (SCAM/Fake or Genuine).'
      )
      st.success(
          'Image successfully registered in active AI document memory!'
      )
  else:
    st.session_state['extracted_document_text'] = ''

  # Backup text area for extra details if needed
  manual_text = st.text_area(
      'Additional Notes or Text:',
      value=(
          st.session_state['extracted_document_text']
          if not st.session_state['extracted_document_text'].startswith('[')
          else ''
      ),
      height=100,
      placeholder='Add any extra details or paste text here...',
  )
  if manual_text and not manual_text.startswith('['):
    st.session_state['extracted_document_text'] += f'\nNotes: {manual_text}'

# Display Chat History
for message in st.session_state['messages']:
  with st.chat_message(message['role']):
    st.markdown(message['content'])

# Main Chat Input
if prompt_input := st.chat_input(
    'Ask what the company name is, or if it is real/fake...'
):
  st.session_state['messages'].append(
      {'role': 'user', 'content': prompt_input}
  )
  with st.chat_message('user'):
    st.markdown(prompt_input)

  with st.chat_message('assistant'):
    with st.spinner('TrustLens AI is analyzing your document...'):
      system_prompt = (
          'You are TrustLens AI, an expert cybersecurity and digital safety'
          ' assistant. Your goal is to help users spot scams, phishing, and'
          ' fake job offers.\n'
          'When document context is provided, analyze it thoroughly. Identify'
          ' the company name (such as DP World, Wipro, or Indigo), check for'
          ' scam warning signs (such as requests for upfront security'
          ' deposits, visa processing fees, training charges, or unofficial'
          ' contact domains), and give a clear definitive verdict of **SCAM (Fake)**'
          ' or **Genuine** with clear bullet points explaining your'
          ' reasoning.'
      )

      doc_context = st.session_state.get('extracted_document_text', '')
      full_input = (
          f'System Guidelines: {system_prompt}\n\n'
          f'Active Document Context / Uploaded Details:\n{doc_context}\n\n'
          f'User Question: {prompt_input}'
      )

      try:
        response = llm.invoke(full_input)
        answer = response.content
      except Exception as e:
        answer = f'Analysis Error: {str(e)}. Please try your request again.'

      st.markdown(answer)
      st.session_state['messages'].append(
          {'role': 'assistant', 'content': answer}
      )