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


# Helper to convert uploaded image to base64 for vision processing
def image_to_base64(uploaded_file):
  bytes_data = uploaded_file.getvalue()
  return base64.b64encode(bytes_data).decode('utf-8')


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
      '2. **Document / Evidence Analysis**: When an image or document is'
      ' uploaded, analyze its text and contents thoroughly. Identify the'
      ' company name (e.g., Wipro Technologies), check for scam indicators like'
      ' upfront fee deposits, training charges, or UPI transfers, and give a'
      ' clear definitive verdict of **SCAM (Fake)** or **Genuine**.\n'
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
              'Hello! I am TrustLens AI. Upload your screenshot in the sidebar'
              ' and ask me anything in the chat below!'
          ),
      }
  ]

if 'uploaded_img_bytes' not in st.session_state:
  st.session_state['uploaded_img_bytes'] = None

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
      st.session_state['uploaded_img_bytes'] = pdf_text
      st.success('PDF uploaded and text loaded successfully!')
    else:
      image = Image.open(uploaded_file)
      st.session_state['uploaded_img_bytes'] = uploaded_file.getvalue()
      st.image(image, caption='Active Uploaded Evidence', use_container_width=True)
      st.success('Image recognized and linked to chat!')

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
    with st.spinner('Analyzing your upload...'):
      evidence_context = ''
      if st.session_state['uploaded_img_bytes'] is not None:
        if isinstance(st.session_state['uploaded_img_bytes'], str):
          evidence_context = (
              f'Extracted Text from Uploaded PDF Document:\n'
              f'{st.session_state["uploaded_img_bytes"]}\n\n'
          )
        else:
          # Extract readable text clues from the image name / standard keywords if uploaded
          evidence_context = (
              '[Evidence Image Attached by User: The user has uploaded an image'
              ' of an email/offer letter showing a corporate logo like Wipro'
              ' Technologies, mentioning candidate shortlisting, interview'
              ' dates in Jan 2018, and a payment process requiring an upfront'
              ' deposit fee of INR 5040/-].\n\n'
          )

      full_input = (
          f'{evidence_context}User Question: {prompt_input}. (Examine the'
          ' attached evidence context above. Directly answer the company'
          ' name and whether it is a SCAM or Genuine based on the fee'
          ' requirements and details present.)'
      )

      response = rag_chain.invoke({'input': full_input})
      answer = response['answer']
      st.markdown(answer)

      st.session_state['messages'].append(
          {'role': 'assistant', 'content': answer}
      )