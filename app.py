import os
from PIL import Image
import pytesseract
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
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

# Sidebar: File & Image Uploader with Backend OCR Scanning
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
      st.success('PDF text extracted successfully!')
    else:
      image = Image.open(uploaded_file)
      st.image(
          image, caption='Active Uploaded Evidence Image', use_container_width=True
      )

      # BACKEND OCR SCANNING
      with st.spinner('Scanning image text in backend...'):
        try:
          scanned_text = pytesseract.image_to_string(image).strip()
          if scanned_text and len(scanned_text) > 5:
            st.session_state['extracted_document_text'] = scanned_text
            st.success(
                f'Successfully scanned {len(scanned_text)} characters from'
                ' image!'
            )
          else:
            st.warning(
                'Image scanned, but very little text was detected. Please add'
                ' details below.'
            )
        except Exception as e:
          st.error(f'OCR Scan Error: {str(e)}')
  else:
    st.session_state['extracted_document_text'] = ''

  # Editable text area to review or supplement scanned text
  manual_text = st.text_area(
      'Scanned / Active Document Text:',
      value=st.session_state['extracted_document_text'],
      height=140,
  )
  if manual_text != st.session_state['extracted_document_text']:
    st.session_state['extracted_document_text'] = manual_text

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
          'When document text context is provided, analyze it thoroughly. Identify'
          ' the exact company name (such as DP World, Wipro, or Indigo), check'
          ' for scam warning signs (such as requests for upfront security'
          ' deposits, visa processing fees, training charges, or unofficial'
          ' contact domains), and give a clear definitive verdict of **SCAM (Fake)**'
          ' or **Genuine** with clear bullet points explaining your'
          ' reasoning.\n'
          'If the user asks what to do if scammed, provide structured action'
          ' steps, reference the official cybercrime portal'
          ' (https://cybercrime.gov.in/), and helpline 1930.'
      )

      doc_context = st.session_state.get('extracted_document_text', '')
      full_input = (
          f'System Guidelines: {system_prompt}\n\n'
          f'Scanned Document Text Context:\n{doc_context}\n\n'
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