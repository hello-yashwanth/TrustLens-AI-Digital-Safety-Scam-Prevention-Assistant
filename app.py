import base64
import os
from PIL import Image
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
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


# 2. Initialize Groq Model
@st.cache_resource
def init_groq_llm():
  return ChatGroq(
      model='llama-3.2-11b-vision-preview',
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
              'Hello! I am TrustLens AI. Upload your offer letter or'
              ' screenshot in the sidebar, and ask me if it is real or fake!'
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
      st.session_state['extracted_document_text'] = pdf_text
      st.success('PDF text loaded successfully!')
    else:
      image = Image.open(uploaded_file)
      st.image(
          image, caption='Active Uploaded Evidence Image', use_container_width=True
      )

      # Convert image to bytes to send directly to Groq Vision API
      bytes_data = uploaded_file.getvalue()
      base64_img = base64.b64encode(bytes_data).decode('utf-8')
      st.session_state['extracted_document_text'] = {
          'type': 'image_url',
          'image_url': {'url': f'data:image/jpeg;base64,{base64_img}'},
      }
      st.success('Image successfully loaded into AI Vision memory!')
  else:
    st.session_state['extracted_document_text'] = ''

# Display Chat History
for message in st.session_state['messages']:
  with st.chat_message(message['role']):
    st.markdown(message['content'])

# Main Chat Input
if prompt_input := st.chat_input('Ask if this offer letter is real or fake...'):
  st.session_state['messages'].append(
      {'role': 'user', 'content': prompt_input}
  )
  with st.chat_message('user'):
    st.markdown(prompt_input)

  with st.assistant if hasattr(st, 'assistant') else st.chat_message('assistant'):
    with st.spinner('TrustLens AI Vision is analyzing your document...'):
      system_prompt = (
          'You are TrustLens AI, an expert cybersecurity and digital safety'
          ' assistant. Your goal is to help users spot scams, phishing, and'
          ' fake job offers.\n'
          'When an image or document is provided, you MUST analyze its visual'
          ' contents directly. Identify the company name (such as DP World,'
          ' Wipro, or Indigo), check for scam warning signs (such as requests'
          ' for upfront security deposits, registration fees, training charges,'
          ' or unofficial domains), and give a clear definitive verdict of'
          ' **SCAM (Fake)** or **Genuine** with clear bullet points explaining'
          ' your reasoning.'
      )

      content_blocks = [{
          'type': 'text',
          'text': f'{system_prompt}\n\nUser Question: {prompt_input}',
      }]

      # Attach the image block if present
      doc_content = st.session_state.get('extracted_document_text')
      if doc_content:
        if isinstance(doc_content, dict):
          content_blocks.append(doc_content)
        else:
          content_blocks[0]['text'] += (
              f'\n\nDocument Text Context:\n{doc_content}'
          )

      try:
        message = HumanMessage(content=content_blocks)
        response = llm.invoke([message])
        answer = response.content
      except Exception as e:
        # Fallback if vision payload errors out
        answer = (
            f'Analysis Error: {str(e)}. Please make sure your image file is a'
            ' clear PNG or JPG.'
        )

      st.markdown(answer)
      st.session_state['messages'].append(
          {'role': 'assistant', 'content': answer}
      )