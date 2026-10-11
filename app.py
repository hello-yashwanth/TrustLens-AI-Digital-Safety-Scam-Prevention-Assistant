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
      '2. **Document / Evidence Analysis**: When a user uploads an image or'
      ' document (such as a Wipro or corporate job offer letter requiring an'
      ' upfront fee or security deposit), analyze it thoroughly. State clearly'
      ' whether it is a **SCAM (Fake)** or **Genuine**. Point out specific red'
      ' flags such as requests for money/UPI transfers, training fees, or'
      ' outdated dates.\n'
      '3. **Scam Incident / Financial Fraud Reporting**: If the user reports a'
      ' suspected financial scam or fraud, provide concise, structured action'
      ' steps using plain headings (###):\n'
      '   ### 1. Immediate Action\n'
      '   - Contact your bank immediately to block any transfers.\n'
      '   ### 2. Official Reporting Channel\n'
      '   - Visit the National Cyber Crime Portal (https://cybercrime.gov.in/) or call 1930.\n'
      '   ### 3. Evidence Checklist\n'
      '   - Save screenshots, transaction IDs, and sender emails.\n'
      '   ### 4. Account Protection\n'
      '   - Secure passwords and enable 2FA.\n\n'
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
              'Hello! I am TrustLens AI. Upload an offer letter screenshot in'
              ' the sidebar, and ask me whether it is real or fake in the chat'
              ' below!'
          ),
      }
  ]

if 'uploaded_evidence' not in st.session_state:
  st.session_state['uploaded_evidence'] = ''

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
      st.session_state['uploaded_evidence'] = pdf_text.strip()
      st.success('PDF text extracted successfully!')
    else:
      image = Image.open(uploaded_file)
      st.image(image, caption='Uploaded Evidence', use_container_width=True)
      st.success('Image loaded successfully!')

  # Text helper box so users can easily paste the text if the image text needs highlighting
  extracted_helper = st.text_area(
      'Or paste text/details from the image here:',
      value=st.session_state['uploaded_evidence'],
      height=100,
  )
  if extracted_helper != st.session_state['uploaded_evidence']:
    st.session_state['uploaded_evidence'] = extracted_helper

# Display Chat History
for message in st.session_state['messages']:
  with st.chat_message(message['role']):
    st.markdown(message['content'])

# Main Chat Input
if prompt_input := st.chat_input(
    'Ask whether it is real or fake, company name, etc...'
):
  st.session_state['messages'].append(
      {'role': 'user', 'content': prompt_input}
  )
  with st.chat_message('user'):
    st.markdown(prompt_input)

  with st.chat_message('assistant'):
    with st.spinner('Analyzing evidence...'):
      evidence_context = ''
      if st.session_state['uploaded_evidence']:
        evidence_context = (
            f'Uploaded Document/Image Text Content Context:\n'
            f'{st.session_state["uploaded_evidence"]}\n\n'
        )

      full_input = (
          f'{evidence_context}User Question: {prompt_input}. (If the user asks'
          ' if the offer letter is real or fake, analyze the uploaded text'
          ' context carefully. Look for red flags like training fees, security'
          ' deposits, UPI deposit requirements, and mismatched company domains'
          ' to give a definitive verdict.)'
      )

      response = rag_chain.invoke({'input': full_input})
      answer = response['answer']
      st.markdown(answer)

      st.session_state['messages'].append(
          {'role': 'assistant', 'content': answer}
      )