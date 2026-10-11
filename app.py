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
      '2. **Document / Evidence Analysis**: If an uploaded document or evidence'
      ' snippet is provided and the user asks to analyze it, structure your'
      ' response into an Evidence-Based Risk Report with prominent large headings'
      ' and smaller bullet descriptions. Give clear assessments like "Real" or "Fake"'
      ' when asked directly based on the evidence.\n'
      '3. **Scam Incident / Financial Fraud Reporting**: If the user reports a'
      ' suspected financial scam, online fraud, or asks what to do after being'
      ' scammed, you MUST generate a concise, structured answer in **plain text list format**'
      ' (DO NOT use tables). Use large markdown headings (###) for each of the following 6 points:\n'
      '   ### 1. Immediate Action\n'
      '   - What to do first, especially if money has been transferred (contact bank immediately to freeze transaction).\n'
      '   ### 2. Official Reporting Channel\n'
      '   - The government portal (https://cybercrime.gov.in/) and helpline (Call 1930).\n'
      '   ### 3. Evidence Checklist\n'
      '   - Screenshots, transaction IDs, dates, sender details, and URLs.\n'
      '   ### 4. Account Protection\n'
      '   - Secure affected accounts, change passwords, and enable 2FA.\n'
      '   ### 5. Source References\n'
      '   - Official guidance links ([National Cyber Crime Reporting Portal](https://cybercrime.gov.in/)).\n'
      '   ### 6. Tailored Follow-Up Questions\n'
      '   - Ask whether money was lost, a link was clicked, or personal information was shared so advice can be tailored.\n\n'
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
              'Hello! I am TrustLens AI. Upload an image or document in the'
              ' sidebar and ask me any question in the chat below!'
          ),
      }
  ]

if 'uploaded_image' not in st.session_state:
  st.session_state['uploaded_image'] = None

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
      st.session_state['uploaded_image'] = pdf_text.strip()
      st.success('PDF uploaded and indexed successfully!')
    else:
      image = Image.open(uploaded_file)
      st.session_state['uploaded_image'] = image
      st.image(image, caption='Active Evidence Image', use_container_width=True)
      st.success('Image uploaded successfully! You can now ask questions below.')
  else:
    if st.button('Clear Uploaded Evidence'):
      st.session_state['uploaded_image'] = None
      st.rerun()

# Display Chat History
for message in st.session_state['messages']:
  with st.chat_message(message['role']):
    st.markdown(message['content'])

# Main Chat Input (Handles both text and context together)
if prompt_input := st.chat_input(
    'Ask whether it is real or fake, or ask anything else...'
):
  st.session_state['messages'].append(
      {'role': 'user', 'content': prompt_input}
  )
  with st.chat_message('user'):
    st.markdown(prompt_input)

  with st.chat_message('assistant'):
    with st.spinner('Analyzing...'):
      evidence_context = ''
      if st.session_state['uploaded_image'] is not None:
        if isinstance(st.session_state['uploaded_image'], str):
          evidence_context = (
              f'Uploaded PDF Text Context:\n{st.session_state["uploaded_image"]}\n\n'
          )
        else:
          # If it's an image, pass a descriptive note for RAG retrieval alongside the query
          evidence_context = (
              '[Note: The user has uploaded an image/screenshot of an offer'
              ' letter or document for verification.]\n\n'
          )

      full_input = evidence_context + f'User Query: {prompt_input}'
      response = rag_chain.invoke({'input': full_input})
      answer = response['answer']
      st.markdown(answer)

      st.session_state['messages'].append(
          {'role': 'assistant', 'content': answer}
      )