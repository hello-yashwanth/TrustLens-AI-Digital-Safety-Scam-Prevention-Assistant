import os
import numpy as np
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


# Helper function to check text quality
def validate_ocr_quality(text):
  if not text or len(text.strip()) < 10:
    return False, 'Text is too short or empty.'
  valid_chars = sum(c.isalnum() or c.isspace() for c in text)
  quality_ratio = valid_chars / len(text)
  if quality_ratio < 0.6:
    return False, 'Too many garbled or unreadable characters detected.'
  return True, 'OK'


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
      ' and smaller bullet descriptions.\n'
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

# Initialize session state for chat messages and document text
if 'messages' not in st.session_state:
  st.session_state['messages'] = [
      {
          'role': 'assistant',
          'content': (
              'Hello! I am TrustLens AI. You can paste suspicious text or'
              ' screenshots, or ask me how to report a scam.'
          ),
      }
  ]

if 'doc_text' not in st.session_state:
  st.session_state['doc_text'] = ''

# Sidebar: Evidence Upload (Using PIL/Streamlit text fallback to guarantee stability)
with st.sidebar:
  st.header('📁 Evidence Upload')
  st.write('Upload or paste text from suspicious messages.')
  
  pasted_text = st.text_area('Or paste suspicious message text here:')
  if pasted_text:
    st.session_state['doc_text'] = pasted_text.strip()
    st.success('Text loaded into active context!')

  if st.session_state['doc_text']:
    with st.expander('View Extracted Document Memory'):
      st.text_area(
          'Active Context:',
          value=st.session_state['doc_text'],
          height=150,
          disabled=True,
      )

# Display Chat History
for message in st.session_state['messages']:
  with st.chat_message(message['role']):
    st.markdown(message['content'])

# Clickable Question Suggestions (Rendered if document memory exists)
if st.session_state['doc_text']:
  st.markdown('---')
  st.markdown('### 💡 Suggested Questions About This Document')
  st.write(
      'Choose a quick question to investigate the uploaded message further:'
  )

  col1, col2 = st.columns(2)
  selected_suggestion = None

  with col1:
    if st.button('🔍 Is this job offer genuine or fake?'):
      selected_suggestion = 'Is this job offer genuine or fake?'
    if st.button('🚩 What are the biggest red flags?'):
      selected_suggestion = 'What are the biggest red flags?'
    if st.button('📄 Which exact text indicates a scam?'):
      selected_suggestion = 'Which exact text indicates a scam?'

  with col2:
    if st.button('🏢 How can I verify the company?'):
      selected_suggestion = 'How can I verify the company?'
    if st.button('🛡️ What should I do next?'):
      selected_suggestion = 'What should I do next?'
    if st.button('📚 Which official sources support your answer?'):
      selected_suggestion = 'Which official sources support your answer?'

  if selected_suggestion:
    st.session_state['messages'].append(
        {'role': 'user', 'content': selected_suggestion}
    )
    with st.chat_message('user'):
      st.markdown(selected_suggestion)

    with st.chat_message('assistant'):
      with st.spinner('Analyzing query...'):
        full_input = (
            f'Context Document / Extracted Evidence:\n'
            f'{st.session_state["doc_text"]}\n\nUser'
            f' Query / Question: {selected_suggestion}'
        )
        response = rag_chain.invoke({'input': full_input})
        answer = response['answer']
        st.markdown(answer)

        st.session_state['messages'].append(
            {'role': 'assistant', 'content': answer}
        )
        st.rerun()

# Static Bottom Chat Input
if prompt_input := st.chat_input(
    'Ask TrustLens AI or analyze an uploaded document...'
):
  st.session_state['messages'].append(
      {'role': 'user', 'content': prompt_input}
  )
  with st.chat_message('user'):
    st.markdown(prompt_input)

  with st.chat_message('assistant'):
    with st.spinner('Thinking...'):
      full_input = prompt_input
      if st.session_state['doc_text']:
        full_input = (
            f'Context Document / Extracted Evidence:\n'
            f'{st.session_state["doc_text"]}\n\nUser'
            f' Query / Question: {prompt_input}'
        )

      response = rag_chain.invoke({'input': full_input})
      answer = response['answer']
      st.markdown(answer)

      st.session_state['messages'].append(
          {'role': 'assistant', 'content': answer}
      )