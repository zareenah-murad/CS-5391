# CS-5391
Weekly progress updates for our CS 5391 Project exploring Retrieval Augmented Generation and Large Language Models.


## Week 1

**Article Notes: What is retrieval-augmented generation?**
* https://research.ibm.com/blog/retrieval-augmented-generation-RAG
* “..an AI framework for retrieving facts from an external knowledge base to ground large language models (LLMs) on the most accurate, up-to-date information and to give users insight into LLMs’ generative process” 
improves “the quality of LLM-generated responses by grounding the model on external sources of knowledge”
* Because you ground an LLM on a set of external verifiable facts, there are fewer opportunities for the model to leak sensitive data or “hallucinate” and provide incorrect information
* RAG reduces the need for users to continuously train the model on new data and update its parameters -> lower computational and financial costs 
* Transformer: AI architecture that “turns heaps of raw data into a compressed representation of its basic structure” 
* Two phases of RAG: retrieval and content generation
* Retrieval: algorithms search for information relevant to the prompt
* Open-domain consumer setting: facts come from indexed documents on the internet
* Closed-domain enterprise setting: facts come from a narrower set of sources 

**Article Notes: Retrieval-augmented Generation (RAG) Explained: Understanding Key Concepts**
* https://www.datastax.com/guides/what-is-retrieval-augmented-generation
* RAG “enhances traditional language model responses by incorporating real-time, external data retrieval”
* After receiving the query, RAG taps into external data sources (databases, APIs, document repositories, etc) to gain information beyond the language model’s initial training data
* The external data and user query are transformed into numerical vector representations 
* Next is augmentation, it integrates the new information into the language model in a way that maintains the context and flow of the original query
* To maintain efficacy of RAG, external data sources are regularly updated 
* Source data: starting point, a vast corpus of text documents, websites, or databases
* Data chunking: data is divided into manageable chunks, ensures the system can efficiently scan through the data and enables quick retrieval of relevant content, effective chunking strategies can drastically improve speed and accuracy 
* Text-to-vector conversion (embeddings): transform text into mathematical vectors using complex software models, create a vector database, the vectors encapsulate the semantics and context of the text
* Link between source data & embeddings: the link between the source data and embeddings is vital

##

**Downloading Ollama and Connecting to Genuse**
* Downloaded Ollama, loaded llama3 locally on mac
* Started a conversation: ollama run llama3
* Ended a conversation: control + d or /bye 

* Connected to genuse 60 (20 dual core CPUs, 768 GB memory)
* Loaded llama3 onto genuse60
* Command Window 1: connect to genuse, %ollama serve
* Command Window 2: connect to genuse, %ollama run llama3

* Create embeddingsTest.py file to test embeddings on genuse server, code provided by Dr. C
* File wont work, permission denied for typing_extensions.py
* Ran embeddingsTest.py locally after I installed pip and then installed transformers and torch


**Building a RAG and Running it Locally Using Langchain, Ollama, & Streamlit**
* https://medium.com/@vndee.huynh/build-your-own-rag-and-run-it-locally-langchain-ollama-streamlit-181d42805895
* Pulled the mistral LLM model: %ollama pull mistral
* Ingest: accepts a file path and loads it into vector storage, it splits the document into smaller chunks (to fit the token limit of the LLM) and then vectorizes the chunks using Qdrant FastEmbeddings and store into Chroma
* Ask: handles user queries, RetrivialQAChain retrieves relevant contexts (document chunks) using vector similarity search techniques and then using the user’s question and retrieved contexts, we can make a prompt and request from the LLM server 
* Created python script with the code that defined the ChatPDF class (chat_pdf.py) and provided a RAG pipeline for handling PDFs, contains ingest and ask 
* Installed streamlit %install streamlit streamlit-chat
* Created python script with the code that made a simple UI using streamlit (chatpdf_app.py)
* The application allows user to upload a PDF, then that’s processed by the ChatPDF class and enables question/answering based on document’s context
* Ran the code %streamlit run chatpdf_app.py
* It did not work, I had to update the scripts to match my filenames
* Still did not work, I had to install langchain
* The UI popped up, but when I uploaded the PDF I got an error because I needed to install pypdf, chroma, openai
* Streamlit said to install the Watchdog module for better performance 
* Install qdrant-client
* Install chromadb
* Uploaded one file
* IT WORKED! It said ingesting for a second and then gave me the option to enter a message

**Improvements**
* Edited chat_pdf.py
* Verified imports to ensure all are from existing and correctly named modules
* Fixed initialization of self.vector_store in the ingest method and used self.vector_store instead of vector_store for consistency 
* Ensured clear method resets self.vector_store, self.retriever, and self.chain to None
* Edited chatpdf_app.py
* Used st.set_page_config to set page title
* Added a check for st.session_state initialization to ensure consistent initialization of the ChatPDF instance and message history

**Suggestions for Fine-Tuning (according to the article)**
* Add memory to the conversation chain so that it remembers the conversation flow 
* Allow multiple file uploads
* Use other LLM models 
* Enhance the RAG pipeline (retrieval metric, embedding model, add layers like a re-ranker, etc)

### Code

**embeddingsTest.py**
``` 
      import torch
      from transformers import AutoTokenizer, AutoModel
      
      # Load pre-trained LLM model and tokenizer
      model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
      tokenizer = AutoTokenizer.from_pretrained(model_name)
      model = AutoModel.from_pretrained(model_name)
      
      # Example sentences
      sentence1 = "The quick brown fox jumps over the lazy dog."
      sentence2 = "A fast brown fox leaps over a lazy dog."
      
      # Tokenize and encode the sentences
      inputs1 = tokenizer(sentence1, return_tensors="pt", padding=True, truncation=True)
      inputs2 = tokenizer(sentence2, return_tensors="pt", padding=True, truncation=True)
      
      # Get sentence embeddings
      with torch.no_grad():
          embeddings1 = model(**inputs1).last_hidden_state.mean(dim=1)
          embeddings2 = model(**inputs2).last_hidden_state.mean(dim=1)
      
      # Compute cosine similarity
      cosine_similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
      
      print(f"Sentence 1: {sentence1}")
      print(f"Sentence 2: {sentence2}")
      print(f"Cosine Similarity: {cosine_similarity.item()}")
``` 


**chat_pdf.py**
``` 
      from langchain_community.vectorstores import Chroma
      from langchain_community.chat_models import ChatOllama
      from langchain_community.embeddings import FastEmbedEmbeddings  # Make sure this module aligns with your environment
      from langchain.schema.output_parser import StrOutputParser
      from langchain_community.document_loaders import PyPDFLoader
      from langchain.text_splitter import RecursiveCharacterTextSplitter
      from langchain.vectorstores.utils import filter_complex_metadata
      from langchain.schema.runnable import RunnablePassthrough
      from langchain.prompts import PromptTemplate
      
      
      class ChatPDF:
          vector_store = None
          retriever = None
          chain = None
      
          def __init__(self):
              self.model = ChatOllama(model="mistral")
              self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
              self.prompt = PromptTemplate.from_template(
                  """
                  <s> [INST] Vous êtes un assistant pour les tâches de réponse aux questions. Utilisez les éléments de contexte suivants pour répondre à la question. 
                  Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas.. Utilisez trois phrases
                   maximum et soyez concis dans votre réponse. [/INST] </s> 
                  [INST] Question: {question} 
                  Context: {context} 
                  Answer: [/INST]
                  """
              )
      
          def ingest(self, pdf_file_path: str):
              docs = PyPDFLoader(file_path=pdf_file_path).load()
              chunks = self.text_splitter.split_documents(docs)
              chunks = filter_complex_metadata(chunks)
      
              self.vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
              self.retriever = self.vector_store.as_retriever(
                  search_type="similarity_score_threshold",
                  search_kwargs={
                      "k": 3,
                      "score_threshold": 0.5,
                  },
              )
      
              self.chain = (
                  {"context": self.retriever, "question": RunnablePassthrough()}
                  | self.prompt
                  | self.model
                  | StrOutputParser()
              )
      
          def ask(self, query: str):
              if not self.chain:
                  return "Please, add a PDF document first."
      
              return self.chain.invoke(query)
      
          def clear(self):
              self.vector_store = None
              self.retriever = None
              self.chain = None
``` 

**chatpdf_app.py**
``` 
      import os
      import tempfile
      import streamlit as st
      from streamlit_chat import message
      from chat_pdf import ChatPDF
      
      st.set_page_config(page_title="ChatPDF")
      
      
      def display_messages():
          st.subheader("Chat")
          for i, (msg, is_user) in enumerate(st.session_state["messages"]):
              message(msg, is_user=is_user, key=str(i))
          st.session_state["thinking_spinner"] = st.empty()
      
      
      def process_input():
          if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
              user_text = st.session_state["user_input"].strip()
              with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
                  agent_text = st.session_state["assistant"].ask(user_text)
      
              st.session_state["messages"].append((user_text, True))
              st.session_state["messages"].append((agent_text, False))
      
      
      def read_and_save_file():
          st.session_state["assistant"].clear()
          st.session_state["messages"] = []
          st.session_state["user_input"] = ""
      
          for file in st.session_state["file_uploader"]:
              with tempfile.NamedTemporaryFile(delete=False) as tf:
                  tf.write(file.getbuffer())
                  file_path = tf.name
      
              with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
                  st.session_state["assistant"].ingest(file_path)
              os.remove(file_path)
      
      
      def page():
          if len(st.session_state) == 0:
              st.session_state["messages"] = []
              st.session_state["assistant"] = ChatPDF()
      
          st.header("ChatPDF")
      
          st.subheader("Upload a document")
          st.file_uploader(
              "Upload document",
              type=["pdf"],
              key="file_uploader",
              on_change=read_and_save_file,
              label_visibility="collapsed",
              accept_multiple_files=True,
          )
      
          st.session_state["ingestion_spinner"] = st.empty()
      
          display_messages()
          st.text_input("Message", key="user_input", on_change=process_input)
      
      
      if __name__ == "__main__":
          page()
``` 
