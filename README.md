# CS-5391
Weekly progress updates for our CS 5391 Project exploring Retrieval Augmented Generation and Large Language Models.

## Week 3

**Starting Over from Scratch**
* Asked ChatGPT how to create a vector database on my local machine
* These were the steps provided: choose a vector database library, install required libraries, prepare your data, prepare your dataset of vectors, create and train the index , perform searches, save and load the index
* I was able to successfully build the RAG and run it locally using Langchain, Ollama, and Streamlit based on the article in week 1, now I’m going to adopt that code to create a local generative AI search engine using RAG
* Made some code modifications as recommended by chat GPT
* Created a requirements.txt file, created a virtual environment, installed requirements, and upgraded pip
* I ran the search engine, which resulted in a Streamlit tab in my browser
* I realized that for this project I don't actually need to be using Streamlit right now, so I decided to start from scratch with starter code from chatGPT
* When testing the code I had an error with RAGChain, I needed to import it from rag and not langchain_community
* I got Hugging Face and NVIDIA API Keys when working off the article a little bit ago and decided to work with the HF one for this project
* I created an environment_var.sh file with my API key, I had to %chmod +x environment_var.sh and %source environment_var.sh before running the code
* I separated the code out into four scripts: text_extraction.py, indexer.py, search_engine.py, and model_utils.py
* When I ran the entire search engine, I got a segmentation fault
* So, I isolated ad tested each individual component: text extraction, loading models, embedding generation, indexing
* I reintegrated each component after making some changes and confirming that they worked to try and pinpoint the issue
* I ran the code through the python debugger and experimented with creating a new virtual environment and reinstalling all modules
* When I tested each component individually they worked fine, I can load the models, extract text from different file types, generate embeddings, and create a FAISS index
* But, for some reason when I run the search engine altogether there is a seg fault

**Next Steps**
* Pinpoint the reason for the segmentation fault
* Reseach Dot Product Similarity (vs. Cosine Similarity)
* Watch video on how to build agents: https://youtu.be/AxnL5GtWVNA?si=wnxBmJmKh46SjpTr
* Look into Google Colab and agent setup (give it a spreadsheet, ask it to summarize)
* Consider how the concept of agents affects the future of coding and teaching CS?

### Code

**text_extraction.py**

Extracts readable text from .docx, .pdf, .pptx, and .xlsx file types
``` 
      import PyPDF2
      import pptx
      import openpyxl
      from docx import Document
      
      def read_docx(file_path):
          doc = Document(file_path)
          return " ".join([para.text for para in doc.paragraphs if para.text])
      
      def read_pdf(file_path):
          with open(file_path, "rb") as file:
              reader = PyPDF2.PdfReader(file)
              text = [page.extract_text() for page in reader.pages if page.extract_text()]
          return " ".join(text)
      
      def read_pptx(file_path):
          ppt = pptx.Presentation(file_path)
          text = []
          for slide in ppt.slides:
              for shape in slide.shapes:
                  if hasattr(shape, "text") and shape.text:
                      text.append(shape.text)
          return " ".join(text)
      
      def read_xlsx(file_path):
          workbook = openpyxl.load_workbook(file_path, data_only=True)
          text = []
          for sheet in workbook:
              for row in sheet.iter_rows(values_only=True):
                  if any(row):
                      text.append(" ".join([str(cell) for cell in row if cell is not None]))
          return " ".join(text)

      
      
``` 

**indexer.py**

Manages indexing and retrieval of documents using embeddings
``` 
      from model_utils import get_embeddings_model, get_embeddings
      import numpy as np
      import faiss
      import json
      
      def create_faiss_index(embeddings):
          dimension = embeddings.shape[1]
          index = faiss.IndexFlatL2(dimension)
          index.add(embeddings)
          return index
      
      def index_documents(texts, model_name):
          model = get_embeddings_model(model_name)
          embeddings = get_embeddings(texts, model)
          faiss_index = create_faiss_index(embeddings)
          faiss.write_index(faiss_index, "document_embeddings.faiss")
          print("Index created and saved.")
      
      
``` 

**search_engine.py**

Acts as the main interface, orchestrating the whole search engine process
``` 
      import os
      import faiss
      import numpy as np
      import torch
      from text_extraction import read_docx, read_pdf, read_pptx, read_xlsx
      from model_utils import get_embeddings_model, get_embeddings
      
      def create_faiss_index(embeddings):
          print("Creating FAISS index...")
          if isinstance(embeddings, torch.Tensor):
              embeddings = embeddings.cpu().numpy()  # Convert to CPU and NumPy array
          dimension = embeddings.shape[1]
          index = faiss.IndexFlatL2(dimension)
          index.add(embeddings)
          print("FAISS index created successfully.")
          return index
      
      def main():
          print("Prompting for directory path...")
          directory_path = input("Enter the directory path to index: ")
          print("Loading embedding model...")
          model = get_embeddings_model('sentence-transformers/msmarco-bert-base-dot-v5')
          print("Model loaded successfully.")
      
          texts = []
          print(f"Collecting texts from {directory_path}...")
          for root, dirs, files in os.walk(directory_path):
              for file in files:
                  file_path = os.path.join(root, file)
                  if file.endswith('.docx'):
                      texts.append(read_docx(file_path))
                  elif file.endswith('.pdf'):
                      texts.append(read_pdf(file_path))
                  # Include other file types similarly
      
          print("Generating embeddings...")
          embeddings = get_embeddings(texts, model)
          print("Embeddings generated successfully. Shape:", embeddings.shape)
      
          print("Creating FAISS index...")
          index = create_faiss_index(embeddings)
      
          print("System ready for queries or further processing.")
      
      if __name__ == "__main__":
          main()
```

**model_utils.py**

Handles all operations related to loading models and generating text embeddings
```
      from sentence_transformers import SentenceTransformer
      import faiss
      import torch
      from transformers import AutoModel, AutoTokenizer  # Import additional Hugging Face libraries
      
      def get_embeddings_model(model_name):
          """Retrieve a Sentence Transformer model based on the model name."""
          model = SentenceTransformer(model_name)
          return model
      
      def get_embeddings(texts, model):
          """Generate embeddings for a list of texts using a provided model."""
          return model.encode(texts, convert_to_tensor=True)
      
      def generate_response(query):
          """Placeholder function for generating responses using LLaMA3."""
          return "Response generation based on LLaMA3 needs to be implemented."
      
      def get_model_and_tokenizer(model_name):
          """Retrieves a general Hugging Face model and its corresponding tokenizer based on a given model name."""
          tokenizer = AutoTokenizer.from_pretrained(model_name)
          model = AutoModel.from_pretrained(model_name)
          return model, tokenizer 
```

##

## Week 2

**Article Notes: How to build a generative search engine for your local files using Llama3**
* https://towardsdatascience.com/how-to-build-a-generative-search-engine-for-your-local-files-using-llama-3-399551786965
* System design: an index with the context of the local files and an information retrieval engine to retrieve the most relevant documents, a language model to use selected content from local documents and generate a summarized answer, a user interface 
* Basic process: index local files, when a user asks a questions you’d use embeddings or something to find the most relevant documents, then pass the user query and documents to the LLM which would use the content of the documents to generate answers 
* Use Qdrant as vector storage
* Asymmetric search problems: when you have short queries and long documents proper information retrieval may fail
* This model uses dot product, cosine similarity could work but it only focuses on difference in angles, and dot product consider angle and magnitude
* BERT like models have limited context size, so you can either consider the first 512 tokens of the document and ignore the rest or chunk the document and store it in the index (using prebuilt chunker from LangChain)
* File retrieval is recursive, the user defines a folder they want to index, the indexer retrieves all files from that folder and subfolders recursively and indexes supported file types (PDF, Word, PPT, TXT)
* Will use different python libraries to read through the documents
* Use TokenTextSplitter from LangChain to create chunks of 500 tokens with 50 token overlap 
* Create a web service using FastAPI to host the generative search engine
* API will access the Qdrant client with the indexed data, perform a search for vector similarity, use the top chunks to generate an answer with Llama 3, and return an answer
* Create user interface using Streamlit
* Note: his version was designed on a M2 Mac with 32GB RAM, my laptop is M1 and 16GB RAM, so need to use 8B parameter model

##

**Working Off of Nikola's Project**
* Forked his Github: https://github.com/nikolamilosevic86/local-genAI-search
* Installed all requirements: %cd local-gen-search, %pip install -r requirements.txt
* Got NVIDIA and Hugging Face API Keys
* Requested access to Llama3 model on Hugging Face
* Generated sample files to index
* Ran index.py on the test files: %python3 index.py TestFolder
* This created a qdrant client index locally and indexed all files in the folder and its subfolders with extensions pdf, txt, docx, pptx
* Indexing was succesful!
* Ran the generative search service: %python uvicorn_start.py
* Got a bunch of errors because I didn't create an environment_var.py file with my HF and NVIDIA API keys
* Another error, needed to install: %pip install -U langchain-huggingface
* Then modified index.py and api.py to import from langchain_huggingface
* Tried to run the generative search service, but there were a BUNCH of error
* Example: "storage folder qdrant/ is already accessed by another instance of Qdrant client. If you require concurrent access, use Qdrant server instead"

**Change of Plans!**
* Discussed with Dr. C at our weekly meeting
* Instead of trying to work off of Nikola's project, I'm going to make my own search service
* The idea is to start simple with basic functionality and then move from there
* Plan: use sentence-transformers to make embeddings, use FAISS for vector storage and querying, use LLaMA3 to generate responses

**Progress on Local Search AI**
* Setup: ensured that PyPDF2, python-docx, and openpyxl were installed
* Created a virtual environment: %python3 -m venv env, %source env/bin/activate, %pip install -U pip
* Decided to build the project in components: document_processor, embeddings, query, response_generator, and search_engine
* I used ChatGPT and Copilot to get a start on each of the scripts and used PyCharm as an editor
* Created a test.py file to debug and check the search engine
* Had a problem with installing fais, I needed to do %pip install faiss-cpu 
* Needed to install packages in the virtual environment: %pip install numpy, pypdf2, docx, openpyxl, faiss-cpu, sentence-transformers
* Had lots of errors, instead of having one overarching test file I'm going to create test scripts for each component of the project

### Code

**document_processor.py**

Scans directories to identify documents, extracts text from documents, and categorizes documents based on type.
``` 
      # this script scans the user's directories for documents, categorizes them, and extracts text
      
      import os
      import json
      from PyPDF2 import PdfReader
      import docx
      import openpyxl
      
      
      def scan_directories(base_path):
          documents = {'pdf': [], 'word': [], 'excel': []}
          for root, _, files in os.walk(base_path):
              for file in files:
                  path = os.path.join(root, file)
                  if file.endswith('.pdf'):
                      documents['pdf'].append(path)
                  elif file.endswith('.docx'):
                      documents['word'].append(path)
                  elif file.endswith('.xlsx'):
                      documents['excel'].append(path)
          return documents
      
      
      def extract_text_from_document(path):
          if path.endswith('.pdf'):
              reader = PdfReader(path)
              return ''.join([page.extract_text() for page in reader.pages])
          elif path.endswith('.docx'):
              doc = docx.Document(path)
              return ''.join([para.text for para in doc.paragraphs])
          elif path.endswith('.xlsx'):
              wb = openpyxl.load_workbook(path)
              sheet = wb.active
              return ''.join([str(cell.value) for row in sheet.iter_rows() for cell in row])
      
      
      base_path = '/Users/zm/Desktop/CS5391/TestFiles'
      documents = scan_directories(base_path)
      
      # example of categorizing based on simple keyword in path
      categorized_documents = {course: [extract_text_from_document(path) for path in paths]
                               for course, paths in documents.items()}
      
      with open('categorized_documents.json', 'w') as f:
          json.dump(categorized_documents, f)
``` 


**embeddings.py**

Converts document text into numerical vectors (embeddings) using the sentence-transformers pre-trained models, normalizes the embeddings to unit length for cosine similarity, stores the generated normalized embeddings into a FAISS (Facebook AI Similarity Search) index, save the FAISS index for later retrieval.
```
      # this script converts documents into vector embeddings using sentence-transformers and stores them in a FAISS index
      
      import numpy as np
      import faiss
      from sentence_transformers import SentenceTransformer
      
      model = SentenceTransformer('all-MiniLM-L6-v2')
      
      
      def normalize_embeddings(embeddings):
          norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
          return embeddings / norms
      
      
      def embed_documents(categorized_documents):
          # Assuming categorized_documents is a dictionary with course names as keys and lists of document texts as values
          embeddings = {course: model.encode(doc_texts) for course, doc_texts in categorized_documents.items()}
          normalized_embeddings = {course: normalize_embeddings(np.array(vecs)) for course, vecs in embeddings.items()}
          return normalized_embeddings
      
      
      def store_embeddings_in_faiss(embeddings):
          dimension = 384  # Assuming 384-dimensional embeddings
          index = faiss.IndexFlatIP(dimension)  # Inner Product (IP) is equivalent to cosine similarity after normalization
          for vecs in embeddings.values():
              index.add(np.array(vecs))
          return index
      
      
      def save_faiss_index(index, index_file):
          faiss.write_index(index, index_file)
```

**query.py**

Queries the FAISS index to retrieve the most relevant documents using cosine similarity according to the user query.
```
      # this script queries the FAISS index based on user input
      
      import faiss
      from embedding import normalize_embeddings, model  # Import the normalize function and the model
      
      INDEX_FILE = 'document_index.faiss'
      
      
      def query_index(query_text, top_k=5):
          index = faiss.read_index(INDEX_FILE)
      
          # Convert query text to embedding
          query_embedding = model.encode([query_text])
      
          # Normalize the query embedding
          query_embedding = normalize_embeddings(query_embedding)
      
          D, I = index.search(query_embedding, top_k)
          return D, I
```

**response_generator.py**

Handles the setup and execution of LLaMA3, uses the pre-trained LLM to generate responses based on the context provided by the document indices.
```
      # this script generates a response to the query by creating a context from the documents and appending that to the query using LLaMA3
      
      from transformers import AutoTokenizer, AutoModelForCausalLM
      
      MODEL_NAME = "huggingface/llama-7b"
      
      
      def generate_response(query, indices):
          tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
          model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
      
          # Construct context from retrieved document indices
          context = ' '.join([categorized_documents[course][i] for i in indices[0]])
      
          input_text = f"Context: {context}\n\nQuery: {query}\nAnswer:"
          input_ids = tokenizer.encode(input_text, return_tensors='pt')
          output = model.generate(input_ids, max_length=200)
      
          return tokenizer.decode(output[0], skip_special_tokens=True)
```

**search_engine.py**

Controls the workflow of the search engine.
```
      # this script controls the main flow of the search engine
      
      
      import os
      import json
      import numpy as np
      from document_processor import scan_directories, extract_text_from_document
      from embeddings import embed_documents, store_embeddings_in_faiss, save_faiss_index
      from query import query_index
      from response_generator import generate_response
      
      # Constants
      BASE_PATH = '/Users/zm/Desktop/CS5391/TestFiles'
      EMBEDDINGS_FILE = 'embeddings.npy'
      INDEX_FILE = 'document_index.faiss'
      QUERY = "Python code for data analysis"
      
      
      def main():
          # Step 1: Document scanning and categorization
          documents = scan_directories(BASE_PATH)
      
          # Simulate categorization based on file extension
          categorized_documents = {course: [extract_text_from_document(path) for path in paths]
                                   for course, paths in documents.items()}
      
          with open('categorized_documents.json', 'w') as f:
              json.dump(categorized_documents, f)
      
          # Step 2: Embedding documents
          embeddings = embed_documents(categorized_documents)
          np.save(EMBEDDINGS_FILE, embeddings)
      
          # Step 3: Store embeddings in FAISS index
          index = store_embeddings_in_faiss(embeddings)
          save_faiss_index(index, INDEX_FILE)
      
          # Step 4: Query the FAISS index
          D, I = query_index(QUERY)
      
          # Step 5: Generate response using LLaMA
          response = generate_response(QUERY, I)
          print(response)
      
      
      if __name__ == "__main__":
          main()
```

**test.py**
```
      # this script tests the search engine
      
      import os
      import unittest
      import json
      import numpy as np
      from document_processor import scan_directories, extract_text_from_document
      from embeddings import embed_documents, store_embeddings_in_faiss, save_faiss_index
      from query import query_index
      from response_generator import generate_response
      
      
      class TestIntegration(unittest.TestCase):
          @classmethod
          def setUpClass(cls):
              cls.base_path = 'test_documents'
              cls.embeddings_file = 'test_embeddings.npy'
              cls.index_file = 'test_document_index.faiss'
              cls.query = "Sample query"
              cls.test_documents = {
                  'pdf': 'sample.pdf',
                  'word': 'sample.docx',
                  'excel': 'sample.xlsx'
              }
              cls.categorized_documents_file = 'test_categorized_documents.json'
      
              # Create test directories and files
              os.makedirs(cls.base_path, exist_ok=True)
              for doc_type, filename in cls.test_documents.items():
                  dir_path = os.path.join(cls.base_path, doc_type)
                  os.makedirs(dir_path, exist_ok=True)
                  file_path = os.path.join(dir_path, filename)
                  with open(file_path, 'w') as f:
                      f.write("This is a test document content for {}.".format(doc_type))
      
          @classmethod
          def tearDownClass(cls):
              # Remove created files and directories
              for root, dirs, files in os.walk(cls.base_path, topdown=False):
                  for name in files:
                      os.remove(os.path.join(root, name))
                  for name in dirs:
                      os.rmdir(os.path.join(root, name))
              os.rmdir(cls.base_path)
              if os.path.exists(cls.embeddings_file):
                  os.remove(cls.embeddings_file)
              if os.path.exists(cls.index_file):
                  os.remove(cls.index_file)
              if os.path.exists(cls.categorized_documents_file):
                  os.remove(cls.categorized_documents_file)
      
          def test_integration(self):
              # Step 1: Document scanning
              documents = scan_directories(self.base_path)
      
              # Simulate categorization based on file extension
              categorized_documents = {course: [extract_text_from_document(path) for path in paths]
                                       for course, paths in documents.items()}
      
              with open(self.categorized_documents_file, 'w') as f:
                  json.dump(categorized_documents, f)
      
              # Step 2: Embedding documents
              embeddings = embed_documents(categorized_documents)
              np.save(self.embeddings_file, embeddings)
      
              # Step 3: Store embeddings in FAISS index
              index = store_embeddings_in_faiss(embeddings)
              save_faiss_index(index, self.index_file)
      
              # Step 4: Query the FAISS index
              D, I = query_index(self.query)
      
              # Step 5: Generate response using LLaMA
              response = generate_response(self.query, I)
              print(response)
      
              # Assertions can be added here to check for expected results
      
      
      if __name__ == '__main__':
          unittest.main()
```

##

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
