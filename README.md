# RAG-Project-using-Open-Source-LLM-Models-And-Groq-Inferencing

# End To End Advanced RAG Project using Open Source LLM Models And Groq Inferencing
 - In this project I have built an end to end advanced RAG project using open source llm model, Mistral using groq inferencing engine.

![Groq for RAG Image](./images/chatbot-image.png)

## DEMO
 - You can try the project live [here](https://8510-01hwj8ynshjz7spkr595x77ec2.cloudspaces.litng.ai/)

## Description
- This project showcase the implementation of an advanced RAG system that uses groq as an llm to retrieve information about langsmith.

I just built an end-to-end retrieval system using LangChain, and let me walk you through how I did it! 🚀

Step 1: Loading the Data 📥
First things first—I needed some data. So, I grabbed the WebBaseLoader from langchain_community.document_loaders and loaded content straight from LangChain’s official docs.

from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()

Step 2: Chunking the Data 📦
Since LLMs work best with smaller chunks, I used RecursiveCharacterTextSplitter to break the text into chunks of 1000 characters.
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

Step 3: Storing Vector Embeddings 🔍
To make my system searchable, I converted the text chunks into embeddings using HuggingFaceInstructEmbeddings and stored them in a FAISS vector database.

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings

embeddings = HuggingFaceInstructEmbeddings()
vector_db = FAISS.from_documents(chunks, embeddings)
retriever = vector_db.as_retriever()


Step 4: Setting Up the LLM 🤖
For the language model, I chose ChatGroq with the powerful Mixtral-8x7B-32768 model.

from langchain.chat_models import ChatGroq

llm = ChatGroq(model_name="mixtral-8x7b-32768")


Step 5: Crafting the Prompt 🎭
A good LLM setup needs a great prompt, so I used ChatPromptTemplate to guide the conversation.

from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Answer the following question: {query}")


Step 6: Chaining Everything Together 🔗
Finally, I combined everything using RetrievalQA to connect my retriever to the document chain.

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

And that's it! 🎉 Now, I can ask questions, and my system will fetch the most relevant information from the LangChain docs using retrieval-augmented generation (RAG).

## Libraries Used
 - langchain==0.1.20
 - langchain-community==0.0.38
 - langchain-core==0.1.52
 - langchain-groq==0.1.3
 - faiss-cpu==1.8.0
 - python-dotenv

## Installation
 1. Prerequisites
    - Git
    - Command line familiarity
 2. Clone the Repository: `git clone https://github.com/NebeyouMusie/End-To-End-Advanced-RAG-Project-using-Open-Source-LLM-Models-And-Groq-Inferencing.git`
 3. Create and Activate Virtual Environment (Recommended)
    - `python -m venv venv`
    - `source venv/bin/activate`
 4. Navigate to the projects directory `cd ./End-To-End-Advanced-RAG-Project-using-Open-Source-LLM-Models-And-Groq-Inferencing` using your terminal
 5. Install Libraries: `pip install -r requirements.txt`
 6. run `streamlit run app.py`
 7. open the link displayed in the terminal on your preferred browser

## Collaboration
- Collaborations are welcomed ❤️

## Acknowledgments
 - I would like to thank [Krish Naik](https://www.youtube.com/@krishnaik06)
   
## Contact
 - LinkedIn: [Karishma Shaik](https:(https://www.linkedin.com/in/shaik-karishma2004/))
 - Gmail: karishmashaik2802@gmail.com

