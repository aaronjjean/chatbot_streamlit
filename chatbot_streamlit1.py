import tiktoken
import pandas as pd
import numpy as numpy
import pandas as pd 
import openai
from tqdm.auto  import tqdm
import time
import pinecone
from dotenv import load_dotenv
import os
load_dotenv(override=True)


#old
from langchain.embeddings.openai import OpenAIEmbeddings
#new
#from langchain_community.openai import OpenAIEmbeddings
#from langchain.vectorstores import pinecone
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import OpenAI

from langchain_community.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
OPENAI_KEY = os.getenv('OPENAPI_KEY')
PINECONE_KEY=os.getenv('PINECONE_KEY')
GENAI_MODEL ='gpt-3.5-turbo'
EMBEDDING_MODEL="text-embedding-ada-002"
PINECONE_ENV="gcp-starter"
PINECONE_INDEX_NAME="default" # this will be created below
dtype={'id':str}
df = pd.read_csv("updated_rwh-agri-hydro-skilling.csv",dtype=dtype)
filtered_df = df.loc[df["sub"].isin(['Rainwater_Harvesting','Agriculture','Hydropower','Skilling'])]
print(filtered_df["sub"].value_counts())
pinecone.init(api_key=PINECONE_KEY,environment=PINECONE_ENV)
index_list= pinecone.list_indexes()

if len(index_list) == 0:
    print("Creating index...")
    pinecone.create_index(PINECONE_INDEX_NAME, dimension=1536, metric='dotproduct')
    
print(pinecone.describe_index(PINECONE_INDEX_NAME))
index = pinecone.Index(PINECONE_INDEX_NAME)  

# This references the text-embedding-ada-002 OpenAI model we'll use to create embeddings 
# Both for indexing ground knowledge content, and later when searching ground knowledge
# For RAG documents to include in LLM Prompts

embed = OpenAIEmbeddings(model = EMBEDDING_MODEL,openai_api_key=OPENAI_KEY)

# This is a for loop to create embeddings for each of the Detroit & London knowledge articles, and 
# Then add the embeddings and orgiional article text to the vector databse
# Shout-out to Dr. KM Moshin for this code snippet from his Excellent Udemy course on Pinecone!

# batch_size = 10

# for i in tqdm(range(0, len(filtered_df), batch_size)):
#     # OpenAPI has rate limits, and we use batches to slow the pace of embedding requests
#     i_end = min(i+batch_size, len(filtered_df))
#     batch = filtered_df.iloc[i:i_end]
    
#     # When querying the Vector DB for nearest vectors, the metadata 
#     # is what is returned and added to the LLM Prompt (the "Grounding Knowledge")
#     meta_data = [{"subject" : row['sub'], 
#               "context": row['context']} 
#              for i, row in batch.iterrows()]
    
#     # Get a list of documents to submit to OpenAI for embedding  
#     docs = batch['context'].tolist() 
#     emb_vectors = embed.embed_documents(docs) 

#     # The original ID keys are used as the PK in the Vector DB
#     ids = batch['id'].tolist()
    
#     # Add embeddings, associated metadata, and the keys to the vector DB
#     to_upsert = zip(ids, emb_vectors, meta_data)    
#     index.upsert(vectors=to_upsert)
    
#     # Pause for 10 seconds after each batch to avoid rate limits
#     time.sleep(10) 


vectorstore = Pinecone(index, embed, "context")
# query = "what are the iissues with hydropower?" #ask some question that's answerable with the content added to the Vector DB

# x= vectorstore.similarity_search(query, k=1)
# print(x)

llm = ChatOpenAI(openai_api_key =OPENAI_KEY,model_name=GENAI_MODEL,temperature=0.1)

# """
# print("llm model name" + llm.model_name)
# print("filtered dataframe" + filtered_df.head())
# print("embed model name:" embed.model)
# print("pinecone index name: " index)
# """
print("---------------------------------------------------------------------------------")

#to ensure the chat session includes a memory of 5 previous messages 
conv_mem=ConversationBufferWindowMemory(memory_key='hisotry',k=5,return_messages=True)

#create a cain to manage chat sessions
qa = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=vectorstore.as_retriever())

#a=qa.run('how is IOT utilised in agriculture in India?')
#print(a)

#print(qa.invoke('how is IOT used in agri-food'))
#b=qa.run('give me the basic issues with Hydropower in india in 5 points')
# """
# while True:
#     question = str(input("enter your question: "))
#     ans=qa.run(question)
#     print(ans)
#     print("****************************************************************************************************************************")
#     print("****************************************************************************************************************************\n")
#     print("press:1 for another question or 0 to exit")
#     x = int(input())
#     if x==1:
#         continue
#     else:
#         break

# """
# """
# from flask import Flask, render_template, request, jsonify
# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index2.html')

# @app.route('/getresponse', methods=['POST'])
# def get_bot_response():
#     user_input = request.form['user_query']
#     #print("user input received:",user_input)
#     # Here, you integrate your chatbot code and get the response
#     response = qa.run(user_input)
#     #print("respones to be sent",response)
#     # Replace with your chatbot function
#     return response


# if __name__ == '__main__':
#     app.run(debug=True)
# """

import streamlit as st
# Import your chatbot's backend functions
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []


st.title('Chatbot')

user_input = st.text_input("Type your message here:")

if user_input:
    #get chatbot response
    response = qa.run(user_input)
    #update conversation history
    # st.session_state.conversation_history.append(("You", user_input))
    response = qa.run(user_input)
    # st.session_state.conversation_history.append(("Chatbot", response))
    st.text_area("Chatbot says:", value=response, height=200, disabled=True)
    # st.experimental_rerun()
    
if st.button("Clear Conversation"):
    st.session_state.conversation_history = []