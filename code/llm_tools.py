from langchain.text_splitter import CharacterTextSplitter

#tokenizer is load from specific model, assume we use openchat-3.5
model = "openchat/openchat_3.5"
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model)

def get_chunks(story, separator = ". ", chunk_size=1000):
    
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer, chunk_size=chunk_size, chunk_overlap=200, separator=separator,
    )
    text_chunks = text_splitter.split_text(story)
    return text_chunks
def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    # """Returns the number of tokens in a text string."""
    num_tokens = len(tokenizer.encode(string))
    return num_tokens
import requests
import json
#Assume we use openchat-3.5
def small_model_predict(prompt_list, max_tokens=1024):
    url = "http://localhost:18888/v1/chat/completions"
    # The code you provided is making a POST request to a chatbot API. It is sending a list of
    # messages as input to the chatbot and receiving a response. Here's a breakdown of what the
    # code is doing:
    res_list=[]
    for mess in prompt_list:
        data = {
            "model": "openchat_3.5",
            "temperature":0,
            "messages": [{"role": "user", "content": mess}]
        }

        # Make the POST request
        response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(data))

        # Check if the request was successful
        if response.status_code == 200:
            # print("Response:", response.json())
            print("prompt: ", mess)
            print("res: ", response.json()["choices"][0]["message"]["content"])
            res_list.append(response.json()["choices"][0]["message"]["content"])
            # print("prompt: ", mess)
            # print("res:", response.json()["choices"][0]["message"]["content"])
        else:
            print("Error:", response.status_code, response.text)
            
    return res_list
    


import os
#os.environ['OPENAI_API_KEY']="key"
#os.environ["https_proxy"]="http://127.0.0.1:7890"
#os.environ["http_proxy"]="http://127.0.0.1:7890"
from langchain.chat_models import ChatOpenAI
def gpt_4_predict(prompt):
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)
    return llm.predict(prompt)
    