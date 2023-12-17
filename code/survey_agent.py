# Load config
import configparser

config = configparser.ConfigParser()
config.read("./secrets.ini")

# Set API Key
import os
os.environ.update({"OPENAI_API_KEY": config["OPENAI"]["OPENAI_API_KEY"]})

# Set Cache for LLM
from langchain.globals import set_llm_cache
from langchain.llms import OpenAI
from langchain.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# To make the caching really obvious, lets use a slower model.
llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2)
import time
t = time.time()
llm.predict("Tell me a joke")
print(time.time() - t)
t = time.time()

llm.predict("Tell me a joke")
print(time.time() - t)