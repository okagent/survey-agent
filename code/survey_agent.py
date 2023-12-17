# Load config
import configparser

config = configparser.ConfigParser()
config.read("./secrets.ini")

# Set API key
import os
os.environ.update({"OPENAI_API_KEY": config["OPENAI"]["OPENAI_API_KEY"]})

# Set up cache for LLM
from langchain.globals import set_llm_cache
from langchain.llms import OpenAI
from langchain.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# Build our agent
import re
from typing import List, Union

from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    LLMSingleActionAgent,
    Tool,
)
from langchain.chains import LLMChain
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from query_func import get_papers_and_define_collections, get_papercollection_by_name, get_paper_content, get_paper_metadata, update_paper_collection
from arxiv_sanity_func import search_papers, recommend_similar_papers
from query_func import query_area_papers, query_individual_papers 
from langchain.tools import StructuredTool

# Define which tools the agent can use to answer user queries
tools = [
    Tool(
        name="ItemLookup",
        func=(lambda x: vocab_lookup(x, entity_type="item")),
        description="useful for when you need to know the q-number for an item",
    ),
    Tool(
        name="PropertyLookup",
        func=(lambda x: vocab_lookup(x, entity_type="property")),
        description="useful for when you need to know the p-number for a property",
    ),
    Tool(
        name="SparqlQueryRunner",
        func=run_sparql,
        description="useful for getting results from a wikibase",
    ),
]