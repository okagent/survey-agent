# Load config
import datetime
print("="*10 + f"准备开始 - 时间1: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "="*10 )
from utils import config

# Set API key
import os
os.environ.update({"OPENAI_API_KEY": config["openai_apikey"]})
import openai
openai.api_key = config["openai_apikey"]
print(os.environ["OPENAI_API_KEY"])


# Set up cache for LLM
print("="*10 + f"准备开始 - 时间2: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "="*10 )
from langchain.globals import set_llm_cache
from langchain.llms import OpenAI
from langchain.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# Build our agent
import re
import json
from typing import List, Union, Tuple

from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    LLMSingleActionAgent,
    Tool,
)
from langchain.chains import LLMChain
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
print("="*10 + f"准备开始 - 时间3: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "="*10 )
from paper_func import get_papers_and_define_collections, get_papercollection_by_name, get_paper_content, get_paper_metadata, update_paper_collection, retrieve_papers
print("="*10 + f"准备开始 - 时间4: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "="*10 )
from arxiv_sanity_func import search_papers, recommend_similar_papers
print("="*10 + f"准备开始 - 时间5: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "="*10 )
from query_func import query_area_papers, query_individual_papers 
print("="*10 + f"准备开始 - 时间6: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "="*10 )
from langchain.tools import StructuredTool
from langchain.callbacks import HumanApprovalCallbackHandler

# Define which tools the agent can use to answer user queries
# Schema will be automatically inferred by StructuredTool Class

callbacks = [] #[HumanApprovalCallbackHandler()]

tools = [
    StructuredTool.from_function(
        func=get_papers_and_define_collections,
        description="This function processes a list of paper titles, matches them with corresponding entries in the database, and defines a collection of papers under a specified name.",
        callbacks=callbacks
    ),
    StructuredTool.from_function(
        func=get_papercollection_by_name,
        description="Retrieve a specified paper collection by its name, display the paper collection's name and information of its papers.",
        callbacks=callbacks
    ),
    StructuredTool.from_function(
        func=get_paper_content,
        description="Retrieve the content of a paper. Set 'mode' as 'full' for the full paper, or 'abstract' for the abstract.",
        callbacks=callbacks
    ),
    StructuredTool.from_function(
        func=get_paper_metadata,
        description="Retrieve the metadata of a paper, including its title, authors, year and url.",
        callbacks=callbacks
    ),
    StructuredTool.from_function(
        func=update_paper_collection,
        description="This function updates the collection of papers under a specified name.",
        callbacks=callbacks
    ),
    StructuredTool.from_function(
        func=update_paper_collection,
        description='''Updates the target paper collection based on a specified action ('add' or 'del') and paper indices (The format should be comma-separated, with ranges indicated by a dash, e.g., "1, 3-5") from the source collection.''',
        callbacks=callbacks
    ),
    StructuredTool.from_function(
        func=retrieve_papers,
        description="Retrieve the most relevant content in papers based on a given query, using the BM25 retrieval algorithm. Output the relevant paper and content.",
        callbacks=callbacks
    ),
    StructuredTool.from_function(
        func=search_papers,
        description="Searches for papers based on a given query. Optionally filter papers that were published 'time_filter' days ago.",
        callbacks=callbacks
    ), 
    StructuredTool.from_function(
        func=recommend_similar_papers,
        description="Recommends papers similar to those in a specified collection. Optionally filter papers that were published 'time_filter' days ago.",
        callbacks=callbacks
    ),
    StructuredTool.from_function(
        func=query_area_papers,
        description="Query a large collection of papers (based on their abstracts) to find an answer to a specific query.",
        callbacks=callbacks
    ),
    StructuredTool.from_function(
        func=query_individual_papers,
        description="Query a collection of papers (based on their full texts) to find an answer to a specific query.",
        callbacks=callbacks
    ),
]




template = """
You are Survey Agent, an AI-driven tool expertly crafted for researchers to facilitate their exploration and analysis of academic literature. With a suite of advanced functions, you excel in organizing, retrieving and recommending research papers, and answering questions based on these papers.
    
As Survey Agent, you serve as a vital assistant to  researchers, simplifying the task of navigating through the extensive and complex domain of academic literature, and delivering tailored, relevant, and accurate insights. In a nutshell, you should always answer the user's academic queries as best you can.

You shoulde use tools for paper retrieval, paper collection management, paper recommendation, and question answering. Specifically, you have access to the following tools:

{tools}

For single parameter input, please input directly; for multiple parameter input, please use dict format to input.
Use the following format:

Query: the input query for which you must provide a natural language answer
Thought: you should always think about what to do, step by step
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}
"""

## Maybe we need to add some examples later, cause ReAct is based on few-shot settings. 
## This design is based on question: whether we want Survey Agent to use multiple tools for each user query?

# Prompts
## Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        chat_history = kwargs.pop("chat_history")
        
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        
        return self.template.format(**kwargs)



prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps", "chat_history"],
)

# Output parser
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output}, #{"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        try:
            tool_input = json.loads(action_input.strip(" ").strip('"'))
        except:
            tool_input = action_input.strip(" ").strip('"')
        return AgentAction(
            tool=action, tool_input=tool_input, log=llm_output
        )

output_parser = CustomOutputParser()

# Specify the LLM model
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)

#Agent and agent executor
# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, return_intermediate_steps=True
)

from utils import _sync_chat_history
from langchain_core.messages import AIMessage, HumanMessage
# set up chat history 
chat_history_dict = _sync_chat_history()

from utils import DualOutput
import sys
sys.stdout = DualOutput('output.log')

def run_agent(query, uid=None, session_id=None):
    chat_history = chat_history_dict.get((uid, session_id), [])

    output = agent_executor.invoke({"input": query, "chat_history": chat_history})

    response = '\n\n'.join([ step_info[0].log + '\n\nObservation:' + str(step_info[1]) for step_info in output['intermediate_steps'] ] + [output['output']]) 
    ans = output['output'].split("Final Answer:")[-1].strip()

    # relevant_info = ... @shiwei
    chat_history.extend(
        [
            HumanMessage(content=query),
            AIMessage(content=response),
        ]
    )
    chat_history_dict[(uid, session_id)] = chat_history
    _sync_chat_history(chat_history_dict)

    return response, ans

'''

from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
# Define the agent
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

agent_executor = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

'''


if __name__ == "__main__":
    import datetime
    
    query = input("Please enter your query: ")
    while 'stop' not in query.lower():
        try:
            print("="*10 + f"测试开始 - 时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "="*10 )
            # import pdb; pdb.set_trace()
            response, ans = run_agent(query) 
        finally:
            print("\n\n\n" + "="*10 + f"测试结束 - 时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "="*10 )
        query = input("Please enter your query: ")
