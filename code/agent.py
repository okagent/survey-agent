# Load config
import datetime
print("="*10 + f"准备开始 - 时间1: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "="*10 )
from utils import config

# Set API key
import os

os.environ.update({"OPENAI_API_KEY": config["openai_apikey"],"GOOGLE_API_KEY": config["gemini_apikey"]})
#os.environ['http_proxy'] = "http://10.176.64.118:33333"
#os.environ['https_proxy'] = "http://10.176.64.118:33333"
# import openai
# openai.api_key = config["openai_apikey"]
# print(os.environ["OPENAI_API_KEY"])
print('openai key: {}'.format(os.environ["OPENAI_API_KEY"]))
print('gemini key: {}'.format(os.environ["GOOGLE_API_KEY"]))

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
from paper_func import get_papers_and_define_collections, get_papercollection_by_name, get_paper_content, get_paper_metadata, update_paper_collection, retrieve_from_papers, a_get_papers_and_define_collections, a_get_papercollection_by_name, a_get_paper_content, a_get_paper_metadata, a_update_paper_collection, a_retrieve_from_papers
print("="*10 + f"准备开始 - 时间4: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "="*10 )
from arxiv_sanity_func import search_papers, recommend_similar_papers, a_search_papers, a_recommend_similar_papers
print("="*10 + f"准备开始 - 时间5: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "="*10 )
from query_func import query_based_on_paper_collection, a_query_based_on_paper_collection
print("="*10 + f"准备开始 - 时间6: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "="*10 )
from langchain.tools import StructuredTool
from langchain.callbacks import HumanApprovalCallbackHandler

# Define which tools the agent can use to answer user queries
# Schema will be automatically inferred by StructuredTool Class

callbacks = [] #[HumanApprovalCallbackHandler()]

tools = [
    StructuredTool.from_function(
        func=get_papers_and_define_collections,
        coroutine=a_get_papers_and_define_collections,
        description='''This function processes a list of paper titles, matches them with corresponding entries in the database, and defines a collection of papers under a specified name.
        Note that:
            1. If certain papers are not found, do not attempt to use the search_papers function again to look for those papers. 
            2. Only use this function when the user inputs a list of papar titles. Do not use it without explicit intention from the user.
        ''',
        callbacks=callbacks
    ),
    StructuredTool.from_function(
        func=get_papercollection_by_name,
        coroutine=a_get_papercollection_by_name,
        description='''Retrieve a specified paper collection by its name, display the paper collection's name and information of its papers. Only use this function when the user explicitly asks for information about the collection. Avoid using this when the user poses a request about the collection, in which case the agent should use 'query_based_on_paper_collection' instead.
        ''',
        callbacks=callbacks
    ),
    StructuredTool.from_function(
        func=get_paper_content,
        coroutine=a_get_paper_content,
        description="Retrieve the content of a paper. Set 'mode' as 'full' for the full paper, or 'abstract' for the abstract.",
        callbacks=callbacks
    ),
    StructuredTool.from_function(
        func=get_paper_metadata,
        coroutine=a_get_paper_metadata,
        description="Retrieve the metadata of a paper, including its title, authors, year and url.",
        callbacks=callbacks
    ),
    StructuredTool.from_function(
        func=update_paper_collection,
        coroutine=a_update_paper_collection,
        description='''Updates the target paper collection based on a specified action ('add' or 'del') and paper indices (Indices start from 0. The format should be comma-separated, with ranges indicated by a dash, e.g., "0, 2-4") from the source collection.''',
        callbacks=callbacks
    ),
    StructuredTool.from_function(
        func=retrieve_from_papers,
        coroutine=a_retrieve_from_papers,
        description="Retrieve the most relevant content in papers based on a given query, using the BM25 retrieval algorithm. Output the relevant paper and content. This function should be used when the query is about a specific statement, rather than being composed of keywords.",
        callbacks=callbacks
    ),
    StructuredTool.from_function(
        func=search_papers,
        coroutine=a_search_papers,
        description='''Searches for papers based on a given query. Optionally filter papers that were published 'time_filter' days ago.
        The query should consist of keywords rather than a complete paper title. If the user's input seems like a paper title, the agent should use 'get_papers_and_define_collections'.''',
        callbacks=callbacks,
    ), 
    StructuredTool.from_function(
        func=recommend_similar_papers,
        coroutine=a_recommend_similar_papers,
        description='''Recommends papers similar to those in a specified collection. Optionally filter papers that were published 'time_filter' days ago.
        Note that:
        1. Only use this function when the user explicitly asks for recommendation.
        ''',
        callbacks=callbacks,
    ),
    StructuredTool.from_function(
        func=query_based_on_paper_collection,
        coroutine=a_query_based_on_paper_collection,
        description="""
        When the user poses a question or request concerning a specific paper collection, the agent should use this action to generate the answer. This action includes the 'get_papercollection_by_name' function. Therefore, the agent should call this action directly instead of first invoking 'get_papercollection_by_name'.
        Note that:
        1. 'content_type' denotes which part of the papers would be used to answer the query. Choose from "abstract", "intro" or "full" for the abstract, introduction or the full text of the papers respectively.
        2. 'model_type' denotes which kinds of LLMs would be used to answer the query. Use "large" by default to use Gemini-pro, or use "small" for smaller open-source models if specified by the user.
        3. 'chunk' denotes applying the 'chunk-and-merge' algorithm. Set it as False by default unless it is specified by the user. 
        4. If the user-specified paper collection is not found, the agent should finish this round and wait for user instructions.""",
        callbacks=callbacks
    ),
]

'''
StructuredTool.from_function(
        func=query_area_papers,
        description="Query a large collection of papers (based on their abstracts) to find an answer to a specific query. If the user-specified paper collection is not found, the agent should finish this round and wait for user instructions.",
        callbacks=callbacks
    ),
    StructuredTool.from_function(
        func=query_individual_papers,
        description="Query a collection of papers (based on their full texts) to find an answer to a specific query. If the user-specified paper collection is not found, the agent should finish this round and wait for user instructions.",
        callbacks=callbacks
    )
'''
# 135
# f = open(f"/data/survey_agent/prompts/tool_using.txt", "r")
# 130
f = open(f"{config['data_path']}/prompts/tool_using.txt", "r")
tool_using_example = f.read()


template = """
You are Survey Agent, an AI-driven tool expertly crafted for researchers to facilitate their exploration and analysis of academic literature. With a suite of advanced functions, you excel in organizing, retrieving and recommending research papers, and answering questions based on these papers.
    
As Survey Agent, you serve as a vital assistant to  researchers, simplifying the task of navigating through the extensive and complex domain of academic literature, and delivering tailored, relevant, and accurate insights. In a nutshell, you should always answer the user's academic queries as best you can.

You shoulde use tools for paper retrieval, paper collection management, paper recommendation, and question answering. Don\'t answer it yourself if you can use a tool to answer it. Specifically, you have access to the following tools:

{tools}

For single parameter input, please input directly; for multiple parameter input, please use dict format to input.

Here are some simple examples to tell you when to use which tools:

{tool_using_example}

Use the following format:

Query: the input query for which you must provide a natural language answer
Thought: you should always think about what to do, step by step
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action. For boolean parameters, use lowercase (true / false). 
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times.)
Thought: I now know the final answer
Final Answer: the final answer to the original input question (do not repeat large blocks of content that is present in the Observation.)

{chat_history}

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
    
    def _chat_history_input(self, chat_history):
        windows_size = 6
        chat_history_input = ''

        if len(chat_history) <= windows_size:
            for chat in chat_history:
                if type(chat) == HumanMessage:
                    chat_history_input += 'Query: ' + chat.content + '\n'
                elif type(chat) == AIMessage:
                    chat_history_input += chat.content[chat.content.find('Final Answer: '):] + '\n'
        else:
            for chat in chat_history[-windows_size:]:
                if type(chat) == HumanMessage:
                    chat_history_input += 'Query: ' + chat.content.replace('\n\n','\n').strip() + '\n'
                elif type(chat) == AIMessage:
                    chat_history_input += chat.content[chat.content.find('Final Answer: '):].replace('\n\n','\n').strip() + '\n'
            
        return chat_history_input

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        # chat_history = kwargs.pop("chat_history")
        kwargs["chat_history"] = self._chat_history_input(kwargs["chat_history"])
        
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
        kwargs["tool_using_example"] = tool_using_example

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
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, streaming=True)
# llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Agent and agent executor
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
).with_config({"run_name": "Agent"})

from utils import _sync_chat_history
from langchain_core.messages import AIMessage, HumanMessage
# set up chat history 
chat_history_dict = _sync_chat_history()

from utils import DualOutput
import sys
sys.stdout = DualOutput('output.log')
chat_history = []


async def run_agent(query, uid=None, session_id=None):

    
    chat_history = chat_history_dict.get((uid, session_id), [])
    
    chunks = []
    import pprint
    try:
        #output = agent_executor.invoke({"input": query, "chat_history": chat_history})

        async for chunk in agent_executor.astream(
            {"input": query, "chat_history": chat_history}
        ):
            chunks.append(chunk)
            messages = chunk['messages']
            if not len(message) == 1:
                import pdb; pdb.set_trace()
            message = messages 
            is_action = 'actions' in chunk 
            is_observation = 'steps' in chunk 

            # print("------")
            # pprint.pprint(chunk, depth=1)

            if is_observation:
                message = 'Observation:' + message
            message += '\n\n'

            yield message



        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Error: ", e)
        yield e

    #response = '\n\n'.join([ step_info[0].log + '\n\nObservation:' + str(step_info[1]) for step_info in output['intermediate_steps'] ] + [output['output']]) 

    #  output['intermediate_steps'] = [(AgentAction(tool='retrieve_from_papers', tool_input='query: "large language models', log='Thought: The user has asked to find some papers about large language models using the "retrieve_from_papers" function. I will use this function with the query provided.\n\nAction: retrieve_from_papers\nAction Input: query: "large language models"'), <coroutine object retrieve_from_papers at 0x7f0c131ba2d0>), (AgentAction(tool='retrieve_from_papers', tool_input='query: "large language models', log='It seems there was an issue with the retrieval process. I will attempt the action again to see if it resolves the problem.\n\nAction: retrieve_from_papers\nAction Input: query: "large language models"'), <coroutine object retrieve_from_papers at 0x7f0c131b9e70>)]

    '''
    {'actions': [AgentAction(tool='search_papers', tool_input={'query': 'large language models'}, log='Thought: First, I need to check if there is already a collection on large language models. If not, I will use the search function to find related papers.\n\nAction: search_papers\nAction Input: {"query": "large language models"}')], 'messages': [AIMessage(content='Thought: First, I need to check if there is already a collection on large language models. If not, I will use the search function to find related papers.\n\nAction: search_papers\nAction Input: {"query": "large language models"}')]}


    '''
    import pdb
    pdb.set_trace()
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

    #return response

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
    # async def main():
    #     import datetime
    #     import random
    #     session_id = str(random.randint(0, 100000))
        
    #     query = input("Please enter your query: ")
    #     while 'stop' not in query.lower():
    #         try:
    #             print("="*10 + f"测试开始 - 时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "="*10 )
    #             # import pdb; pdb.set_trace()
    #             response = await run_agent(query, session_id=session_id) 
    #             print(response)
    #         finally:
    #             print("\n\n\n" + "="*10 + f"测试结束 - 时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "="*10 )
            
    #         query = input("Please enter your query: ")

    def prettify_response(text):
        prettified_text = ""
        for line in text.split("\n"):
            if line.startswith("Thought:"):
                prettified_text += "<p style='color:#065f46'>" + line + "</p>"
            elif line.startswith("Action:"):
                prettified_text += "<p style='color:#b91c1c'>" + line + "</p>"
            elif line.startswith("Action Input:"):
                prettified_text += "<p style='color:#4338ca'>" + line + "</p>"
            elif line.startswith("Observation:"):
                observation = line.split("Observation:")[1].strip()
                try:
                    observation = json.loads(eval(observation))
                    prettified_text += (
                        "<p style='color:#92400e'>Observation:</p>\n\n```json\n"
                        + json.dumps(observation, indent=4)
                        + "\n```\n\n"
                    )
                except:
                    prettified_text += "<p style='color:#92400e'>" + line + "</p>"
            else:
                prettified_text += line
            prettified_text += "\n"
        return prettified_text

    import asyncio
    # asyncio.run(main())
    def my_generate(question):
        def run_model():
            print('run model')
            import pdb; pdb.set_trace()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
 

            try:
                generated_text = run_agent(
                    question,
                    uid="test_user",
                    session_id=None
                )
                import pdb; pdb.set_trace()
                text = prettify_response(generated_text)
            except Exception as e:
                text = "Error: " + str(e)
            # fetch 'leave' if it exists @shiwei

            yield "data:" + json.dumps(
                {
                    "token": {"id": -1, "text": "", "special": False, "logprob": 0},
                    "generated_text": text,
                    "details": None,
                }
            ) + "\n"
        
        xx = run_model()
        import pdb; pdb.set_trace()
        
        return xx 
    
    query = input("Please enter your query: ")
    while 'stop' not in query.lower():
        try:
            print("="*10 + f"测试开始 - 时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "="*10 )
            # import pdb; pdb.set_trace()
            response = my_generate(query) 
            print(response)
        finally:
            print("\n\n\n" + "="*10 + f"测试结束 - 时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "="*10 )
        
        query = input("Please enter your query: ")