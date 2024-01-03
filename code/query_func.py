import json
from paper_func import *
from llm_tools import *
from paper_func import _get_collection_papers
from paper_func import _get_papercollection_by_name
from paper_func import _get_paper_content
from utils import logger, json2string

uid = 'test_user' 

ANSWER_FILE="/data/survey_agent/query_full_answer.json"

def merge_chunk_responses(responses, question, model_type="small"):    
    
        # f = open(f"../prompts/merge_answer.txt", "r")
        # all_res = responses
        # merge_template = f.read()
        # res_l=0
        # while len(all_res)>1:
        #     res_l+=1
        #     responses_batch, ids = get_batches(all_res, prompt_text=merge_template)
        #     batch_prompts=[]
        #     for res, id in zip(responses_batch, ids):
        #         prompt = merge_template.format(responses=format_res(res), question=self.query)
        #         batch_prompts.append(prompt)
        #     all_res = predict(batch_prompts)
        # if res_l>1:
        #     print("all res level: ", res_l)


        # merged_response = all_res[0]
            
        # return merged_response
        pass

def save_answer(query, full_response):
    """
    save full response information for one specific query

    Args:
        query (str): The query to be answered.
        full_response: A list of dictionaries, each dictionary is {"source": paper content, "paper": paper name, "answer": answer from this chunk}
    """
    data_to_append = {str(query):full_response}
    try:
        with open(ANSWER_FILE, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}
    
    data.update(data_to_append)
    
    with open(ANSWER_FILE, "w") as file:
        json.dump(data, file)
    
    
def read_chunked_papers(paper_list_name: str, question: str, content_type="abstract", model_type="small") -> str:
    """
    Query a large collection of papers (based on their abstracts) to find an answer to a specific query.

    Args:
        paper_list_name (str): The name of the paper collection.
        query (str): The query to be answered.
        content_type (str): choosing from "abstract", "intro", "full".
        model_type (str): choosing from "small" or "large"

    Returns:
        str: A string representing a list of dictionaries containing the answer, the source paragraph, and the source paper.
    """

    paper_list_name = _get_papercollection_by_name(paper_list_name, uid)
    if paper_list_name == COLLECTION_NOT_FOUND_INFO:
        return COLLECTION_NOT_FOUND_INFO
    
    collection_papers = _get_collection_papers(paper_list_name, uid)

    paper_contents = [ {'content':_get_paper_content(paper_name, content_type), 'source': paper_name} for paper_name in collection_papers]

    chunk_list=[]
    for p in paper_contents:
        content = p['content']
        chunks = get_chunks(content) # Generally, one abstract in one chunk
        for c in chunks:
            chunk_list.append((p["source"],c))

    #check for relevant chunks => paper name and paragraph content
    f = open(f"../prompts/check_for_related.txt", "r")
    check_for_related = f.read()
    prompts=[]
    for d in chunk_list:
        prompts.append(check_for_related.format(title=d[0], content=d[1], question=question))
        
    res = small_model_predict(prompts)


    #parse for references, answers
    answer_and_source=[]
    f = open(f"../prompts/collect_answer_from_chunk.txt", "r")
    query_chunk = f.read()
    query_chunk_prompts=[]
    
    #Only return related chunks
    for i,j in zip(res,chunk_list):
        if "yes" in i.lower():
            leave={}
            leave['source_content'] = j[1]
            leave['source_paper'] = j[0]
            answer_and_source.append(leave)
            query_chunk_prompts.append(query_chunk.format(chunk=j[1], question=question))
    if "small" in model_type:
        answers = small_model_predict(query_chunk_prompts)
    else:
        answers = []
        for mess in prompts:
            answers.append(gpt_4_predict(mess))
    
    answer_for_agent = []
    for i,j in zip(answers, answer_and_source):
        j['answer'] = i
        t = {}
        t["source_paper"]=answer_and_source["source_paper"]
        t['answer']=i
        answer_for_agent.append(t)
    
    #save whole responses for UI
    save_answer(question, answer_and_source)

    #merge for final complete answer if needed

    return answer_for_agent

def read_whole_papers(paper_list_name, query, uid, content_type="abstract", model_type="small"):
    """
    Queries a small collection of papers (based on their full text) to find an answer to a specific query.

    Args:
        paper_list_name (str): The name of the paper collection.
        query (str): The query to be queried.
        content_type (str): choosing from "abstract", "intro", "full".
        model_type: either "small" or "large", "small" means local models, "large" means GPT-4

    Returns:
        str: A string representing a list of tuples containing the answer, the source paragraph, and the source paper.
    """

    paper_list_name = _get_papercollection_by_name(paper_list_name, uid)
    if paper_list_name == COLLECTION_NOT_FOUND_INFO:
        return COLLECTION_NOT_FOUND_INFO
    
    collection_papers = _get_collection_papers(paper_list_name, uid)

    #Get the full context of the paper
    paper_contents = [ {'content':_get_paper_content(paper_name, content_type), 'source': paper_name} for paper_name in collection_papers]
    
    
    #Assume we can read all the papers at once using long-context model
    def _filter_full_text(full_text):
        full_text = full_text[:full_text.find('REFERENCES')]
        return full_text        
            
    whole_paper_content = ""
    source_list = []
    for paper in paper_contents:
        whole_paper_content = whole_paper_content+_filter_full_text(paper['content'])+"\n\n\n"
        source_list.append(paper["source"])
        
    f = open(f"../prompts/collect_answer_from_whole_paper.txt", "r")
    query_paper = f.read()
    answer_with_source=[]

    prompt = query_paper.format(paper=whole_paper_content, question=query)

    if "small" in model_type:
        res = small_model_predict([prompt])[0]
    else:
        res = gpt_4_predict(prompt)
        
    leave = {
        "answer": res,
        "source_paper": source_list,
    }
    answer_for_agent = [leave]
    leave['source_content'] = whole_paper_content
    
    #Save for UI
    save_answer(query, [leave])
    
    return answer_for_agent

#Assume use on 130 or 135, you should connect to the huggingface
import os
os.environ["https_proxy"]="http://127.0.0.1:7890"
os.environ["http_proxy"]="http://127.0.0.1:7890"

def _query_papers(paper_list_name, query, uid, content_type, model_type="small", chunk=False ):
    # content type: ["abstract", "intro", "full"]
    
    #try to follow instructions
    try:
        #
        if chunk:
            res = read_chunked_papers(paper_list_name, query, uid, content_type, model_type)
        else:
            res = read_whole_papers(paper_list_name, query, uid, content_type, model_type)
    except:
        print("try to read whole paper failed, retry to read chunked papers...")
        res = read_chunked_papers(paper_list_name, query, uid, content_type, model_type)
    return res

def query_area_papers(paper_list_name: str, question: str) -> str:
    """
    Query a large collection of papers (based on their abstracts) to find an answer to a specific query.

    Args:
        paper_list_name (str): The name of the paper collection.
        query (str): The query to be answered.

    Returns:
        str: A string representing a list of dictionaries containing the answer, the source paragraph, and the source paper.
    """

    paper_list_name = _get_papercollection_by_name(paper_list_name, uid)
    if paper_list_name == COLLECTION_NOT_FOUND_INFO:
        return COLLECTION_NOT_FOUND_INFO
    
    collection_papers = _get_collection_papers(paper_list_name, uid)

    paper_contents = [ {'content':_get_paper_content(paper_name, 'abstract'), 'source': paper_name} for paper_name in collection_papers]
    
    # ....

    # call _display_papers to display the reference information
    # reference_info = _display_papers(...)

    # chunk the large collection of papers into chunks
    chunk_list=[]
    for p in paper_contents:
        content = p['content']
        chunks = get_chunks(content) # Generally, one abstract in one chunk
        for c in chunks:
            chunk_list.append((p["source"],c))

    #check for relevant chunks => paper name and paragraph content
    f = open(f"../prompts/check_for_related.txt", "r")
    check_for_related = f.read()
    prompts=[]
    for d in chunk_list:
        prompts.append(check_for_related.format(title=d[0], content=d[1], question=question))
    res = small_model_predict(prompts)

    #parse for references, answers
    answer_and_source=[]
    f = open(f"../prompts/collect_answer_from_chunk.txt", "r")
    query_chunk = f.read()
    query_chunk_prompts=[]
    
    #Only return related chunks
    for i,j in zip(res,chunk_list):
        if "yes" in i.lower():
            leave={}
            #leave['source_content'] = j[1]
            leave['source_paper'] = j[0]
            answer_and_source.append(leave)
            query_chunk_prompts.append(query_chunk.format(chunk=j[1], question=question))
    answers = small_model_predict(query_chunk_prompts)
    for i,j in zip(answers, answer_and_source):
        j['answer'] = i

    #merge for final complete answer if needed

    return answer_and_source

def query_individual_papers(paper_list_name, query):
    """
    Queries a small collection of papers (based on their full text) to find an answer to a specific query.

    Args:
        paper_list_name (str): The name of the paper collection.
        query (str): The query to be queried.

    Returns:
        str: A string representing a list of tuples containing the answer, the source paragraph, and the source paper.
    """

    paper_list_name = _get_papercollection_by_name(paper_list_name, uid)
    if paper_list_name == COLLECTION_NOT_FOUND_INFO:
        return COLLECTION_NOT_FOUND_INFO
    
    collection_papers = _get_collection_papers(paper_list_name, uid)

    #Get the full context of the paper
    paper_contents = [ {'content':_get_paper_content(paper_name, 'full'), 'source': paper_name} for paper_name in collection_papers]
    
    
    #Assume we can read all the papers at once using long-context model
            
            
    whole_paper_content = ""
    source_list = []
    for paper in paper_contents:
        whole_paper_content = whole_paper_content+paper['content']+"\n\n\n"
        source_list.append(paper["source"])
        
        
    f = open(f"../prompts/collect_answer_from_whole_paper.txt", "r")
    query_paper = f.read()
    answer_with_source=[]

    prompt = query_paper.format(paper=whole_paper_content, question=query)
    res = gpt_4_predict(prompt)
    leave = {
        "answer": res,
        "source_paper": source_list,
        #"source_content": whole_paper_content
    }
    answer_with_source.append(leave)

    return answer_with_source

if __name__ == '__main__':
    uid = 'test_user'   
    res = _query_papers(paper_list_name='persona', query='summarize these papers', uid=uid, content_type="full")
    # res = _query_papers(paper_list_name='123 asd Papers', query='summarize these papers', uid=uid, content_type="abstract")
    print(res)
    