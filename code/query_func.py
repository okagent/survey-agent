
from paper_func import *
from llm_tools import *
from utils import logger, json2string

uid = 'test_user' 

def merge_chunk_responses(responses, question):    
    #merge all the chunk answers if needed
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
        prompts.append(check_for_related.format(title=d[0], content=d[1]))
    res = small_model_predict(prompts)

    #parse for references, answers
    answer_and_source=[]
    f = open(f"../prompts/collect_answer_from_chunk.txt", "r")
    query_chunk = f.read()
    query_chunk_prompts=[]
    for i,j in zip(res,chunk_list):
        if "yes" in i.lower():
            leave={}
            leave['source_content'] = j[1]
            leave['source_paper'] = j[0]
            answer_and_source.append(leave)
            query_chunk_prompts.append(query_chunk.format(chunk=j[1], question=question))
    answers = small_model_predict(query_chunk_prompts)
    for i,j in zip(answers, answer_and_source):
        j['answer'] = i

    #merge for final complete answer if needed

    return answer_and_source

def query_individual_papers(paper_list_name, query, uid):
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

    #Todo: how to get the full context of the paper?
    paper_contents = [ {'content':_get_paper_content(paper_name, 'abstract'), 'source': paper_name} for paper_name in collection_papers]
    
    # ....

    # call _display_papers to display the reference information
    # reference_info = _display_papers(...) 

    # Assume we can get a list of paper names
    paper_list = get_paperlist_by_name(paper_list_name)
    
    # chunk the large collection of papers into chunks
    chunk_list=[]
    for p in paper_list:
        content = _get_paper_content(p ,mode="short")
        chunks = get_chunks(content)
        for c in chunks:
            chunk_list.append((p,c))
            
    #check for relevant chunks => paper name and paragraph content
    f = open(f"../prompts/check_for_related.txt", "r")
    check_for_related = f.read()
    prompts=[]
    for d in chunk_list:
        prompts.append(check_for_related.format(title=d[0], content=d[1]))
    res = small_model_predict(prompts)
    
    #parse for references, answers
    answer_and_source=[]
    f = open(f"../prompts/collect_answer_from_chunk.txt", "r")
    query_chunk = f.read()
    query_chunk_prompts=[]
    for i,j in zip(res,chunk_list):
        if "yes" in i.lower():
            leave={}
            leave['source_content'] = j[1]
            leave['source_paper'] = j[0]
            answer_and_source.append(leave)
            query_chunk_prompts.append(query_chunk.format(chunk=j[1], question=question))
    answers = small_model_predict(query_chunk_prompts)
    for i,j in zip(answers, answer_and_source):
        j['answer'] = i
        
    #merge for final complete answer if needed
    
    return answer_and_source
    


if __name__ == '__main__':
    uid = 'test_user'   
    query_area_papers(paper_list_name='123 asd Papers', query='summarize this papers', uid=uid)
    # Assume we can get a list of paper names
    paper_list = get_paperlist_by_name(paper_list_name)
    
    # Assume we can put all the content of one paper into the model
    f = open(f"../prompts/collect_answer_from_whole_paper.txt", "r")
    query_paper = f.read()
    answer_with_source=[]
    for p in paper_list:
        paper_content = _get_paper_content(p, mode="full")
        prompt = query_paper.format(content=paper_content, question=question)
        res = gpt_4_predict(prompt)
        leave = {
            "answer": res,
            "source_paper": p,
            "source_content": paper_content
        }
        answer_with_source.append(leave)
    
    #return answer_with_source
