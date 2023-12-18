from paper_func import _define_paper_collection, _display_papers, _get_papercollection_by_name, COLLECTION_NOT_FOUND_INFO
from paper_func import paper_corpus
import random 
import string
from utils import logger, json2string

uid = 'test_user' 

def _arxiv_sanity_search(uid, search_query, search_type, time_filter):
    # search_type: "by_keywords"表示按关键词搜索，"by_collections"表示按论文列表推荐

    # 基本就把arxiv-sanity-lite里serve.py的def main()里的代码拿过来就行了，或者直接调用它的main()

    ## search for papers
    if search_type == 'search':
        request = {'opt_rank': 'search', 'q': search_query, 'time_filter': time_filter}
    elif search_type == 'recommend':
        request = {'opt_rank': 'tags', 'tags': search_query, 'time_filter': time_filter, 'skip_have': True}
    
    ## _call_arxiv_sanity_search(request)
    
    # 需要维护paper_collections和arxiv-sanity-lite后端的tags一致
    
    pass



    # placeholder: 随便返回一些paper
    found_papers = random.sample(paper_corpus.keys(), 3)


    # Define the search result as a paper collection
    
    random_str = ''.join(random.choice(string.ascii_letters + string.digits) for i in range(5))
    paper_collection_name = f'{random_str}, {search_type} results of {search_query}'
    _define_paper_collection(found_papers, paper_collection_name, uid)

    return _display_papers(found_papers)


def search_papers(query: str, time_filter: str = '') -> str:
    """
    Searches for papers based on a given query. Optionally filter papers that were published 'time_filter' days ago.

    Args:
        query (str): The search query used to find relevant papers.
        time_filter (str, optional): Filter papers that were publised 'time_filter' days ago. Defaults to an empty string (no time filtering).

    Returns:
        str: A JSON string representing the search result papers.
    """
    return json2string(_arxiv_sanity_search(uid, query, search_type="search", time_filter=time_filter))

def recommend_similar_papers(collection_name: str, time_filter: str = '') -> str:
    """
    Recommends papers similar to those in a specified collection. Optionally filter papers that were published 'time_filter' days ago.

    Args:
        collection_name (str): The name of the paper collection based on which recommendations are to be made.
        time_filter (str, optional): Filter papers that were publised 'time_filter' days ago. Defaults to an empty string (no time filtering).

    Returns:
        str: If the collection is found, returns a JSON string representing recommended papers.
             If the collection is not found, returns an error message.
    """
    collection_name = _get_papercollection_by_name(collection_name, uid)
    if collection_name == COLLECTION_NOT_FOUND_INFO:
        return COLLECTION_NOT_FOUND_INFO
    else:
        return json2string(_arxiv_sanity_search(uid, collection_name, search_type="recommend", time_filter=time_filter))

if __name__ == '__main__':
    uid = 'test_user'  
    print('Search Papers: \n', search_papers(uid, 'persona of LLMs'))

    print('Recommend Papers: \n', recommend_similar_papers(uid, 'persona of LLMs'))
    # Sorry, we cannot find the paper collection you are looking for.
    print('Recommend Papers: \n', recommend_similar_papers(uid, '123 asd Papers'))