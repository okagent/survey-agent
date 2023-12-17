from paper_func import _get_paper_content, _get_papercollection_by_name, _get_collection_papers, _display_papers

def query_area_papers(paper_list_name, query, uid):
    
    """
    Queries a large collection of papers to find an answer to a specific query.

    Args:
    paper_list_name (str): The name of the paper collection.
    query (str): The query to be queried.

    Returns:
    list: A list of tuples containing the answer, the source paragraph, and the source paper.
    """
    # TODO: Implement the logic to query a large collection of papers
    paper_collection = _get_papercollection_by_name(paper_list_name, uid)
    collection_papers = _get_collection_papers(paper_collection, uid)
    paper_contents = [ {'content':_get_paper_content(paper_name, 'short'), 'source': paper_name} for paper_name in collection_papers]
    
    # ....

    # call _display_papers to display the reference information
    # reference_info = _display_papers(...)

def query_individual_papers(paper_list_name, query, uid):

    """
    Queries a small collection of papers to find an answer to a specific query.

    Args:
    paper_list_name (str): The name of the paper collection.
    query (str): The query to be queried.

    Returns:
    list: A list of tuples containing the answer, the source paragraph, and the source paper.
    """
    # TODO: Implement the logic to query a small collection of papers
    pass

    paper_collection = _get_papercollection_by_name(paper_list_name, uid)
    collection_papers = _get_collection_papers(paper_collection, uid)
    paper_contents = [ {'content':_get_paper_content(paper_name, 'full'), 'source': paper_name} for paper_name in collection_papers]
    # call _display_papers to display the reference information
    # reference_info = _display_papers(...)

if __name__ == '__main__':
    uid = 'test_user'   
    query_area_papers(paper_list_name='123 asd Papers', query='summarize this papers', uid=uid)