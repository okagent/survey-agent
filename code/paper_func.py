import difflib
import pdb 
import random
from utils import logger

PAPER_NOT_FOUND_INFO = "Sorry, we cannot find the paper you are looking for."
COLLECTION_NOT_FOUND_INFO = "Sorry, we cannot find the paper collection you are looking for."
RETRIEVE_NOTHING_INFO = "Sorry, we retrieve no relevant paper for your query."
ERRORS = [PAPER_NOT_FOUND_INFO, COLLECTION_NOT_FOUND_INFO]

# load paper_corpus.json
import json
'''
paper_corpus_path='../data/arxiv_full_papers.json'
with open(paper_corpus_path, 'r') as f:
    paper_corpus_json = json.load(f)[0]
    paper_corpus = { p['title']:p for p in random.sample(paper_corpus_json, 20)}

json.dump(paper_corpus, open('../data/sample_papers.json', 'w'))
'''
# load sample_papers instead
with open('../data/sample_papers.json', 'r') as f:
    paper_corpus = json.load(f)

def _sync_paper_collections(paper_collections=None):
    """Synchronize/Load paper collections with the database."""

    paper_collections_path = '../data/paper_collections.json'
    if paper_collections:
        with open(paper_collections_path, 'w') as f:
            json.dump(paper_collections, f)
    else:
        with open(paper_collections_path, 'r') as f:
            paper_collections = json.load(f)
    return paper_collections

# load paper collections data
paper_collections = _sync_paper_collections()

def get_papers_and_define_collections(paper_titles, paper_collection_name, uid):
    # Input a list of papers, match with papers in our database, and define as a paper collection
    found_papers = _get_papers_by_name(paper_titles)
    _define_paper_collection(found_papers, paper_collection_name, uid)
    return _display_papers(found_papers, user_inputs=paper_titles)

def _get_paper_content(paper_name, mode):
    """Get text content of a paper based on its exact name."""
    if paper_name in paper_corpus.keys():
        if mode == 'full':
            return paper_corpus[paper_name]['full_text'] 
        else: # == 'short':
            return paper_corpus[paper_name]['abstract'] 
    else:
        return PAPER_NOT_FOUND_INFO

def get_paper_content(paper_name, mode):
    return _get_paper_content(paper_name, mode)

def _get_paper_metadata(paper_name):
    """Get metadata of a paper based on its exact name."""
    if paper_name in paper_corpus.keys():
        return { key: paper_corpus[paper_name][key] for key in ['title', 'authors', 'year', 'url'] }
    else:
        return PAPER_NOT_FOUND_INFO

def get_paper_metadata(paper_name):
    return _get_paper_metadata(paper_name)
    
def _get_papers_by_name(paper_titles):
    """Find corresponding papers based on a list of fuzzy paper names."""
    found_papers = []
    for fuzzy_name in paper_titles:
        matches = difflib.get_close_matches(fuzzy_name, paper_corpus.keys(), n=1, cutoff=0.8)
        # 这个matches还挺慢的，@鑫凤 可不可以优化一下？
       
        if matches:
            found_papers.append(matches[0])
        else:
            found_papers.append(None)  # Append None if no matching paper is found
    # log relevant information
    #found_papers = [p for p in found_papers if p]  # Remove None from the list
    logger.info(f"Found {len([p for p in found_papers if p])} papers out of {len(paper_titles)}")
    
    return found_papers

def _display_papers(paper_titles, user_inputs=None):
    """Display paper information based on a list of exact paper names."""
    paper_info = []
    if user_inputs:
        # called by get_papers_and_define_collections, where there might be None in paper titles
        for paper_name, user_input_name in zip(paper_titles, user_inputs):
            if paper_name:
                paper_info.append(_get_paper_metadata(paper_name))
                paper_info[-1]['authors'] = ', '.join(paper_info[-1]['authors'])
            else:
                paper_info.append({'title': user_input_name, 'status': PAPER_NOT_FOUND_INFO})
    else:
        # called by other functions, where paper titles are all valid
        for paper_name in paper_titles:
            paper_info.append(_get_paper_metadata(paper_name))
            paper_info[-1]['authors'] = ', '.join(paper_info[-1]['authors'])
    return json.dumps(paper_info, indent=4) 

def _define_paper_collection(found_papers, paper_collection_name, uid):
    """Define a paper list based on a list of exact paper names."""
    found_papers = [p for p in found_papers if p]  # Remove None from the list

    paper_collections.setdefault(uid, {}) 
    if paper_collection_name in paper_collections[uid]:
        # paper_collection_name already exists, use a random name instead
        import string 
        import random 
        new_paper_collection_name = ''.join(random.choice(string.ascii_letters + string.digits) for i in range(5))
        logger.info(f"Paper collection name {paper_collection_name} already exists, use a random name {new_paper_collection_name} instead.")
        paper_collection_name = new_paper_collection_name

    paper_collections[uid][paper_collection_name] = found_papers
    # log relevant info
    logger.info(f"Paper collection {paper_collection_name} created for user {uid} with {len(found_papers)} papers.")
    _sync_paper_collections(paper_collections)
    return True 

    
def _get_papercollection_by_name(collection_name, uid):
    """Find the name of the paper collection that best matches a fuzzy collection name."""
    # Find the closest match for the collection name
    match = difflib.get_close_matches(collection_name, paper_collections[uid].keys(), n=1, cutoff=0.7)
    if match:
        return match[0]
    else:    
        return COLLECTION_NOT_FOUND_INFO


def _get_collection_papers(collection_name, uid):
    paper_titles = paper_collections[uid][collection_name]
    return paper_titles

def get_papercollection_by_name(collection_name, uid):
    paper_collection_name = _get_papercollection_by_name(collection_name, uid)
    if paper_collection_name == COLLECTION_NOT_FOUND_INFO:
        return COLLECTION_NOT_FOUND_INFO
    else:
        collection_papers = _get_collection_papers(paper_collection_name, uid)[:3]
        return {'Collection': paper_collection_name, 'Papers': _display_papers(collection_papers)}

def update_paper_collection(target_collection_name, source_collection_name, paper_indexes, action, uid):
    _target_collection_name = _get_papercollection_by_name(target_collection_name, uid)
    if _target_collection_name:
        target_collection_name = _target_collection_name 

    
    source_collection_name = _get_papercollection_by_name(source_collection_name, uid)
    if source_collection_name == COLLECTION_NOT_FOUND_INFO:
        logger.error(f"Source collection {source_collection_name} does not exist.")
        return False

    """Update target paper collection based on provided action and paper numbers from the source collection. """
    # Convert string paper_indexes to actual list of indices
    indices = []
    for part in paper_indexes.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    
    # sort and remove duplicates
    indices = sorted(list(set(indices)))

    # Get the current list of papers for the user
    paper_collections.setdefault(uid, {})
    target_collection = paper_collections.get(uid, {}).get(target_collection_name, [])
    try:
        source_collection = paper_collections[uid][source_collection_name]
    except:
        raise ValueError(f"Source collection {source_collection_name} does not exist.")
    
    update_papers = [source_collection[i] for i in indices]
    target_collection_set = set(target_collection)
    
    # Perform the add or delete action
    if action == "add":
        target_collection += [p for p in update_papers if p not in target_collection_set]
    elif action == "del":
        target_collection = [p for p in target_collection if p not in update_papers]

    paper_collections[uid][target_collection_name] = target_collection
    _sync_paper_collections(paper_collections)
    return True


from langchain.retrievers import BM25Retriever
from langchain.schema import Document

from langchain.docstore.document import Document

paper_docs = [] #[ Document(page_content=p['full_text'], metadata={k:p[k] for k in ['title']})]
for title, p in paper_corpus.items():
    page_content = p['full_text']
    # 将p['full_text']划分为多个段落，每个段落1000个字符，naive
    page_content_pieces = [page_content[i:i+1000] for i in range(0, len(page_content), 1000)]

    paper_docs.extend([Document(page_content=page_content_piece, metadata={**{k:p[k] for k in ['title']}, **{'ith_piece': i}}) for i, page_content_piece in enumerate(page_content_pieces)])

retriever = BM25Retriever.from_documents(paper_docs)

def retrieve_papers(query):
    result = retriever.get_relevant_documents(query)
    if len(result) > 0:
        return result[0]
    else:
        raise None


if __name__ == '__main__':
    uid = 'test_user' 

    
    print('get_papers_and_define_collections: ', get_papers_and_define_collections(paper_titles=["Semantic Relation Classification via Bidirectional LSTM Networks with Entity-aware Attention using", 'Robust Numerical Question Answering: Diagnosing Numerical Capabilities of NLP', 'Does Role-Playing Chatbots Capture the Character Personalities? Assessing Personality Traits for Role-Playing'], paper_collection_name='Paper Collection 123',uid=uid))
    

    print('get_papercollection_by_name: ', get_papercollection_by_name("Paper Collection ", uid=uid))
    
    print('retrieve_papers: ', retrieve_papers('''what is Numerical Question Answering?''').page_content[:100])
    
    print('_get_paper_content: ', _get_paper_content('Towards Robust Numerical Question Answering: Diagnosing Numerical Capabilities of NLP Systems', mode='short'))

    print('update_paper_collection ', update_paper_collection('123 asd Papers', 'Paper Collection 123', '1-2', 'del', uid))
    


    