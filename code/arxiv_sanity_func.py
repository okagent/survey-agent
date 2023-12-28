import pickle
from paper_func import _define_paper_collection, _display_papers, _get_papercollection_by_name, COLLECTION_NOT_FOUND_INFO, paper_corpus, paper_collections, _get_collection_papers, _get_paper_content
from feature_func import load_features
from utils import convert_to_timestamp, json2string

import os
import time
import random

import numpy as np
from sklearn import svm
import datetime
import string

uid = 'test_user' 

RET_NUM = 25

print("="*10 + f"准备开始 - 时间4.1: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "="*10 )

# paper_meta = {k:{"_time": convert_to_timestamp(v["published_date"])} for k, v in paper_corpus.items()}
paper_meta = {
    k: {"_time": convert_to_timestamp(v["published_date"]) if v["published_date"] != '' else None}
    for k, v in paper_corpus.items()
}

print("="*10 + f"准备开始 - 时间4.2: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + "="*10 )

def search_rank(q: str = ''):
    if not q:
        return [], [] # no query? no results
    qs = q.lower().strip().split() # split query by spaces and lowercase

    match = lambda s: sum(min(3, s.lower().count(qp)) for qp in qs)
    matchu = lambda s: sum(int(s.lower().count(qp) > 0) for qp in qs)
    pairs = []
    for pid, p in paper_corpus.items():
        score = 0.0
        score += 10.0 * matchu(' '.join(p['authors']))
        score += 20.0 * matchu(p['title'])
        score += 1.0 * match(p['abstract'])
        if score > 0:
            pairs.append((score, pid))

    pairs.sort(reverse=True)
    paper_titles = [p[1] for p in pairs]
    scores = [p[0] for p in pairs]
    return paper_titles, scores

def svm_rank(uid, tags: str = '', pid: str = '', C: float = 0.01):
    # tag can be one tag or a few comma-separated tags or 'all' for all tags we have in db
    # pid can be a specific paper id to set as positive for a kind of nearest neighbor search
    if not (tags or pid):
        return [], [], []

    # load all of the features
    features = load_features()
    x, paper_titles = features['x'], features['paper_titles']
    n, d = x.shape
    ptoi, itop = {}, {}
    for i, p in enumerate(paper_titles):
        ptoi[p] = i
        itop[i] = p

    # construct the positive set
    y = np.zeros(n, dtype=np.float32)
    if pid:
        y[ptoi[pid]] = 1.0
    elif tags:
        tags_filter_to = paper_collections[uid].keys() if tags == 'all' else set(tags.split(','))
        for tag, paper_titles in paper_collections[uid].items():
            if tag in tags_filter_to:
                for pid in paper_titles:
                    if pid in ptoi:
                        y[ptoi[pid]] = 1.0

    if y.sum() == 0:
        return [], [], [] # there are no positives?

    # classify
    clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=C)
    clf.fit(x, y)
    s = clf.decision_function(x)
    sortix = np.argsort(-s)
    paper_titles = [itop[ix] for ix in sortix]
    scores = [100*float(s[ix]) for ix in sortix]

    # get the words that score most positively and most negatively for the svm
    ivocab = {v:k for k,v in features['vocab'].items()} # index to word mapping
    weights = clf.coef_[0] # (n_features,) weights of the trained svm
    sortix = np.argsort(-weights)
    words = []
    for ix in list(sortix[:40]) + list(sortix[-20:]):
        words.append({
            'word': ivocab[ix],
            'weight': weights[ix],
        })

    return paper_titles, scores, words

def time_rank():
    ms = sorted(paper_meta.items(), key=lambda kv: (kv[1]['_time'] is not None, kv[1]['_time']), reverse=True)
    tnow = time.time()
    paper_titles = [k for k, v in ms]
    # scores = [(tnow - v['_time'])/60/60/24 for k, v in ms] # time delta in days
    scores = [
        (tnow - v['_time']) / 60 / 60 / 24 if v['_time'] is not None else float('inf')
        for k, v in ms
    ]
    return paper_titles, scores

def random_rank():
    paper_titles = list(paper_meta.keys())
    random.shuffle(paper_titles)
    scores = [0 for _ in paper_titles]
    return paper_titles, scores

def _call_arxiv_sanity_search(uid, rank='time', tags='', pid='', time_filter='', q='', skip_have='no', svm_c='', page_number=1):
    # if a query is given, override rank to be of type "search"
    # this allows the user to simply hit ENTER in the search field and have the correct thing happen
    if q:
        rank = 'search'

    # try to parse opt_svm_c into something sensible (a float)
    try:
        C = float(svm_c)
    except ValueError:
        C = 0.01 # sensible default, i think

    # rank papers: by tags, by time, by random
    words = [] # only populated in the case of svm rank
    if rank == 'search':
        paper_titles, scores = search_rank(q=q)
    elif rank == 'tags':
        paper_titles, scores, words = svm_rank(uid=uid, tags=tags, C=C)
    elif rank == 'pid':
        paper_titles, scores, words = svm_rank(uid=uid, pid=pid, C=C)
    elif rank == 'time':
        paper_titles, scores = time_rank()
    elif rank == 'random':
        paper_titles, scores = random_rank()
    else:
        raise ValueError("rank %s is not a thing" % (rank, ))

    # filter by time
    if time_filter:
        kv = {k:v for k,v in paper_meta.items()} # read all of metas to memory at once, for efficiency
        tnow = time.time()
        deltat = int(time_filter)*60*60*24 # allowed time delta in seconds
        keep = [i for i, pid in enumerate(paper_titles) if kv[pid]['_time'] is not None and (tnow - kv[pid]['_time']) < deltat]
        paper_titles, scores = [paper_titles[i] for i in keep], [scores[i] for i in keep]

    # optionally hide papers we already have
    if skip_have == 'yes':
        have = set().union(*paper_collections[uid].values())
        keep = [i for i,pid in enumerate(paper_titles) if pid not in have]
        paper_titles, scores = [paper_titles[i] for i in keep], [scores[i] for i in keep]

    # crop the number of results to RET_NUM, and paginate
    try:
        page_number = max(1, int(page_number))
    except ValueError:
        page_number = 1

    start_index = (page_number - 1) * RET_NUM # desired starting index
    end_index = min(start_index + RET_NUM, len(paper_titles)) # desired ending index
    paper_titles = paper_titles[start_index:end_index]
    scores = scores[start_index:end_index]

    return paper_titles

def _arxiv_sanity_search(uid, search_query, search_type, time_filter):
    # search_type: "by_keywords"表示按关键词搜索，"by_collections"表示按论文列表推荐

    # 基本就把arxiv-sanity-lite里serve.py的def main()里的代码拿过来就行了，或者直接调用它的main()

    ## search for papers
    if search_type == 'search':
        request = {'rank': 'search', 'q': search_query, 'time_filter': time_filter}
    elif search_type == 'recommend':
        request = {'rank': 'tags', 'tags': f'{uid}-{search_query}', 'time_filter': time_filter, 'skip_have': 'yes'}
    
    # 需要维护paper_collections和arxiv-sanity-lite后端的tags一致

    found_papers = _call_arxiv_sanity_search(uid=uid, **request)
    
    if search_type == 'recommendation':
        # 用GPT-4对abstract再过滤一次

        # 获取用户输入论文的abstract
        source_collection = search_query
        source_collection_papers = _get_collection_papers(source_collection, uid)
        source_paper_contents = [ {'content':_get_paper_content(paper_name, 'abstract'), 'source': paper_name} for paper_name in source_collection_papers]

        # 获取推荐论文的abstract
        target_paper_contents = [ {'content':_get_paper_content(paper_name, 'abstract'), 'source': paper_name} for paper_name in found_papers]

        # 让GPT-4过滤一遍推荐论文。如果GPT-4认为某篇论文不相关，就把它从found_papers里删掉。
        # sanity-check，要确保GPT-4推荐的论文名称和原名称一致。可以再用_get_papers_by_name对齐到原论文。
        pass





    # Define the search result as a paper collection
    
    random_str = ''.join(random.choice(string.ascii_letters + string.digits) for i in range(5))
    paper_collection_name = f'<{random_str}, {search_type} results of "{search_query}">'
    _define_paper_collection(found_papers, paper_collection_name, uid)

    return _display_papers(found_papers, paper_collection_name)


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
        uid (str): The user id.
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

    print('Recommend Papers: \n', recommend_similar_papers('Paper Collection 123'))

    print('Search Papers: \n', search_papers('persona of LLMs'))

    print('Recommend Papers: \n', recommend_similar_papers('persona of LLMs'))
    # Sorry, we cannot find the paper collection you are looking for.
    #print('Recommend Papers: \n', recommend_similar_papers('123 asd Papers'))
    