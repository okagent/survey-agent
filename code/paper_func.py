import difflib
import pdb 

# load paper_corpus.json
import json
paper_corpus_path='../data/arxiv_full_papers.json'
with open(paper_corpus_path, 'r') as f:
    paper_corpus_json = json.load(f)[0]
    paper_corpus = { p['title']:p for p in paper_corpus_json}


# Example papers
example_papers = [
    {"name": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "author": "Devlin et al.", "year": 2018, "content": '''We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.
BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).'''},
    {"name": "Attention Is All You Need", "author": "Vaswani et al.", "year": 2017, 'content': '''The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.'''}
]


# Example paper collections
paper_collections = {
    "Transformer Collection": ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "Attention Is All You Need"]
}


def get_paper_by_name(paper_titles):
    """Find corresponding papers based on a list of fuzzy paper names."""
    found_papers = []
    for fuzzy_name in paper_titles:
        matches = difflib.get_close_matches(fuzzy_name, paper_corpus.keys(), n=1, cutoff=0.6)
        # 这个matches还挺慢的，@鑫凤 可不可以优化一下？
       
        if matches:
            found_papers.append(matches[0])
        else:
            found_papers.append(None)  # Append None if no matching paper is found
    return found_papers

def get_paperlist_by_name(collection_name):
    """Find the name of the paper collection that best matches a fuzzy collection name."""
    # Find the closest match for the collection name
    match = difflib.get_close_matches(collection_name, paper_collections.keys(), n=1, cutoff=0.5)
    if match:
        return match[0]
    else:
        raise None

def get_paper_content(paper_name, mode):
    """Get text content of a paper based on its exact name."""
    if paper_name in paper_corpus.keys():
        if mode == 'full':
            return paper_corpus[paper_name]['full_text'] 
        elif mode == 'short':
            return paper_corpus[paper_name]['abstract'] 
        else:
            raise ValueError("Invalid input. Please enter either 'full' or 'short'.")
    else:
        return None

def update_paper_list(user, list_name, action, paper_numbers):
    # TODO ... 
    """Update a paper list based on provided action and paper numbers."""
    # Convert string paper_numbers to actual list of indices
    indices = []
    for part in paper_numbers.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))

    # Get the current list of papers for the user
    current_list = paper_lists.get((user, list_name), [])

    # Perform the add or delete action
    if action == "add":
        current_list.extend(indices)
    elif action == "del":
        current_list = [i for i in current_list if i not in indices]

    paper_lists[(user, list_name)] = current_list
    return True


from langchain.retrievers import BM25Retriever
from langchain.schema import Document

from langchain.docstore.document import Document
paper_docs = [ Document(page_content=p['full_text'], metadata={k:p[k] for k in ['title']}) for title, p in paper_corpus.items()]
# 这里没有对paper_docs做分段，只给了一个最naive的实现，可能导致retrieve回了一整篇paper，@石玮 看看怎么处理？

retriever = BM25Retriever.from_documents(paper_docs)

def retrieve_papers(query):
    result = retriever.get_relevant_documents(query)
    if len(result) > 0:
        return result[0]
    else:
        raise None


if __name__ == '__main__':
    # Example usage
    print(get_paper_by_name(["Attention is all you need", "BERT: xxx"]))
    print(get_paperlist_by_name("Transformer Coll"))

    print(retrieve_papers('''Good-enough compositional\ndata augmentation. In Proceedings of the 58th An-\nnual Meeting of the Association for Computational\nLinguistics, pages 7556–7566, Online. Association\nfor Computational Linguistics.\nDavid Baehrens, Timon Schroeter, Stefan Harmel-\ning, Motoaki Kawanabe'''))

    print(get_paper_content('Not All Attention Is All You Need', mode='short'))