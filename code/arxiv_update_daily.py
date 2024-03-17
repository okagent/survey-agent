import arxiv
import re
import concurrent.futures

from tqdm import tqdm
from datetime import datetime, date, time, timedelta
from dateutil.relativedelta import relativedelta
import scipdf
import json
import re
import os
from paper_func import load_paper_pickle
from feature_func import compute_feature
from utils import config

def validateTitle(title):
    # Correct the messy path format of the paper title
    rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
    new_title = re.sub(rstr, "_", title)  # Replace with underscore
    return new_title
    
def get_papers_for_daily(start_date, end_date, json_path):
    client = arxiv.Client()
    papers = []
    
    print(f"Processing from {start_date} to {end_date}")

    # Create search query
    search = arxiv.Search(
        query="cat:cs.CL AND submittedDate:[" + start_date.strftime("%Y%m%d") + "* TO " + end_date.strftime("%Y%m%d") + "*]",
        # query="cat:cs.CL",
        max_results=None,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    # Execute search and process results
    results = list(client.results(search))
    print(f"下载了{len(results)}篇")

    for result in results:
        # Process each paper's information
            
        paper_id = result.get_short_id()[:10]
        authors = [author.name for author in result.authors]
        summary = result.summary
        summary = summary.replace("\n", " ")
        published_date = result.published.date()
        # url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        pdf_url = result.pdf_url

        paper = {
            'authors':authors,
            'title':result.title,
            'url':pdf_url,
            'abstract':summary,
            'arxiv_id':paper_id,
            "published_date": published_date.isoformat(),
            "year": published_date.year
        }
        papers.append(paper)
        
    for i, paper in tqdm(enumerate(papers)):
        ind = 0
        while ind < 3:
            try:
                processed_result = process_paper(paper['url'])
                if processed_result:
                    papers[i].update(processed_result)
                else:
                    papers[i].update({"introduction": "", "conclusion": "", "full_text": ""})
                break
            except Exception as e:
                with open(f"pdf_error.log", "a") as f:
                    f.write(f"Error processing {paper['title']}: {e}\n")
                ind += 1
        if ind == 3:
            papers[i].update({"introduction": "", "conclusion": "", "full_text": ""})
            with open(f"pdf_error.log", "a") as f:
                f.write(f"skip processing {paper['title']}")
            continue

    with open(f'{json_path}/processed_arxiv_{start_date.strftime("%Y%m%d")}.json', 'w', encoding='utf-8') as f:
        json.dump(papers, f, ensure_ascii=False, indent=4)    

# Iterate to get the papers of the previous day until there is already a processed file
def get_papers_from_arxiv_api(json_path):
    # Set a fixed end date as today
    current_date = datetime.combine(date.today() - timedelta(days=1), time.min)  # From 0:00 a.m. yesterday to 0.00 a.m. today
    end_date = datetime.combine(date.today(), time.min) 

    while current_date < end_date:

        file_path = f'{json_path}/processed_arxiv_{current_date.strftime("%Y%m%d")}.json'
        if not os.path.exists(file_path):
            get_papers_for_daily(current_date, end_date, json_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if len(data) == 0:
                    get_papers_for_daily(current_date, end_date, json_path)
                else:
                    break
        current_date -= relativedelta(days=1)
        end_date -= relativedelta(days=1)

other_titles = ['ethical statement', 'limitations', 'references', 'reference', 'appendix', 'acknowledgements', 'acknowledgments']

def process_paper(url):
    paper = scipdf.parse_pdf_to_dict(f"{url}.pdf")
    introduction = ''
    for section in paper['sections']:
        if 'introduction' in section['heading'].lower():
            introduction = section['text'].strip()
            break
    conclusion = ''
    conclusion_index = None
    for i, section in enumerate(paper['sections']):
        if 'conclusion' in section['heading'].lower():
            conclusion = section['text'].strip()
            conclusion_index = i
            
    include_up_to = len(paper['sections'])
    if conclusion_index is not None:
        include_up_to = conclusion_index + 1
    else:
        # If no Conclusion, find first of other specified sections
        exclude_start_index = next((index for index, section in enumerate(paper['sections']) if any(title in section['heading'] for title in other_titles)), None)
        if exclude_start_index is not None:
            include_up_to = exclude_start_index
    # Construct full text
    full_text = paper['abstract'] + '\n'  # Start with abstract
    for section in paper['sections'][:include_up_to]:
        full_text += section['heading'] + '\n' + section['text'] + '\n\n'
        
    return {
        "introduction": introduction,
        "conclusion": conclusion,
        "full_text": full_text
    }

if __name__ == "__main__":
    json_path = "/home/yxf/WIP/sva/pdf/extract_24_3_6"
    paper_pickle_path = f"{config['data_path']}/data/paper_corpus.pkl"
    get_papers_from_arxiv_api(json_path)
    load_paper_pickle(paper_pickle_path)
    compute_feature()