from typing import List, Dict, Union

class Paper:
    def __init__(self, title: str, authors: List[str], abstract: str):
        self.title = title
        self.authors = authors
        self.abstract = abstract
        # And other relevant attributes

# Assuming these are placeholder implementations for the sake of demonstration
# In practice, you would have more sophisticated logic to actually retrieve papers

def get_paper_by_name(paper_names: List[str]) -> Dict[str, Union[Paper, None]]:
    """
    Simulates retrieval of papers by their name.

    :param paper_names: List of approximate names of the papers.
    :return: Dictionary where keys are the names provided and values are Paper objects or None
    """
    # Placeholder implementation: return dummy papers or None
    return found_papers

def get_paper_details(paper_name: str, mode: str) -> Union[
    dict[str, str], dict[str, str], dict[str, str], dict[str, str], None]:
    """
    Simulates retrieval of paper details.

    :param paper_name: Exact name of the paper.
    :param mode: Type of details required ('short' or 'full').
    :return: Dictionary with details of the paper or None if not found.
    """
    # Placeholder implementation: return dummy details based on mode
    return None

