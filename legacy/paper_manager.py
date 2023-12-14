from typing import Dict, List, Optional
from paper_retriever import Paper

class PaperList:
    def __init__(self, name: str):
        self.name = name
        self.papers: List[Paper] = []

    def add_paper(self, paper: Paper):
        # Add a paper to the list
        if paper not in self.papers:
            self.papers.append(paper)

    def remove_paper(self, paper: Paper):
        # Remove a paper from the list
        if paper in self.papers:
            self.papers.remove(paper)

    def get_papers(self) -> List[Paper]:
        # Get all papers in the list
        return self.papers


class PaperManager:
    def __init__(self, storage_backend: Dict[str, PaperList]):
        self.storage_backend = storage_backend

    def get_paperlist_by_name(self, list_name: str) -> str:
        """
        Finds the most matching paper list name based on a fuzzy list name.

        :param list_name: The fuzzy name of the paper list
        :return: The name of the best matching paper list
        :raises: Exception if the matching threshold is not met
        """
        pass

    def update_paper_list(self, list_name: str, action: str, paper_indices: str) -> bool:
        """
        Updates a paper list with specified actions such as add or delete.

        :param list_name: The name of the PaperList to update
        :param action: The action to perform (add, del)
        :param paper_indices: Paper indices in the format "1-2, 4" or "1,3,5"
        :return: True if the operation was successful, False otherwise
        """
        pass


# might be replaced with a database

if __name__ == "__main__":
    in_memory_storage_backend: Dict[str, PaperList] = {}