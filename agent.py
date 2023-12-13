from paper_retriever import get_paper_by_name, get_paper_details
from paper_manager import PaperManager, PaperList
from search_engine import search_papers, retrieve_papers
from query_processor import query_area_papers, query_individual_papers


def parse_query(user_input: str) -> Tuple[str, dict]:
    """
    Parses the user query using natural language processing to determine the
    intended action and the parameters needed for that action.

    :param user_input: The raw query string entered by the user.
    :return: A tuple of the action type as a string and a dictionary of parameters associated with that action.

    Possible actions include:
    - "search_similar_papers": Finding papers similar to a list of provided paper names.
    - "find_related_papers": Finding papers related to a specific topic or query.
    - "summarize_papers": Summarizing a set of papers or providing a summary of a specific field.
    - "find_proof": Identifying specific papers that prove a given statement or claim.
    # Example return structure: (action, params)
    """
    return "some_action", {"some_key": "some_value"}


class SurveyAgent:
    def __init__(self):
        self.paper_manager = PaperManager()

    def handle_input(self, user_input: str, user_id: str):
        """
        Handles user input by using an NLP function to parse the query
        and determine the corresponding action.
        """
        # Using an NLP function (parse_query) to understand and parse user queries
        action, params = parse_query(user_input)

        # Perform actions based on the type of action determined by the NLP function
        if action == "search_similar_papers":
            paper_names = params["paper_names"]
            found_papers = get_paper_by_name(paper_names)
            list_name = self._create_paper_list(found_papers, user_id)
            return search_papers(rank='search', tags=list_name)

        elif action == "find_related_papers":
            query_text = params["query_text"]
            retrieved_papers = retrieve_papers(query_text)
            return query_area_papers(user_id, query_text)

        elif action == "summarize_papers":
            list_name = params["list_name"]
            paper_list = self.paper_manager.get_paperlist_by_name(list_name)
            return query_area_papers(list_name, "summarize")

        elif action == "find_proof":
            query_text = params["query_text"]
            retrieved_papers = retrieve_papers(query_text)
            return query_individual_papers(user_id, query_text)

        else:
            raise ValueError(f"Unrecognized action request: {action}")

    def _create_paper_list(self, found_papers, user_id):
        """
        Helper method to create a paper list for the user based on found papers.
        """
        # Logic to create and store a new paper list for the user
        ...
        return list_name


# Sample Usage
if __name__ == "__main__":
    agent = SurveyAgent()
    user_id = "user123"  # Example user ID
    user_input = input("Please enter your query: ")
    result = agent.handle_input(user_input, user_id)
    print(result)