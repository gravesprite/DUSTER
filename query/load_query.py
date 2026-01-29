import os
import json

def load_queries(args):
    """
    Load and format queries into 'random' and 'sequential' categories.

    Args:
        query_data (dict): A dictionary containing query information.
                           Expected keys are 'random' and 'sequential'.
        args: An object containing the parameter `query_path`, which specifies the directory
              containing the query JSON files.

    Returns:
        dict: A dictionary with 'random' and 'sequential' queries formatted.
    """

    query_data ={}

    random_queries_path = os.path.join(args.query_path, "random_queries_{}.json".format(args.sequence_id))
    sequential_queries_path = os.path.join(args.query_path, "sequential_queries_{}.json".format(args.sequence_id))

    # Load random queries
    if os.path.exists(random_queries_path):
        with open(random_queries_path, "r") as f:
            query_data["random"] = json.load(f)
    else:
        query_data["random"] = []

    # Load sequential queries
    if os.path.exists(sequential_queries_path):
        with open(sequential_queries_path, "r") as f:
            query_data["sequential"] = json.load(f)
    else:
        query_data["sequential"] = []

    formatted_queries = {
        "random": query_data.get("random", []),
        "sequential": query_data.get("sequential", [])
    }
    return formatted_queries