import os
from pymongo import MongoClient
from urllib.parse import quote
from dotenv import load_dotenv
import json
from typing import Dict, Any
from pathlib import Path

load_dotenv()

username = os.getenv("MONGO_USERNAME")
password = os.getenv("MONGO_PASSWORD")
hostname = os.getenv("MONGO_HOST")
port = os.getenv("MONGO_PORT")

if not all([username, password, hostname, port]):
    raise RuntimeError("MongoDB credentials not set in environment variables")

encoded_username = quote(username)
encoded_password = quote(password)

mongodb_url = f"mongodb://{encoded_username}:{encoded_password}@{hostname}:{port}/"
client = MongoClient(mongodb_url)

def query_mongo(database_name, collection_name, query=None, return_fields=None):
    if query is None:
        query = {}
    if return_fields is None:
        return_fields = {'_id': 0}
    db = client[database_name]
    collection = db[collection_name]
    return list(collection.find(query, return_fields))


def build_and_store_athlete_season_index_from_db(
    start_year: int,
    end_year: int,
    top_n: int,
    output_path: Path
) -> Dict[str, Any]:
    """
    Query season-based World Cup data from MongoDB, build an athlete-centered
    index grouped by IBUID, and store the result as a JSON file.

    The function:
    - Queries the 'SeasonScores' collection for each season
    - Filters discipline = 'NonTeam'
    - Assigns every athlete a gender:
        index 0 -> women
        index 1 -> men
    - Uses the top N athletes per gender (based on existing order)
    - Aggregates season scores per athlete (IBUID-based)

    Parameters
    ----------
    start_year : int
        First season year (inclusive).
    end_year : int
        Last season year (inclusive).
    top_n : int
        Number of top athletes per season and gender.
    output_path : pathlib.Path
        Full output file path INCLUDING `.json`.

    Returns
    -------
    dict
        Athlete-centered dictionary indexed by IBUID.
    """

    athletes_by_ibuid: Dict[str, Any] = {}

    gender_by_index = {
        0: "women",
        1: "men"
    }

    for year in range(start_year, end_year + 1):
        season_docs = query_mongo(
            database_name="Analysis",
            collection_name="SeasonScores",
            query={
                "year": year,
                "discipline": "NonTeam"
            }
        )

        if not season_docs:
            continue

        for idx, season_doc in enumerate(season_docs):
            gender = gender_by_index.get(idx)
            if gender is None:
                continue

            top_athletes = season_doc["scores"][:top_n]

            for athlete in top_athletes:
                ibuid = athlete["ibuid"]

                if ibuid not in athletes_by_ibuid:
                    athletes_by_ibuid[ibuid] = {
                        "givenName": athlete["givenName"],
                        "familyName": athlete["familyName"],
                        "gender": gender
                    }

                athletes_by_ibuid[ibuid][str(year)] = {
                    "totalScore": athlete["total"]
                }

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON exactly to the given path
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(athletes_by_ibuid, f, ensure_ascii=False, indent=2)

    return athletes_by_ibuid