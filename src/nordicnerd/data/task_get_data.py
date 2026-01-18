from pathlib import Path
from pytask import task

from nordicnerd.data.get_data import build_and_store_athlete_season_index_from_db
from nordicnerd.config import BLD

@task
def task_create_dataset_top50_athletes_per_season(produces: Path = BLD / "data" / "top50_2020_2025.json"):
    """
    Build and store the top 50 athletes per season from MongoDB,
    saving the output as a JSON file.

    Parameters
    ----------
    produces : Path
        Path where the JSON file will be written.
    """

    # Ensure parent directory exists
    produces.parent.mkdir(parents=True, exist_ok=True)

    build_and_store_athlete_season_index_from_db(
        start_year=2020,
        end_year=2025,
        top_n=50,
        output_path=produces
    )
