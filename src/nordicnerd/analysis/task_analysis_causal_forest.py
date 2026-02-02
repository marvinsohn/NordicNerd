from pathlib import Path
import pytask

from nordicnerd.config import BLD
from nordicnerd.analysis.causal_forest import run_causal_forest_analysis


@pytask.mark.product(
    ate_summary=BLD / "results" / "ate_summary.csv",
    cate_time=BLD / "results" / "cate_time_behind.csv",
    cate_form=BLD / "results" / "cate_form.csv",
    cate_rank=BLD / "results" / "cate_rank_before_shooting.csv",
    cate_shooting=BLD / "results" / "cate_shooting_number.csv",
)
def task_run_causal_forest(
    data_path: Path = BLD / "data" / "race_data_processed.pkl",
):
    """
    Run causal forest analysis and write all result tables.
    """

    output_dir = BLD / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_causal_forest_analysis(
        data_path=data_path,
        output_dir=output_dir,
    )
