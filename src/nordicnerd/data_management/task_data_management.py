from pytask import task
from nordicnerd.config import BLD
import pandas as pd
import json
import re

def parse_behind_to_seconds(behind):
    if behind in [None, ""]:
        return None
    if isinstance(behind, (int, float)):
        return float(behind)
    behind = behind.strip().lstrip("+")
    if ":" in behind:
        try:
            minutes, seconds = behind.split(":")
            return float(minutes) * 60 + float(seconds)
        except ValueError:
            return None
    try:
        return float(behind)
    except ValueError:
        return None

def extract_race_state(intermediate_times, shooting_number):
    if not intermediate_times:
        return None, None
    shooting_pattern = re.compile(
        rf"shooting\s*{shooting_number}", re.IGNORECASE
    )
    for it in intermediate_times:
        category = it.get("category", "")
        if not shooting_pattern.search(category):
            continue
        if "standings" in category.lower():
            continue
        return (
            it.get("rank"),
            parse_behind_to_seconds(it.get("behind"))
        )
    return None, None

@task
def task_build_race_df(depends_on=BLD / "data/races_2020_2025.json",
                       produces=BLD / "data/race_data_processed.pkl"):
    
    with open(depends_on, "r") as f:
        raw_race_data = json.load(f)

    rows = []
    for race in raw_race_data:
        race_id = race.get("raceId")
        season = race.get("season") or race.get("year")
        weather = race.get("weather", {}).get("start", {})
        air_temp = weather.get("airTemperature")
        snow_temp = weather.get("snowTemperature")
        snow_condition = weather.get("snowCondition")

        for athlete in race.get("athletes", []):
            athlete_id = athlete.get("ibuid")
            athlete_name = athlete.get("shortName") or athlete.get("nameMeta")
            individual_shots = athlete.get("individualShots", [])
            intermediate_times = athlete.get("intermediateTimes", [])

            for shooting_idx, shooting in enumerate(individual_shots, start=1):
                shooting_time = shooting.get("shootingTime")
                shot_count = shooting.get("shotCount")
                missed_shots = shooting.get("missedShots")
                penalty_laps = shooting.get("penaltyLapsCount")
                penalty_time = shooting.get("penaltyTime")

                hits = (
                    shot_count - missed_shots
                    if shot_count is not None and missed_shots is not None
                    else None
                )

                rank_before, time_behind_before = extract_race_state(
                    intermediate_times, shooting_idx
                )

                rows.append({
                    "race_id": race_id,
                    "season": season,
                    "athlete_id": athlete_id,
                    "athlete_name": athlete_name,
                    "shooting_number": shooting_idx,
                    "shooting_position": "prone" if shooting_idx % 2 == 1 else "standing",
                    "shooting_time": shooting_time,
                    "shots": shot_count,
                    "misses": missed_shots,
                    "hits": hits,
                    "penalty_laps": penalty_laps,
                    "penalty_time": penalty_time,
                    "rank_before_shooting": rank_before,
                    "time_behind_before": time_behind_before,
                    "air_temp": air_temp,
                    "snow_temp": snow_temp,
                    "snow_condition": snow_condition,
                })

    df = pd.DataFrame(rows)

    numeric_cols = [
        "shooting_time", "shots", "misses", "hits",
        "penalty_laps", "penalty_time",
        "rank_before_shooting", "time_behind_before",
        "air_temp", "snow_temp"
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df = df.sort_values(
        ["athlete_id", "season", "race_id", "shooting_number"]
    ).reset_index(drop=True)

    # Ski Form
    df["ski_form_race"] = -df["time_behind_before"]
    df["ski_form_race_z"] = (
        df.groupby("race_id")["ski_form_race"]
          .transform(lambda x: (x - x.mean()) / x.std())
    )

    season_form = (
        df[df["shooting_number"] == 1]
        .groupby(["athlete_id", "season"])["ski_form_race"]
        .mean()
        .reset_index()
        .rename(columns={"ski_form_race": "ski_form_season"})
    )
    df = df.merge(
        season_form,
        on=["athlete_id", "season"],
        how="left"
    )
    df["ski_form_season_z"] = (
        df.groupby("season")["ski_form_season"]
          .transform(lambda x: (x - x.mean()) / x.std())
    )

    # Aggressive Shooting Treatment
    MAIN_QUANTILE = 0.2
    ROBUSTNESS_QUANTILES = [0.1, 0.3]
    df["shooting_time_q20"] = (
        df.groupby(["athlete_id", "season"])["shooting_time"]
          .transform(lambda x: x.quantile(MAIN_QUANTILE))
    )
    df["is_aggressive"] = df["shooting_time"] < df["shooting_time_q20"]
    for q in ROBUSTNESS_QUANTILES:
        q_col = f"shooting_time_q{int(q*100)}"
        t_col = f"is_aggressive_q{int(q*100)}"
        df[q_col] = (
            df.groupby(["athlete_id", "season"])["shooting_time"]
              .transform(lambda x: x.quantile(q))
        )
        df[t_col] = df["shooting_time"] < df[q_col]

    # Speichere Output
    df.to_pickle(produces)