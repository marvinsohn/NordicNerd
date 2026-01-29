# NordicNerd – Causal Analysis of Shooting Strategies in Biathlon

## Motivation
This project studies the causal effect of aggressive shooting strategies in biathlon on shooting performance. Using observational race data, it aims to disentangle correlation from causation and quantify heterogeneity in treatment effects across athletes and race situations.

## Data
The analysis is based on processed biathlon race data at the shooting-level, including information on shooting time, misses, rank before shooting, weather conditions, and athlete performance indicators.  
Data preprocessing and feature engineering are performed upstream and stored as serialized datasets.

## Methodology
Causal effects are estimated using a **Causal Forest (Double Machine Learning)** approach implemented via `econml`.

- **Treatment:** Aggressive shooting (`is_aggressive`, defined via shooting time quantiles)
- **Outcome:** Number of misses
- **Covariates:** Race context, athlete form indicators, weather conditions, and rank before shooting

This framework allows estimation of both the **average treatment effect (ATE)** and **heterogeneous treatment effects**.

## Results
The estimated average treatment effect suggests that aggressive shooting **reduces the number of misses on average**.  
However, treatment effects are heterogeneous: while most observations benefit from aggressive shooting, a subset of situations shows neutral or slightly negative effects.

## Contribution
This project demonstrates how modern causal machine learning methods can be applied to sports analytics, providing interpretable and policy-relevant insights beyond standard predictive models.
