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
The estimated average treatment effect (ATE) indicates that aggressive shooting **reduces the expected number of misses on average**.

- **ATE:** −0.19 misses  
- **Standard deviation of individual treatment effects:** 0.16  
- **Quantiles of treatment effects (5%, 50%, 95%):**  
  −0.47, −0.17, 0.04  

The distribution of individual treatment effects reveals substantial heterogeneity. While most observations show a performance benefit from aggressive shooting, a non-negligible subset exhibits near-zero or slightly positive effects, indicating that aggressive strategies are not universally optimal.

## Limitations
- The analysis relies on **observational data**, and causal identification depends on the assumption of no unobserved confounding.
- Aggressive shooting is defined via **shooting-time quantiles**, which may imperfectly capture strategic intent.
- Athlete-specific learning, psychological factors, and team-level strategies are not explicitly modeled.
- Results are averaged across disciplines and race formats, potentially masking context-specific effects.

## Next Steps
- Estimate **athlete-level or group-level treatment effects** (e.g., by experience or rank).
- Explore **alternative treatment definitions**, such as continuous shooting time or discipline-specific thresholds.
- Perform **robustness checks** using alternative causal estimators (e.g., DR Learners, linear DML).
- Extend the framework to downstream outcomes such as **final race position** or **time penalties**.

## Contribution
This project demonstrates how modern causal machine learning methods can be applied to sports analytics, providing interpretable and decision-relevant insights beyond standard predictive models.