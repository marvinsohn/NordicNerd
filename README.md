# NordicNerd — Causal Analysis of Aggressive Shooting in Biathlon

This project investigates the causal effect of aggressive shooting behavior on shooting performance in professional biathlon races.  
Using observational race data, the analysis estimates heterogeneous treatment effects with modern causal machine learning methods.

*Personal note: Being a biathlon fan myself and spending numerous winter weekends each year watching races, one starts to notice clear differences in how athletes approach the shooting range under pressure. Some slow down deliberately to "increase their chances" of hitting all targets, others try to stick to their usual routine at all costs, while some athletes suddenly shoot much faster than they normally would, seemingly trusting automatisms over conscious control.*

*These different approaches made me wonder whether shooting more aggressively actually helps performance or whether it increases the risk of misses. Is it beneficial to aim for a very fast standing shoot when an athlete typically takes much longer, or is it better to remain within one’s established rhythm?*

*This project grew out of that curiosity and aims to quantify the causal effect of aggressive shooting strategies on shooting accuracy.*

---

## Data

The dataset consists of race-level observations, including athlete characteristics, race context, environmental conditions, and shooting outcomes.

**Outcome**
- `misses`: Number of missed shots during a shooting bout

**Treatment**
- `is_aggressive`: Binary indicator for aggressive shooting behavior, defined via shooting time quantiles

**Covariates**
- Weather conditions (air and snow temperature)
- Athlete form indicators (race- and season-level z-scores)
- Race context (rank before shooting, time behind leader)
- Shooting order and race position before the shooting

*Personal note: Perhaps unsurprisingly, the data wrangling stage turned out to be the most tedious part of the project.*

*To keep observations comparable across races, I decided to focus exclusively on sprint and individual races, given the strong influence of race dynamics on shooting behavior.*

*In sprint and individual races, athletes generally have to “play their own game” regardless of their current (virtual) position, as competitors starting later may still outperform them. In contrast, in pursuits or mass starts, an athlete leading comfortably before the final shooting — sometimes by a full minute — may experience substantially less pressure at the range.*

*Similarly, skiing effort can vary strongly in pursuits and mass starts depending on current position, whereas sprint and individual formats require athletes to push at a consistently high level throughout the race.*

---

## Methodology

Causal effects are estimated using a **Causal Forest (Double Machine Learning)** approach as implemented in the `econml` library.

- Outcome and treatment models: Random Forest regressors
- Estimation target: Individual Treatment Effects (ITEs)
- Focus: Heterogeneous effects rather than a single average effect

The model is trained on preprocessed numerical and one-hot encoded features, with observations containing missing values removed.

---

## Results

The estimated treatment effects indicate that aggressive shooting is, on average, associated with **more missed shots**, though substantial heterogeneity exists.

**Summary statistics**
- Average Treatment Effect (ATE): **−0.19 misses**
- Standard deviation of treatment effects: **0.16**
- 5% / 50% / 95% quantiles: **[−0.47, −0.17, 0.04]**

These results suggest that aggressive shooting is generally harmful for accuracy, but not uniformly so across all situations and athletes.

---

## Interpretation

The results support the hypothesis that shooting faster increases the risk of misses on average, while also highlighting that the effect depends strongly on context and athlete characteristics.

This underlines the importance of moving beyond average effects when analyzing strategic decisions in elite sports.

---

## Limitations

- Observational data cannot fully rule out unobserved confounding
- Treatment definition relies on relative shooting time thresholds
- No athlete-specific fixed effects are explicitly modeled
- Results are sensitive to feature selection and preprocessing choices

---

## Next Steps

- Explore alternative treatment definitions (continuous or athlete-relative)
- Add formal uncertainty estimates for individual effects
- Investigate subgroup effects (e.g. by experience or race type)
- Compare causal forest results with simpler causal baselines

---

## Project Status

This project is part of an ongoing research-oriented exploration of causal machine learning methods applied to elite sports performance data.
