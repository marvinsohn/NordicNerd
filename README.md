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

## Results I — Average Treatment Effect (ATE)

The first step of the analysis examines the **average causal effect** of aggressive shooting behavior across all observed athletes and race situations.

**Summary statistics**
- Average Treatment Effect (ATE): **−0.19 misses**
- Standard deviation of treatment effects: **0.16**
- 5% / 50% / 95% quantiles: **[−0.47, −0.17, 0.04]**

The negative ATE indicates that, on average, aggressive shooting is associated with **fewer missed shots** compared to non-aggressive shooting strategies.

### Implications (ATE)

At the aggregate level, the results suggest that aggressive shooting does not harm accuracy on average and may even improve shooting performance.  
However, the substantial spread of estimated effects suggests that this average masks meaningful variation across athletes and race contexts.

*Personal note: The negative average treatment effect suggests that, in some situations, shooting more aggressively may actually be beneficial for accuracy. This resonates with examples of athletes appearing to be completely “in the zone”, such as Jacquelin during the 2021 World Championship pursuit or Wierer in the 2020 Antholz World Championship relay (although these performances are not part of the underlying dataset).*

*Interestingly, this contrasts with the common behavior in the individual race format, where many athletes deliberately slow down their shooting due to the one-minute penalty time. The estimated ATE points in a different direction: on average, slowing down does not appear to improve accuracy, and in some cases, allowing well-trained shooting automatisms to take over may even be advantageous.*

---

## Results II — Heterogeneous Treatment Effects (Exploratory)

Conditional average treatment analysis (CATE) analysis reveals that the effect of aggressive shooting varies across different race situations and athlete characteristics.

**CATE by time behind leader**
| time_behind_bin | mean   | std    | count |
|-----------------|--------|--------|-------|
| low             | -0.124 | 0.147  | 8768  |
| mid-low         | -0.125 | 0.145  | 8774  |
| mid-high        | -0.214 | 0.134  | 8759  |
| high            | -0.292 | 0.139  | 8766  |

**CATE by season form**
| form_bin | mean   | std    | count |
|----------|--------|--------|-------|
| low form | -0.210 | 0.167  | 11708 |
| mid form | -0.182 | 0.154  | 11672 |
| high form| -0.174 | 0.148  | 11687 |

**CATE by rank before shooting**
| rank_bin   | mean   | std    | count |
|------------|--------|--------|-------|
| front pack | -0.129 | 0.121  | 8818  |
| upper mid  | -0.173 | 0.150  | 8808  |
| lower mid  | -0.196 | 0.160  | 8784  |
| back pack  | -0.259 | 0.168  | 8657  |

**CATE by shooting number**
| shooting_number | mean   | std    | count |
|-----------------|--------|--------|-------|
| 1               | -0.113 | 0.146  | 14021 |
| 2               | -0.230 | 0.147  | 14000 |
| 3               | -0.241 | 0.129  | 3529  |
| 4               | -0.273 | 0.144  | 3517  |

### Implications (Heterogeneity)

- Athletes who are further behind the leader benefit more from aggressive shooting, suggesting it may help them recover time without substantially increasing misses.  
- Less experienced or lower-performing athletes (low season form) see a stronger positive effect of aggressive shooting, potentially because they rely more on automatisms when shooting faster.  
- Athletes in the back of the pack (rank_before_shooting) benefit more from shooting aggressively, likely as a catch-up strategy.  
- Later shooting bouts (3rd and 4th) show larger effects of aggressiveness, which may reflect accumulated pressure or fatigue during the race.  

These heterogeneous effects highlight that **the optimal shooting strategy is context-dependent**, supporting the use of causal forests to uncover nuanced decision rules in elite sports.

---

## Limitations

- Observational data cannot fully rule out unobserved confounding
- Treatment definition relies on relative shooting time thresholds
- Athlete-specific fixed effects are not explicitly modeled
- Heterogeneous effects are estimated but not yet formally decomposed

---

## Next Steps

- Systematically analyze conditional average treatment effects (CATEs)
- Explore subgroup effects by athlete characteristics and race context
- Consider alternative treatment definitions (continuous or athlete-relative)
- Add uncertainty quantification for individual and subgroup effects
- Compare causal forest estimates with simpler causal baselines

---

