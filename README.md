# Factors Affecting Output of Firms in Vietnam's Furniture Industry
## Econometric Analysis — Panel Data (2012–2018)

---

## 1. Research Overview

**Topic:** Factors affecting output of firms in Vietnam's furniture industry (VSIC code 31)

**Model:**

$$\ln(Output_{it}) = \beta_0 + \beta_1\ln(Labor_{it}) + \beta_2\ln(Capital_{it}) + \beta_3 Leverage_{it} + \beta_4\ln(Wage_{it}) + \beta_5\ln(Size_{it}) + \varepsilon_{it}$$

**Variable definitions:**

| Variable | Definition | Source column |
|---|---|---|
| ln(Output) | Log of net sales revenue (million VND) | `net_sales` |
| ln(Labor) | Log of number of employees (year-end) | `labor` |
| ln(Capital) | Log of fixed assets (million VND) | `end_fixed_assets` |
| Leverage | Ratio of total debt to total assets | `leverage` |
| ln(Wage) | Log of average labor cost per employee | `avg_wage` |
| ln(Size) | Log of total assets (million VND) | `end_assets` |

**Data:** GSO Enterprise Survey (Dieu tra Doanh nghiep), Vietnam, 2012–2018
**Sample:** 2,294 firm-year observations, 590 unique firms (VSIC 31 — Furniture manufacturing)

---

## 2. Project Structure

```
KTL_DUYANH_19032026/
├── Data/                          # Raw data (GSO Enterprise Survey .dta files)
│   ├── 2012_2018_1a+1am.dta       # Main panel dataset (2012–2018) — PRIMARY
│   ├── DTDN/                      # Additional DTDN files (2014–2017)
│   ├── Data GSO_DN1620/           # GSO data 2014–2023
│   ├── Stata_2000/ ... Stata_2012/ # Year-specific Stata files
│   └── ...
│
├── scripts/                       # Python analysis scripts (run in order)
│   ├── 01_data_cleaning.py        # Extract furniture firms, compute variables
│   ├── 02_descriptive_stats.py    # Tables & EDA figures
│   ├── 03_regression_analysis.py  # OLS / FE / RE / Hausman / diagnostics
│   └── 04_visualization.py        # Coefficient plots, residual diagnostics
│
├── notebooks/
│   └── analysis.ipynb             # Interactive Jupyter notebook (full analysis)
│
├── output/
│   ├── data/
│   │   └── furniture_cleaned.csv  # Cleaned panel dataset (2,294 obs)
│   ├── tables/                    # All regression and descriptive tables (CSV)
│   └── figures/                   # All charts and plots (PNG)
│       ├── fig1_distributions.png
│       ├── fig2_correlation_heatmap.png
│       ├── fig3_trends_over_time.png
│       ├── fig4_scatter_plots.png
│       ├── fig5_ownership_boxplots.png
│       ├── fig6_coefficient_plot.png
│       ├── fig7_residual_diagnostics.png
│       ├── fig8_actual_vs_predicted.png
│       ├── fig9_year_fixed_effects.png
│       └── fig10_panel_balance.png
│
├── run_pipeline.sh                # One-shot pipeline runner
├── README.md                      # This file
└── README_TASK.md                 # Original task description
```

---

## 3. How to Run

### Quick start (all steps at once):
```bash
bash run_pipeline.sh
```

### Step by step:
```bash
python scripts/01_data_cleaning.py      # ~60s
python scripts/02_descriptive_stats.py  # ~10s
python scripts/03_regression_analysis.py # ~30s
python scripts/04_visualization.py      # ~30s
```

### Open interactive notebook:
```bash
jupyter notebook notebooks/analysis.ipynb
```

### View HTML report:
Open `output/analysis_report.html` in any browser.

---

## 4. Data Cleaning Summary

| Step | Records |
|---|---|
| Raw furniture observations (VSIC 31) | 2,346 |
| After dropping missing core variables | 2,333 |
| After removing non-positive Capital | 2,300 |
| After removing negative Leverage | 2,294 |
| **Final analysis sample** | **2,294** |

- **Firm identifier:** `ma_thue` (tax code) — available for all years
- **Log variables winsorized** at 1st–99th percentile to reduce outlier influence
- **Years covered:** 2012, 2013, 2014, 2015, 2016, 2017, 2018
- **Ownership:** 66.4% Private, 12.2% FDI (SOE observations dropped in filtering)

---

## 5. Descriptive Statistics

### Raw Variables (Pre-log)

| Variable | N | Mean | Std Dev | Min | Median | Max |
|---|---|---|---|---|---|---|
| Output (mil VND) | 2,294 | 168,039 | 357,229 | 8 | 55,697 | 3,756,251 |
| Labor (employees) | 2,294 | 382 | 729 | 1 | 170 | 7,944 |
| Capital (mil VND) | 2,294 | 159,783 | 435,247 | 10 | 45,751 | 9,904,743 |
| Leverage | 2,294 | 0.683 | 0.339 | 0.04 | 0.692 | 2.36 |
| Wage (mil VND/emp) | 2,294 | 61.66 | 37.83 | 0.60 | 57.43 | 655.47 |
| Size (mil VND) | 2,294 | 150,307 | 387,327 | 213 | 47,697 | 6,439,532 |

### Key Correlations (Log Variables)

| | ln(Output) | ln(Labor) | ln(Capital) | Leverage | ln(Wage) | ln(Size) |
|---|---|---|---|---|---|---|
| ln(Output) | 1.000 | **0.891** | 0.803 | 0.117 | 0.537 | 0.854 |
| ln(Capital) | 0.803 | 0.802 | 1.000 | 0.109 | 0.456 | **0.884** |

> Note: High correlation between ln(Capital) and ln(Size) (r = 0.884) due to fixed assets being a component of total assets — multicollinearity is expected and addressed via FE estimation.

---

## 6. Regression Results (Main Findings)

### Table: Coefficient Estimates Across Models

| Variable | Pooled OLS | FE (1-way) | **FE (2-way)** | RE |
|---|---|---|---|---|
| ln(Labor) | 0.7153*** | 0.5050*** | **0.4841***** | 0.6247*** |
| ln(Capital) | −0.0873*** | 0.0365 | **0.1117**** | 0.0266 |
| Leverage | 0.0152 | −0.0180 | **−0.0274** | 0.0257 |
| ln(Wage) | 0.5337*** | 0.3454*** | **0.3564***** | 0.4067*** |
| ln(Size) | 0.4700*** | 0.2675*** | **0.2572****** | 0.4045*** |
| Constant | 0.9243*** | — | — | 1.3398*** |
| R² (within) | 0.266 | 0.344 | **0.338** | 0.317 |
| Observations | 2,294 | 2,294 | **2,294** | 2,294 |

*Significance levels: \*\*\* p<0.01, \*\* p<0.05, \* p<0.1. Robust standard errors.*
**Bold = preferred model (Hausman test)**

---

## 7. Model Selection: Hausman Test

| Test | Statistic | DF | p-value | Conclusion |
|---|---|---|---|---|
| Hausman (FE vs RE) | 41.9956 | 5 | **0.0000** | **Reject H0 — Use Fixed Effects** |

The Hausman test strongly rejects the null hypothesis that firm-level effects are uncorrelated with the regressors. This confirms that **Fixed Effects (FE)** is the appropriate estimator.

---

## 8. Interpretation of Results (Preferred Model: FE 2-way)

### Statistically Significant Effects:

| Variable | Coefficient | Interpretation |
|---|---|---|
| **ln(Labor)** | **0.484\*\*\*** | A 1% increase in workforce → **+0.48% output** (dominant factor) |
| **ln(Capital)** | **0.112\*\*** | A 1% increase in fixed assets → **+0.11% output** |
| **ln(Wage)** | **0.356\*\*\*** | A 1% increase in average wage → **+0.36% output** (proxy for labor quality/skill) |
| **ln(Size)** | **0.257\*\*\*** | A 1% increase in firm size → **+0.26% output** (economies of scale) |

### Non-Significant Effect:

| Variable | Coefficient | Interpretation |
|---|---|---|
| **Leverage** | −0.027 (n.s.) | Debt ratio has **no significant within-firm effect** on output |

### Key Insights:

1. **Labor is the most important factor**: Vietnam's furniture industry is highly labor-intensive. A 10% increase in workforce is associated with a ~5% increase in output, even after controlling for unobserved firm characteristics.

2. **Wage effect (skill premium)**: Higher wages reflect higher-skilled workers. The positive wage coefficient suggests that investing in human capital quality pays off in terms of output.

3. **Capital has a smaller role**: Fixed assets contribute positively (β₂ = 0.112) but their effect is smaller than labor. This is consistent with furniture being a labor-intensive manufacturing sector.

4. **Leverage is neutral**: The level of debt financing does not significantly affect output within firms over time. This suggests financial structure does not constrain or boost production in the short run.

5. **Returns to scale**: Sum of labor and capital elasticities (0.484 + 0.112 = 0.596) suggests **decreasing returns to scale** in the within-firm dimension.

### Robustness Check (ln(Value Added) as output):
Results are consistent: Labor (β = 0.530\*\*\*), Capital (β = 0.079\*), Wage (β = 0.747\*\*\*) remain significant. Leverage becomes significantly negative (β = −0.205\*\*) when using value added, suggesting debt may reduce productive efficiency.

---

## 9. Diagnostic Tests

| Test | Result | Implication |
|---|---|---|
| Breusch-Pagan | LM = 424.3, p = 0.000 | Heteroscedasticity present → robust SE used |
| Durbin-Watson | DW = 1.214 | Positive serial correlation present |
| VIF (lnCapital) | 164 | High multicollinearity with lnSize (expected) |
| VIF (lnSize) | 172 | High multicollinearity with lnCapital (expected) |
| F-test poolability | F = 7.55, p = 0.000 | Reject pooled OLS — FE is needed |

> The high VIF between ln(Capital) and ln(Size) is expected since fixed assets are a component of total assets. The FE estimator remains consistent despite multicollinearity, as it exploits within-firm variation.

---

## 10. Dependencies

```
Python 3.11+
pandas, numpy, scipy
statsmodels
linearmodels      # Panel data (FE/RE)
pyreadstat        # Read Stata .dta files
matplotlib, seaborn
jupyter, nbconvert
openpyxl
```

Install: `pip install pandas numpy scipy statsmodels linearmodels pyreadstat matplotlib seaborn jupyter nbconvert openpyxl`
