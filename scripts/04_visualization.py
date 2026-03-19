"""
Script 04: Regression Diagnostics & Results Visualization
=========================================================
Generates publication-quality figures for regression results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS as LMPooledOLS
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_IN  = os.path.join(BASE_DIR, "output", "data", "furniture_cleaned.csv")
TBL_DIR  = os.path.join(BASE_DIR, "output", "tables")
FIG_DIR  = os.path.join(BASE_DIR, "output", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 120,
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── Load data & re-run models ─────────────────────────────────────────────────
df = pd.read_csv(DATA_IN)
DEPVAR   = 'lnOutput'
INDVARS  = ['lnLabor', 'lnCapital', 'Leverage', 'lnWage', 'lnSize']
ALL_VARS = [DEPVAR] + INDVARS

VAR_LABELS = {
    'lnLabor':   'ln(Labor)',
    'lnCapital': 'ln(Capital)',
    'Leverage':  'Leverage',
    'lnWage':    'ln(Wage)',
    'lnSize':    'ln(Size)',
}

df_model = df[['firm_id', 'year'] + ALL_VARS].dropna()
panel    = df_model.set_index(['firm_id', 'year']).sort_index()
Y        = panel[DEPVAR]
X_vars   = panel[INDVARS]
X_const  = sm.add_constant(X_vars)

# Fit models
ols_res = LMPooledOLS(Y, X_const).fit(cov_type='robust')
fe_res  = PanelOLS(Y, X_vars, entity_effects=True, time_effects=True).fit(cov_type='robust')
fe1_res = PanelOLS(Y, X_vars, entity_effects=True, time_effects=False).fit(cov_type='robust')
re_res  = RandomEffects(Y, X_const).fit(cov_type='robust')

print("Models re-fitted.")

# ── Figure 6: Coefficient plot (FE preferred model) ──────────────────────────
def coef_plot(results_dict, title, figname, indvars=INDVARS):
    """Plot coefficients with confidence intervals for multiple models."""
    n_models = len(results_dict)
    colors   = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0'][:n_models]

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos   = np.arange(len(indvars))
    width   = 0.2

    for i, (model_name, res) in enumerate(results_dict.items()):
        params = res.params[indvars]
        ci_lo  = res.conf_int(level=0.95).loc[indvars, 'lower']
        ci_hi  = res.conf_int(level=0.95).loc[indvars, 'upper']
        offset = (i - n_models/2 + 0.5) * width
        ax.barh(y_pos + offset, params.values,
                height=width * 0.85,
                color=colors[i], alpha=0.75, label=model_name)
        ax.errorbar(params.values, y_pos + offset,
                    xerr=[params.values - ci_lo.values, ci_hi.values - params.values],
                    fmt='none', color='black', capsize=3, linewidth=1.2)

    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([VAR_LABELS.get(v, v) for v in indvars])
    ax.set_xlabel('Coefficient estimate (95% CI)')
    ax.set_title(title, fontweight='bold', pad=12)
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, figname), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {figname}")

coef_plot(
    {'OLS': ols_res, 'FE (1-way)': fe1_res, 'FE (2-way)': fe_res, 'RE': re_res},
    "Figure 6: Coefficient Estimates Across Models\n(Vietnam Furniture Industry, 2012–2018)",
    "fig6_coefficient_plot.png"
)

# ── Figure 7: Residual plots (FE model) ─────────────────────────────────────
fe_fitted = fe_res.fitted_values
fe_resid  = fe_res.resids

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# (a) Residuals vs Fitted
ax = axes[0]
ax.scatter(fe_fitted, fe_resid, alpha=0.3, s=10, color='steelblue')
ax.axhline(0, color='red', linewidth=1, linestyle='--')
ax.set_xlabel('Fitted values')
ax.set_ylabel('Residuals')
ax.set_title('(a) Residuals vs Fitted', fontweight='bold')

# (b) Q-Q plot
ax = axes[1]
from scipy import stats as scipy_stats
resid_vals = fe_resid.values
(osm, osr), (slope, intercept, r) = scipy_stats.probplot(resid_vals, dist='norm')
ax.scatter(osm, osr, alpha=0.4, s=10, color='steelblue')
ax.plot(osm, slope * np.array(osm) + intercept, color='red', linewidth=1.5)
ax.set_xlabel('Theoretical Quantiles')
ax.set_ylabel('Sample Quantiles')
ax.set_title('(b) Normal Q-Q Plot', fontweight='bold')

# (c) Residual histogram
ax = axes[2]
ax.hist(resid_vals, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
x_range = np.linspace(resid_vals.min(), resid_vals.max(), 200)
density  = scipy_stats.norm.pdf(x_range, resid_vals.mean(), resid_vals.std())
ax2 = ax.twinx()
ax2.plot(x_range, density, color='red', linewidth=1.5)
ax2.set_ylabel('Density', color='red')
ax.set_xlabel('Residuals')
ax.set_title('(c) Residual Distribution', fontweight='bold')

plt.suptitle("Figure 7: Regression Diagnostics — Fixed Effects Model (2-way)",
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig7_residual_diagnostics.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig7_residual_diagnostics.png")

# ── Figure 8: Predicted vs Actual ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
actual  = Y.values
pred    = fe_fitted.values
ax.scatter(actual, pred, alpha=0.3, s=12, color='steelblue')
lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='45° line')
ax.set_xlabel('Actual ln(Output)')
ax.set_ylabel('Predicted ln(Output)')
ax.set_title("Figure 8: Actual vs Predicted ln(Output)\n(Fixed Effects Model, 2-way)",
             fontweight='bold')
r2_within = fe_res.rsquared_within
ax.text(0.05, 0.92, f"R² (within) = {r2_within:.4f}", transform=ax.transAxes,
        fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig8_actual_vs_predicted.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig8_actual_vs_predicted.png")

# ── Figure 9: Year fixed effects (from FE2W model) ───────────────────────────
# Estimate year effects by demeaning and adding back time average
print("\n  Computing year effects...")

# FE within estimator: time effects via time dummies
# Re-run with time dummies explicitly to extract coefficients
import statsmodels.formula.api as smf
df_fe = df_model.copy()
df_fe['year_cat'] = df_fe['year'].astype(str)

# Group-demean to control for firm FE
df_fe_dm = df_fe.copy()
for var in ALL_VARS:
    df_fe_dm[var] = df_fe[var] - df_fe.groupby('firm_id')[var].transform('mean')

formula_yr = 'lnOutput ~ lnLabor + lnCapital + Leverage + lnWage + lnSize + C(year_cat)'
ols_yr_res = smf.ols(formula_yr, data=df_fe_dm).fit()

yr_effects = {int(k.split('[T.')[1].rstrip(']')): v
              for k, v in ols_yr_res.params.items() if 'year_cat' in k}
yr_df = pd.DataFrame(list(yr_effects.items()), columns=['Year', 'Effect']).sort_values('Year')

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(yr_df['Year'], yr_df['Effect'], color='steelblue', alpha=0.8, edgecolor='white')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('Year')
ax.set_ylabel('Year Fixed Effect (relative to base year)')
ax.set_title("Figure 9: Year Fixed Effects on ln(Output)\n(Vietnam Furniture Industry)",
             fontweight='bold')
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig9_year_fixed_effects.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig9_year_fixed_effects.png")

# ── Figure 10: Panel balance ──────────────────────────────────────────────────
obs_per_firm = df_model.groupby('firm_id')['year'].count().value_counts().sort_index()
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(obs_per_firm.index.astype(str), obs_per_firm.values, color='steelblue', alpha=0.8, edgecolor='white')
ax.set_xlabel('Number of years observed')
ax.set_ylabel('Number of firms')
ax.set_title("Figure 10: Panel Balance — Observations per Firm\n(Vietnam Furniture Industry, 2012–2018)",
             fontweight='bold')
for i, (x, v) in enumerate(zip(obs_per_firm.index, obs_per_firm.values)):
    ax.text(i, v + 1, str(v), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig10_panel_balance.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig10_panel_balance.png")

print("\nDone. Visualization complete.\n")
