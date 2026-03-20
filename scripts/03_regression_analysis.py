"""
Script 03: Econometric Regression Analysis
==========================================
Topic: Factors affecting output of firms in Vietnam's furniture industry

Model:
    ln(Output_it) = β0 + β1*ln(Labor_it) + β2*ln(Capital_it)
                  + β3*Leverage_it + β4*ln(Wage_it) + β5*ln(Size_it)
                  + β6*HHI_t + ε_it

Estimators:
    1. Pooled OLS
    2. Fixed Effects (FE) — within estimator
    3. Random Effects (RE) — GLS estimator
    4. Hausman test (FE vs RE)
    5. Diagnostics: heteroscedasticity, serial correlation, multicollinearity

Note: HHI is year-level (varies by t only). In 2-way FE it is absorbed by
time dummies and dropped automatically.

Output: regression tables saved to output/tables/
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS as LMPooledOLS
from linearmodels.panel import compare
import warnings
import os
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_IN  = os.path.join(BASE_DIR, "output", "data", "furniture_cleaned.csv")
TBL_DIR  = os.path.join(BASE_DIR, "output", "tables")
os.makedirs(TBL_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_IN)
print(f"Loaded: {df.shape[0]:,} obs × {df.shape[1]} cols")

# Merge HHI (year-level) into panel
conc_path = os.path.join(BASE_DIR, "output", "tables", "table_market_concentration.csv")
hhi_df = pd.read_csv(conc_path)[['year', 'HHI']]
df = df.merge(hhi_df, on='year', how='left')
print(f"HHI merged: {df['HHI'].notna().sum()} non-missing")

# Model variables
DEPVAR  = 'lnOutput'
INDVARS = ['lnLabor', 'lnCapital', 'Leverage', 'lnWage', 'lnSize', 'HHI']
# INDVARS without HHI — for 2-way FE where HHI is absorbed by time dummies
INDVARS_NO_HHI = ['lnLabor', 'lnCapital', 'Leverage', 'lnWage', 'lnSize']
ALL_VARS = [DEPVAR] + INDVARS

# Drop any remaining missing values in model variables
df_model = df[['firm_id', 'year'] + ALL_VARS].dropna()
print(f"Model sample: {len(df_model):,} obs, {df_model['firm_id'].nunique():,} firms")

# ── Set up panel index ────────────────────────────────────────────────────────
panel = df_model.set_index(['firm_id', 'year']).sort_index()
Y = panel[DEPVAR]
X_vars = panel[INDVARS]
X_with_const = sm.add_constant(X_vars)

# ════════════════════════════════════════════════════════════════════════════
# MODEL 1: Pooled OLS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL 1: Pooled OLS")
print("="*60)

ols_mod = LMPooledOLS(Y, X_with_const)
ols_res = ols_mod.fit(cov_type='robust')   # HC robust SE
print(ols_res.summary)

# ════════════════════════════════════════════════════════════════════════════
# MODEL 2: Fixed Effects (Two-Way: firm + year FE)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL 2: Fixed Effects (Entity FE + Time FE)")
print("="*60)

# 2-way FE: HHI absorbed by time dummies → exclude it
fe_mod = PanelOLS(Y, panel[INDVARS_NO_HHI], entity_effects=True, time_effects=True)
fe_res = fe_mod.fit(cov_type='robust')   # cluster-robust SE
print(fe_res.summary)

# ── Also run one-way FE (entity only) — includes HHI ──────────────────────
fe1_mod = PanelOLS(Y, X_vars, entity_effects=True, time_effects=False)
fe1_res = fe1_mod.fit(cov_type='robust')

# ════════════════════════════════════════════════════════════════════════════
# MODEL 3: Random Effects (GLS)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL 3: Random Effects (GLS)")
print("="*60)

re_mod  = RandomEffects(Y, X_with_const)
re_res  = re_mod.fit(cov_type='robust')
print(re_res.summary)

# ════════════════════════════════════════════════════════════════════════════
# MODEL 4: Hausman Test (FE vs RE)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL 4: Hausman Test (Fixed Effects vs Random Effects)")
print("="*60)

# Hausman test compares FE and RE coefficients
# Under H0: RE is consistent and efficient (random effects are uncorrelated with X)
# Under H1: FE is consistent but RE is not (use FE)

from linearmodels.panel import compare as panel_compare

# Manual Hausman statistic (compare 1-way FE vs RE on shared variables)
fe1_params = fe1_res.params[INDVARS]
re_params  = re_res.params[INDVARS]
diff_params = fe1_params - re_params

fe1_cov = fe1_res.cov.loc[INDVARS, INDVARS]
re_cov  = re_res.cov.loc[INDVARS, INDVARS]
diff_cov = fe1_cov - re_cov

try:
    from scipy import stats as scipy_stats
    # Use pseudo-inverse for numerical stability
    diff_cov_inv = np.linalg.pinv(diff_cov.values)
    H_stat = float(diff_params.values @ diff_cov_inv @ diff_params.values)
    df_H   = len(INDVARS)
    p_val  = 1 - scipy_stats.chi2.cdf(H_stat, df_H)

    print(f"  Hausman statistic: {H_stat:.4f}")
    print(f"  Degrees of freedom: {df_H}")
    print(f"  p-value: {p_val:.4f}")
    if p_val < 0.05:
        print("  -> Reject H0 at 5%: USE FIXED EFFECTS (FE) model")
        preferred_model = "Fixed Effects"
    else:
        print("  -> Fail to reject H0: Random Effects (RE) is preferred")
        preferred_model = "Random Effects"
    hausman_result = {'H_stat': H_stat, 'df': df_H, 'p_value': p_val, 'preferred': preferred_model}
except Exception as e:
    print(f"  Hausman test error: {e}")
    hausman_result = {'H_stat': np.nan, 'df': len(INDVARS), 'p_value': np.nan, 'preferred': 'N/A'}

# ════════════════════════════════════════════════════════════════════════════
# MODEL 5: Robustness — RE with ln(VA) as output
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL 5: Robustness — Fixed Effects with ln(VA) as output")
print("="*60)

if 'lnVA' in df.columns:
    df_rob = df[['firm_id', 'year', 'lnVA'] + INDVARS].dropna()
    df_rob = df_rob[np.isfinite(df_rob['lnVA'])]
    panel_rob = df_rob.set_index(['firm_id', 'year']).sort_index()
    Y_rob = panel_rob['lnVA']
    X_rob = panel_rob[INDVARS_NO_HHI]   # 2-way FE: HHI absorbed by time dummies
    try:
        fe_rob_mod = PanelOLS(Y_rob, X_rob, entity_effects=True, time_effects=True)
        fe_rob_res = fe_rob_mod.fit(cov_type='robust')
        print(fe_rob_res.summary)
        rob_available = True
    except Exception as e:
        print(f"  Robustness check failed: {e}")
        rob_available = False
        fe_rob_res = None
else:
    rob_available = False
    fe_rob_res = None
    print("  lnVA not available, skipping robustness check")

# ════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC TESTS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("DIAGNOSTICS")
print("="*60)

# 1. VIF — Multicollinearity
print("\n[1] Variance Inflation Factor (VIF)")
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_arr = df_model[INDVARS].values
vif_data = pd.DataFrame({
    'Variable': INDVARS,
    'VIF': [variance_inflation_factor(X_arr, i) for i in range(X_arr.shape[1])]
})
vif_data['VIF'] = vif_data['VIF'].round(3)
print(vif_data.to_string(index=False))
vif_data.to_csv(os.path.join(TBL_DIR, "diag_vif.csv"), index=False)

# 2. Breusch-Pagan (heteroscedasticity) on pooled OLS residuals
print("\n[2] Breusch-Pagan Test for Heteroscedasticity (Pooled OLS)")
ols_sm = sm.OLS(df_model[DEPVAR], sm.add_constant(df_model[INDVARS])).fit()
# (ols_sm uses all INDVARS including HHI)
bp_stat, bp_pval, _, _ = het_breuschpagan(ols_sm.resid, ols_sm.model.exog)
print(f"  LM statistic: {bp_stat:.4f}")
print(f"  p-value:      {bp_pval:.4f}")
if bp_pval < 0.05:
    print("  -> Heteroscedasticity detected — use robust standard errors Done.")
else:
    print("  -> No significant heteroscedasticity")

# 3. Durbin-Watson (serial correlation)
print("\n[3] Durbin-Watson Statistic (Pooled OLS residuals)")
dw_stat = durbin_watson(ols_sm.resid)
print(f"  DW statistic: {dw_stat:.4f}")
if dw_stat < 1.5:
    print("  -> Positive serial correlation detected")
elif dw_stat > 2.5:
    print("  -> Negative serial correlation detected")
else:
    print("  -> No strong serial correlation")

# ════════════════════════════════════════════════════════════════════════════
# COMPILE RESULTS TABLE
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("COMPILED RESULTS")
print("="*60)

VAR_LABELS = {
    'lnLabor':   'ln(Labor)',
    'lnCapital': 'ln(Capital)',
    'Leverage':  'Leverage',
    'lnWage':    'ln(Wage)',
    'lnSize':    'ln(Size)',
    'HHI':       'HHI',
    'const':     'Constant',
}

def extract_results(res, model_name, has_const=True, indvars=None):
    """Extract coefficients, se, t-stat, p-val from linearmodels result."""
    if indvars is None:
        indvars = INDVARS
    rows = []
    params = res.params
    pvals  = res.pvalues
    se     = res.std_errors
    tstats = res.tstats

    for var in (indvars + (['const'] if has_const else [])):
        if var in params.index:
            coef = params[var]
            pv   = pvals[var]
            stars = '***' if pv < 0.01 else ('**' if pv < 0.05 else ('*' if pv < 0.1 else ''))
            rows.append({
                'Variable': VAR_LABELS.get(var, var),
                f'Coef ({model_name})':  f"{coef:.4f}{stars}",
                f'SE ({model_name})':    f"({se[var]:.4f})",
            })
    return pd.DataFrame(rows)

ols_tbl  = extract_results(ols_res,  'OLS',  has_const=True,  indvars=INDVARS)
fe2_tbl  = extract_results(fe_res,   'FE2W', has_const=False, indvars=INDVARS_NO_HHI)
fe1_tbl  = extract_results(fe1_res,  'FE1W', has_const=False, indvars=INDVARS)
re_tbl   = extract_results(re_res,   'RE',   has_const=True,  indvars=INDVARS)

# Merge all
from functools import reduce
tbls = [ols_tbl, fe1_tbl, fe2_tbl, re_tbl]
merged = reduce(lambda a, b: pd.merge(a, b, on='Variable', how='outer'), tbls)

# Goodness of fit rows
def get_stats(res, model_name, n_obs, n_firms):
    return {
        'Variable': f'R² (within)',
        f'Coef ({model_name})': f"{getattr(res, 'rsquared_within', getattr(res, 'rsquared', np.nan)):.4f}",
        f'SE ({model_name})': '',
    }

stat_rows = []
for res, name in [(ols_res,'OLS'), (fe1_res,'FE1W'), (fe_res,'FE2W'), (re_res,'RE')]:
    r2 = getattr(res, 'rsquared_within', getattr(res, 'rsquared', np.nan))
    stat_rows.append({'Variable': 'R² (within)', f'Coef ({name})': f"{r2:.4f}", f'SE ({name})': ''})

# Add N, N_firms
n_row  = {'Variable': 'Observations'}
nf_row = {'Variable': 'Unique Firms'}
for res, name in [(ols_res,'OLS'), (fe1_res,'FE1W'), (fe_res,'FE2W'), (re_res,'RE')]:
    n_row[f'Coef ({name})']  = str(res.nobs)
    n_row[f'SE ({name})']    = ''
    nf_row[f'Coef ({name})'] = str(df_model['firm_id'].nunique())
    nf_row[f'SE ({name})']   = ''

# Hausman row
h_row = {
    'Variable': 'Hausman p-value',
    'Coef (OLS)': '', 'SE (OLS)': '',
    'Coef (FE1W)': f"{hausman_result.get('p_value', np.nan):.4f}", 'SE (FE1W)': '',
    'Coef (FE2W)': '', 'SE (FE2W)': '',
    'Coef (RE)': '', 'SE (RE)': '',
}

extra_rows = pd.DataFrame([stat_rows[0], stat_rows[1], stat_rows[2], stat_rows[3],
                            n_row, nf_row, h_row])
results_table = pd.concat([merged, extra_rows], ignore_index=True)

print(results_table.to_string(index=False))
results_table.to_csv(os.path.join(TBL_DIR, "table6_regression_results.csv"), index=False)
print(f"\n  Saved: table6_regression_results.csv")

# ── Save Hausman summary ──────────────────────────────────────────────────
hausman_df = pd.DataFrame([{
    'Test': 'Hausman (FE vs RE)',
    'Statistic': round(hausman_result.get('H_stat', np.nan), 4),
    'DF': hausman_result.get('df', len(INDVARS)),
    'p-value': round(hausman_result.get('p_value', np.nan), 4),
    'Preferred Model': hausman_result.get('preferred', 'N/A'),
}])
hausman_df.to_csv(os.path.join(TBL_DIR, "table7_hausman_test.csv"), index=False)
print(f"  Saved: table7_hausman_test.csv")

# ── Save diagnostics summary ──────────────────────────────────────────────
diag_df = pd.DataFrame([
    {'Test': 'Breusch-Pagan (heteroscedasticity)', 'Statistic': round(bp_stat,4), 'p-value': round(bp_pval,4)},
    {'Test': 'Durbin-Watson (serial correlation)',  'Statistic': round(dw_stat,4), 'p-value': 'N/A'},
])
diag_df.to_csv(os.path.join(TBL_DIR, "table8_diagnostics.csv"), index=False)
print(f"  Saved: table8_diagnostics.csv")

# Expose results for notebook use
REGRESSION_RESULTS = {
    'ols': ols_res, 'fe1w': fe1_res, 'fe2w': fe_res, 're': re_res,
    'hausman': hausman_result, 'bp': (bp_stat, bp_pval), 'dw': dw_stat,
    'vif': vif_data, 'preferred': preferred_model if 'preferred_model' in dir() else 'N/A',
    'fe_rob': fe_rob_res if rob_available else None,
}

print(f"\nDone. Regression analysis complete. Preferred model: {hausman_result.get('preferred','N/A')}\n")
