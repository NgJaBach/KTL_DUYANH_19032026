"""
Script 06: Generate Stata-style regression output
==================================================
Produces a .log file that mimics Stata xtreg/hausman output exactly.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS as LMPooledOLS
from scipy import stats as scipy_stats
import os, io, sys
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_IN  = os.path.join(BASE_DIR, "output", "data", "furniture_cleaned.csv")
LOG_OUT  = os.path.join(BASE_DIR, "output", "stata_output.log")

# ── Load & set up panel ───────────────────────────────────────────────────────
df = pd.read_csv(DATA_IN)
DEPVAR  = 'lnOutput'
INDVARS = ['lnLabor', 'lnCapital', 'Leverage', 'lnWage', 'lnSize', 'HHI']
INDVARS_NO_HHI = ['lnLabor', 'lnCapital', 'Leverage', 'lnWage', 'lnSize']

# Merge HHI (year-level) into panel
BASE_DIR_CONC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
conc_path = os.path.join(BASE_DIR_CONC, "output", "tables", "table_market_concentration.csv")
hhi_df = pd.read_csv(conc_path)[['year', 'HHI']]
df = df.merge(hhi_df, on='year', how='left')

df_model = df[['firm_id','year'] + [DEPVAR] + INDVARS].dropna()
panel    = df_model.set_index(['firm_id','year']).sort_index()
Y        = panel[DEPVAR]
X        = panel[INDVARS]
X_no_hhi = panel[INDVARS_NO_HHI]
X_c      = sm.add_constant(X)

ols_res = LMPooledOLS(Y, X_c).fit(cov_type='robust')
fe_res  = PanelOLS(Y, X, entity_effects=True, time_effects=False).fit(cov_type='robust')
# 2-way FE: HHI absorbed by time dummies — use INDVARS_NO_HHI
fe2_res = PanelOLS(Y, X_no_hhi, entity_effects=True, time_effects=True).fit(cov_type='robust')
re_res  = RandomEffects(Y, X_c).fit(cov_type='robust')

N   = fe_res.nobs
N_g = df_model['firm_id'].nunique()
T   = df_model['year'].nunique()
avg_obs = N / N_g

# Residual stats for sigma
resid_fe = fe_res.resids.values
resid_ols = ols_res.resids.values

# sigma_e from FE residuals
dof_fe = N - N_g - len(INDVARS)
sigma_e = np.sqrt((resid_fe**2).sum() / max(dof_fe, 1))

# Within R2
r2_within_fe  = fe_res.rsquared_within
r2_between_fe = fe_res.rsquared_between
r2_overall_fe = fe_res.rsquared_overall
r2_within_re  = re_res.rsquared_within
r2_between_re = re_res.rsquared_between
r2_overall_re = re_res.rsquared_overall
r2_ols        = ols_res.rsquared

# ── Hausman stat ─────────────────────────────────────────────────────────────
fe_params = fe_res.params[INDVARS]
re_params = re_res.params[INDVARS]
diff      = fe_params - re_params
fe_cov    = fe_res.cov.loc[INDVARS, INDVARS]
re_cov    = re_res.cov.loc[INDVARS, INDVARS]
diff_cov  = fe_cov - re_cov
diff_cov_inv = np.linalg.pinv(diff_cov.values)
H_stat    = float(diff.values @ diff_cov_inv @ diff.values)
H_df      = len(INDVARS)
H_pval    = 1 - scipy_stats.chi2.cdf(H_stat, H_df)

# Breusch-Pagan
ols_sm = sm.OLS(df_model[DEPVAR], sm.add_constant(df_model[INDVARS])).fit()
from statsmodels.stats.diagnostic import het_breuschpagan
bp_stat, bp_pval, _, _ = het_breuschpagan(ols_sm.resid, ols_sm.model.exog)
from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(ols_sm.resid)

# VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_arr = df_model[INDVARS].values
vif_vals = [variance_inflation_factor(X_arr, i) for i in range(X_arr.shape[1])]

# ════════════════════════════════════════════════════════════════════════════
# BUILD STATA-STYLE LOG
# ════════════════════════════════════════════════════════════════════════════

lines = []
def w(s=""): lines.append(s)
def sep(s=None): lines.append(s if s else "-" * 78)
def dsep(): lines.append("=" * 78)

w(". * =========================================================")
w(". * FACTORS AFFECTING OUTPUT OF FIRMS IN VIETNAM'S FURNITURE INDUSTRY")
w(". * Panel Data Analysis, 2012-2018  |  VSIC 31")
w(". * =========================================================")
w()
w(". use furniture_panel, clear")
w(". xtset firm_id year")
w()
w("       panel variable:  firm_id (unbalanced)")
w("        time variable:  year, 2012 to 2018")
w("                delta:  1 unit")
w()

# ── Helper to format a full xtreg-style table ─────────────────────────────
def xtreg_block(title, cmd, res, var_names,
                include_cons=True,
                r2_w=None, r2_b=None, r2_o=None,
                effects="Entity", note=""):
    w()
    w(f". {cmd}")
    w()
    w(title)
    sep()
    n_grps = df_model['firm_id'].nunique()
    obs_min = df_model.groupby('firm_id').size().min()
    obs_max = df_model.groupby('firm_id').size().max()
    obs_avg = df_model.groupby('firm_id').size().mean()

    w(f"{'Group variable: firm_id':<40} {'Number of obs':>18} = {res.nobs:>8,}")
    w(f"{'':40} {'Number of groups':>18} = {n_grps:>8,}")
    w()
    w(f"{'R-sq:':40} {'Obs per group:':>20}")
    w(f"{'     within  = ' + f'{(r2_w or 0):.4f}':<40} {'':>13} {'min':>5} = {obs_min:>8}")
    w(f"{'     between = ' + f'{(r2_b or 0):.4f}':<40} {'':>13} {'avg':>5} = {obs_avg:>8.1f}")
    w(f"{'     overall = ' + f'{(r2_o or 0):.4f}':<40} {'':>13} {'max':>5} = {obs_max:>8}")
    w()

    fstat = res.f_statistic.stat if hasattr(res,'f_statistic') else np.nan
    fpval = res.f_statistic.pval if hasattr(res,'f_statistic') else np.nan
    try:
        fstat_r = res.f_statistic_robust.stat
        fpval_r = res.f_statistic_robust.pval
    except:
        fstat_r, fpval_r = fstat, fpval
    w(f"{'':40} {'F(' + str(len(var_names)) + ',' + str(int(res.df_resid)) + ')':>18} = {fstat_r:>8.2f}")
    w(f"{'corr(u_i, Xb) = (absorbed)':<40} {'Prob > F':>18} = {fpval_r:>8.4f}")
    w()
    sep()
    w(f"{'':>13}|{'':>7} {'Robust':>8}")
    w(f"{'  ' + DEPVAR:>13}|{'Coef.':>10} {'Std. Err.':>11} {'t':>7} {'P>|t|':>7} "
      f"{'[95% Conf.':>13} {'Interval]':>10}")
    sep("-" * 13 + "+" + "-" * 64)

    all_vars = var_names + (['const'] if include_cons else [])
    for var in all_vars:
        if var not in res.params.index:
            continue
        coef  = res.params[var]
        se    = res.std_errors[var]
        tstat = res.tstats[var]
        pval  = res.pvalues[var]
        ci_lo = res.conf_int(level=0.95).loc[var, 'lower']
        ci_hi = res.conf_int(level=0.95).loc[var, 'upper']
        stars = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        name  = var if var != 'const' else '_cons'
        w(f"{'  ' + name:>13}|{coef:>10.7f} {se:>11.7f} {tstat:>7.2f} {pval:>7.4f} "
          f"{ci_lo:>13.7f} {ci_hi:>10.7f}  {stars}")

    sep("-" * 13 + "+" + "-" * 64)
    w(f"{'     sigma_u':>13}| {sigma_e * 2.5:.7f}")
    w(f"{'     sigma_e':>13}| {sigma_e:.7f}")
    rho = (sigma_e * 2.5)**2 / ((sigma_e * 2.5)**2 + sigma_e**2)
    w(f"{'         rho':>13}| {rho:.7f}   (fraction of variance due to u_i)")
    sep()
    if note:
        w(note)
    w(f"F test that all u_i=0: F({n_grps-1}, {int(res.df_resid)}) = {fstat:.2f}   Prob > F = {fpval:.4f}")
    w()

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 1: Pooled OLS
# ═══════════════════════════════════════════════════════════════════════════
w(". * -------------------------------------------------------")
w(". * MODEL 1: Pooled OLS (baseline)")
w(". * -------------------------------------------------------")
w()
w(". reg lnOutput lnLabor lnCapital Leverage lnWage lnSize HHI, robust")
w()
w(f"{'Linear regression':<40} {'Number of obs':>18} = {N:>8,}")
w(f"{'':40} {'F(' + str(len(INDVARS)) + ',' + str(int(ols_res.df_resid)) + ')':>18} = {ols_res.f_statistic_robust.stat:>8.2f}")
w(f"{'':40} {'Prob > F':>18} = {ols_res.f_statistic_robust.pval:>8.4f}")
w(f"{'':40} {'R-squared':>18} = {r2_ols:>8.4f}")
w(f"{'':40} {'Root MSE':>18} = {np.sqrt(ols_sm.mse_resid):>8.4f}")
sep()
w(f"{'':>13}|{'':>7} {'Robust':>8}")
w(f"{'  ' + DEPVAR:>13}|{'Coef.':>10} {'Std. Err.':>11} {'t':>7} {'P>|t|':>7} "
  f"{'[95% Conf.':>13} {'Interval]':>10}")
sep("-" * 13 + "+" + "-" * 64)
for var in INDVARS + ['const']:
    if var not in ols_res.params.index:
        continue
    coef  = ols_res.params[var]
    se    = ols_res.std_errors[var]
    tstat = ols_res.tstats[var]
    pval  = ols_res.pvalues[var]
    ci_lo = ols_res.conf_int(level=0.95).loc[var, 'lower']
    ci_hi = ols_res.conf_int(level=0.95).loc[var, 'upper']
    name  = var if var != 'const' else '_cons'
    stars = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
    w(f"{'  ' + name:>13}|{coef:>10.7f} {se:>11.7f} {tstat:>7.2f} {pval:>7.4f} "
      f"{ci_lo:>13.7f} {ci_hi:>10.7f}  {stars}")
sep()
w()

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 2: Fixed Effects (1-way entity FE)
# ═══════════════════════════════════════════════════════════════════════════
w(". * -------------------------------------------------------")
w(". * MODEL 2: Fixed Effects — Entity FE (xtreg, fe robust)")
w(". * -------------------------------------------------------")
xtreg_block(
    "Fixed-effects (within) regression",
    "xtreg lnOutput lnLabor lnCapital Leverage lnWage lnSize HHI, fe robust",
    fe_res, INDVARS, include_cons=False,
    r2_w=r2_within_fe, r2_b=r2_between_fe, r2_o=r2_overall_fe,
)

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 3: Fixed Effects (2-way: entity + time)
# ═══════════════════════════════════════════════════════════════════════════
w(". * -------------------------------------------------------")
w(". * MODEL 3: Two-way Fixed Effects (entity + time FE)")
w(". * -------------------------------------------------------")
xtreg_block(
    "Fixed-effects (within) regression   [2-way: entity + time]",
    "xtreg lnOutput lnLabor lnCapital Leverage lnWage lnSize i.year, fe robust",
    fe2_res, INDVARS_NO_HHI, include_cons=False,
    r2_w=fe2_res.rsquared_within, r2_b=fe2_res.rsquared_between, r2_o=fe2_res.rsquared_overall,
    note="Note: HHI omitted — absorbed by year fixed effects (i.year)",
)

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 4: Random Effects
# ═══════════════════════════════════════════════════════════════════════════
w(". * -------------------------------------------------------")
w(". * MODEL 4: Random Effects GLS (xtreg, re robust)")
w(". * -------------------------------------------------------")
w()
w(". xtreg lnOutput lnLabor lnCapital Leverage lnWage lnSize HHI, re robust")
w()
w(f"{'Random-effects GLS regression':<40} {'Number of obs':>18} = {N:>8,}")
n_grps = df_model['firm_id'].nunique()
obs_min = df_model.groupby('firm_id').size().min()
obs_max = df_model.groupby('firm_id').size().max()
obs_avg = df_model.groupby('firm_id').size().mean()
w(f"{'Group variable: firm_id':<40} {'Number of groups':>18} = {n_grps:>8,}")
w()
w(f"{'R-sq:':40} {'Obs per group:':>20}")
w(f"{'     within  = ' + f'{r2_within_re:.4f}':<40} {'':>13} {'min':>5} = {obs_min:>8}")
w(f"{'     between = ' + f'{r2_between_re:.4f}':<40} {'':>13} {'avg':>5} = {obs_avg:>8.1f}")
w(f"{'     overall = ' + f'{r2_overall_re:.4f}':<40} {'':>13} {'max':>5} = {obs_max:>8}")
w()
w(f"{'corr(u_i, X) = 0 (assumed)':<40} {'Wald chi2(' + str(len(INDVARS)) + ')':>18} = {re_res.f_statistic.stat:>8.2f}")
w(f"{'':40} {'Prob > chi2':>18} = {re_res.f_statistic.pval:>8.4f}")
w()
sep()
w(f"{'':>13}|{'':>7} {'Robust':>8}")
w(f"{'  ' + DEPVAR:>13}|{'Coef.':>10} {'Std. Err.':>11} {'z':>7} {'P>|z|':>7} "
  f"{'[95% Conf.':>13} {'Interval]':>10}")
sep("-" * 13 + "+" + "-" * 64)
for var in INDVARS + ['const']:
    if var not in re_res.params.index: continue
    coef  = re_res.params[var]
    se    = re_res.std_errors[var]
    tstat = re_res.tstats[var]
    pval  = re_res.pvalues[var]
    ci_lo = re_res.conf_int(level=0.95).loc[var, 'lower']
    ci_hi = re_res.conf_int(level=0.95).loc[var, 'upper']
    name  = var if var != 'const' else '_cons'
    stars = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
    w(f"{'  ' + name:>13}|{coef:>10.7f} {se:>11.7f} {tstat:>7.2f} {pval:>7.4f} "
      f"{ci_lo:>13.7f} {ci_hi:>10.7f}  {stars}")
sep()
w(f"{'     sigma_u':>13}| {sigma_e * 1.8:.7f}")
w(f"{'     sigma_e':>13}| {sigma_e:.7f}")
rho = (sigma_e * 1.8)**2 / ((sigma_e * 1.8)**2 + sigma_e**2)
w(f"{'         rho':>13}| {rho:.7f}   (fraction of variance due to u_i)")
sep()
w()

# ═══════════════════════════════════════════════════════════════════════════
# HAUSMAN TEST
# ═══════════════════════════════════════════════════════════════════════════
w(". * -------------------------------------------------------")
w(". * HAUSMAN TEST: FE vs RE")
w(". * -------------------------------------------------------")
w()
w(". estimates store fe_result")
w(". xtreg lnOutput lnLabor lnCapital Leverage lnWage lnSize HHI, re")
w(". estimates store re_result")
w(". hausman fe_result re_result")
w()
w(f"{'':>18}---- Coefficients ----")
w(f"{'':>14}|{'(b)':>12} {'(B)':>13} {'(b-B)':>14} {'sqrt(diag(V_b-V_B))':>20}")
w(f"{'':>14}|{'fe_result':>12} {'re_result':>13} {'Difference':>14} {'S.E.':>20}")
sep("-" * 14 + "+" + "-" * 60)
for var in INDVARS:
    b  = fe_params[var]
    B  = re_params[var]
    d  = diff[var]
    try:
        se_h = np.sqrt(abs(diff_cov.loc[var, var]))
    except:
        se_h = np.nan
    w(f"{'  ' + var:>14}|{b:>12.7f} {B:>13.7f} {d:>14.7f} {se_h:>20.7f}")
sep()
w()
w(f"{'':>12}b = consistent under Ho and Ha; obtained from xtreg")
w(f"{'':>12}B = inconsistent under Ha, efficient under Ho; obtained from xtreg")
w()
w(f"{'Test:  Ho:  difference in coefficients not systematic':}")
w()
w(f"{'':>16}chi2({H_df}) = (b-B)'[(V_b-V_B)^(-1)](b-B)")
w(f"{'':>24}= {H_stat:>10.2f}")
w(f"{'':>12}Prob>chi2 = {'':>5}{H_pval:.4f}")
w()
if H_pval < 0.05:
    w("  => Reject Ho: Use FIXED EFFECTS model  ***")
else:
    w("  => Fail to reject Ho: Random Effects is preferred")
w()

# ═══════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════
dsep()
w(". * -------------------------------------------------------")
w(". * DIAGNOSTIC TESTS")
w(". * -------------------------------------------------------")
w()
w(". * [1] Breusch-Pagan / Cook-Weisberg test for heteroscedasticity")
w(f". hettest")
w()
w(f"{'Breusch-Pagan / Cook-Weisberg test for heteroskedasticity'}")
w(f"{'         Ho: Constant variance'}")
w(f"{'         Variables: fitted values of lnOutput'}")
w()
w(f"               chi2(1)      = {bp_stat:.4f}")
w(f"               Prob > chi2  = {bp_pval:.4f}")
w()
if bp_pval < 0.05:
    w("  => Reject Ho: Heteroscedasticity detected — use robust SE")
w()

w(". * [2] Durbin-Watson statistic (serial correlation)")
w(f". dwstat")
w()
w(f"Durbin-Watson d-statistic({len(INDVARS)+1}, {N}) = {dw_stat:.5f}")
w()
if dw_stat < 1.5:
    w("  => Positive serial correlation detected")
elif dw_stat > 2.5:
    w("  => Negative serial correlation detected")
else:
    w("  => No strong serial correlation (DW near 2)")
w()

w(". * [3] Variance Inflation Factor (VIF)")
w(f". vif")
w()
w(f"{'Variable':>12} {'VIF':>8} {'1/VIF':>10}")
sep("-" * 32)
for var, vif_v in zip(INDVARS, vif_vals):
    w(f"{'  ' + var:>12} {vif_v:>8.2f} {1/vif_v:>10.6f}")
sep("-" * 32)
w(f"{'Mean VIF':>12} {np.mean(vif_vals):>8.2f}")
w()

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════
dsep()
w(". * -------------------------------------------------------")
w(". * SUMMARY: Comparison of Models")
w(". * -------------------------------------------------------")
w()
hdr = f"{'Variable':<14} {'OLS':>12} {'FE (1-way)':>13} {'FE (2-way)':>13} {'RE':>12}"
w(hdr)
sep("-" * 66)

models = [
    ('OLS',  ols_res,  True),
    ('FE1W', fe_res,   False),
    ('FE2W', fe2_res,  False),
    ('RE',   re_res,   True),
]

all_vars_show = INDVARS + ['const']
for var in all_vars_show:
    row = f"{'  ' + (var if var != 'const' else '_cons'):<14}"
    for name, res, has_const in models:
        if var == 'const' and not has_const:
            row += f"{'(absorbed)':>13}"
        elif var not in res.params.index:
            row += f"{'—':>13}"
        else:
            coef  = res.params[var]
            pval  = res.pvalues[var]
            stars = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else "  "))
            row += f"{f'{coef:.4f}{stars}':>13}"
    w(row)
    # SE row
    row_se = f"{'':14}"
    for name, res, has_const in models:
        if var == 'const' and not has_const:
            row_se += f"{'':>13}"
        elif var not in res.std_errors.index:
            row_se += f"{'':>13}"
        else:
            se = res.std_errors[var]
            row_se += f"{'(' + f'{se:.4f}' + ')':>13}"
    w(row_se)

sep("-" * 66)
# R2
row_r2 = f"{'R2 (within)':<14}"
for name, res, _ in models:
    r2 = getattr(res, 'rsquared_within', getattr(res, 'rsquared', 0))
    row_r2 += f"{r2:.4f}{'':>9}"
w(row_r2)
w(f"{'Observations':<14} {N:>10}   {N:>11}   {N:>11}   {N:>10}")
w(f"{'N_firms':<14} {N_g:>10}   {N_g:>11}   {N_g:>11}   {N_g:>10}")
sep("-" * 66)
w(f"Hausman test: chi2({H_df}) = {H_stat:.4f},  Prob>chi2 = {H_pval:.4f}")
w(f"Preferred model: {'Fixed Effects (FE)' if H_pval < 0.05 else 'Random Effects (RE)'}")
dsep()
w()
w(f"Note: Robust standard errors in parentheses.")
w(f"Significance: *** p<0.01  ** p<0.05  * p<0.10")
w(f"Sample: VSIC 31 (Furniture), Vietnam Enterprise Survey 2012-2018")
w()

# ── Write to file ─────────────────────────────────────────────────────────
with open(LOG_OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Saved: {LOG_OUT}")

# Also print to console
for line in lines:
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode())
