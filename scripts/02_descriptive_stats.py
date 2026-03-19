"""
Script 02: Descriptive Statistics & Exploratory Data Analysis
=============================================================
Topic: Factors affecting output of firms in Vietnam's furniture industry
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_IN  = os.path.join(BASE_DIR, "output", "data", "furniture_cleaned.csv")
TBL_DIR  = os.path.join(BASE_DIR, "output", "tables")
FIG_DIR  = os.path.join(BASE_DIR, "output", "figures")
os.makedirs(TBL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 120,
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
PALETTE = 'Blues_d'

df = pd.read_csv(DATA_IN)
print(f"Loaded cleaned data: {df.shape[0]:,} obs × {df.shape[1]} cols")

MODEL_VARS_RAW = ['Output', 'Labor', 'Capital', 'Leverage', 'Wage', 'Size']
MODEL_VARS_LOG = ['lnOutput', 'lnLabor', 'lnCapital', 'Leverage', 'lnWage', 'lnSize']

LABELS = {
    'Output':    'Output (net sales, mil VND)',
    'Labor':     'Labor (employees)',
    'Capital':   'Capital (fixed assets, mil VND)',
    'Leverage':  'Leverage (liab/assets)',
    'Wage':      'Wage (mil VND/employee)',
    'Size':      'Size (total assets, mil VND)',
    'lnOutput':  'ln(Output)',
    'lnLabor':   'ln(Labor)',
    'lnCapital': 'ln(Capital)',
    'lnWage':    'ln(Wage)',
    'lnSize':    'ln(Size)',
}

# ── Table 1: Descriptive Statistics (raw variables) ──────────────────────────
print("\n--- Table 1: Descriptive Statistics (Raw Variables) ---")

desc = df[MODEL_VARS_RAW].describe(percentiles=[0.25, 0.5, 0.75]).T
desc.index = [LABELS.get(v, v) for v in MODEL_VARS_RAW]
desc.columns = ['N', 'Mean', 'Std Dev', 'Min', 'Q25', 'Median', 'Q75', 'Max']
desc = desc.round(2)
print(desc.to_string())
desc.to_csv(os.path.join(TBL_DIR, "table1_descriptive_raw.csv"))

# ── Table 2: Descriptive Statistics (log-transformed variables) ──────────────
print("\n--- Table 2: Descriptive Statistics (Log Variables) ---")

log_vars_in_df = [v for v in MODEL_VARS_LOG if v in df.columns]
desc_log = df[log_vars_in_df].describe(percentiles=[0.25, 0.5, 0.75]).T
desc_log.index = [LABELS.get(v, v) for v in log_vars_in_df]
desc_log.columns = ['N', 'Mean', 'Std Dev', 'Min', 'Q25', 'Median', 'Q75', 'Max']
desc_log = desc_log.round(4)
print(desc_log.to_string())
desc_log.to_csv(os.path.join(TBL_DIR, "table2_descriptive_log.csv"))

# ── Table 3: Observations by year ─────────────────────────────────────────────
print("\n--- Table 3: Panel structure by year ---")
yr_tbl = df.groupby('year').agg(
    N_obs=('firm_id', 'count'),
    N_firms=('firm_id', 'nunique'),
    Mean_Output=('Output', 'mean'),
    Mean_Labor=('Labor', 'mean'),
    Mean_Leverage=('Leverage', 'mean'),
).round(2)
print(yr_tbl.to_string())
yr_tbl.to_csv(os.path.join(TBL_DIR, "table3_panel_by_year.csv"))

# ── Table 4: By ownership ─────────────────────────────────────────────────────
if 'firm_ownership' in df.columns:
    print("\n--- Table 4: Summary by ownership ---")
    own_map = {1.0: 'SOE', 2.0: 'Private', 3.0: 'FDI'}
    df['ownership_label'] = df['firm_ownership'].map(own_map)
    own_tbl = df.groupby('ownership_label')[MODEL_VARS_RAW].mean().round(2)
    print(own_tbl.to_string())
    own_tbl.to_csv(os.path.join(TBL_DIR, "table4_by_ownership.csv"))

# ── Table 5: Correlation matrix (log vars) ──────────────────────────────────
print("\n--- Table 5: Correlation matrix ---")
corr = df[log_vars_in_df].corr().round(3)
corr.index   = [LABELS.get(v, v) for v in log_vars_in_df]
corr.columns = [LABELS.get(v, v) for v in log_vars_in_df]
print(corr.to_string())
corr.to_csv(os.path.join(TBL_DIR, "table5_correlation.csv"))

# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════

# ── Figure 1: Distribution of log variables ──────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, var in enumerate(log_vars_in_df):
    ax = axes[i]
    data_clean = df[var].dropna()
    ax.hist(data_clean, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(data_clean.mean(), color='darkred', linestyle='--', linewidth=1.5, label=f'Mean={data_clean.mean():.2f}')
    ax.set_title(LABELS.get(var, var), fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
plt.suptitle("Figure 1: Distribution of Model Variables (Log-transformed)\nVietnam Furniture Industry (VSIC 31), 2012–2018",
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_distributions.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig1_distributions.png")

# ── Figure 2: Correlation heatmap ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
mask = np.triu(np.ones_like(corr.values, dtype=bool), k=1)
sns.heatmap(corr.values, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            xticklabels=corr.columns, yticklabels=corr.index,
            linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8},
            vmin=-1, vmax=1)
ax.set_title("Figure 2: Pearson Correlation Matrix\n(Model Variables — Log-transformed)",
             fontweight='bold', pad=15)
plt.xticks(rotation=30, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_correlation_heatmap.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig2_correlation_heatmap.png")

# ── Figure 3: Trends over time ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
trend_vars = [('Output', 'Mean Output\n(mil VND)', 'steelblue'),
              ('Labor', 'Mean Labor\n(employees)', 'seagreen'),
              ('Leverage', 'Mean Leverage\n(ratio)', 'tomato')]

for ax, (var, ylabel, color) in zip(axes, trend_vars):
    yr_grp = df.groupby('year')[var]
    means  = yr_grp.mean()
    sems   = yr_grp.sem()
    ax.plot(means.index, means.values, marker='o', color=color, linewidth=2)
    ax.fill_between(means.index,
                    (means - 1.96*sems).values,
                    (means + 1.96*sems).values,
                    alpha=0.15, color=color)
    ax.set_title(f'Mean {var} by Year', fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Year')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.suptitle("Figure 3: Key Variable Trends — Vietnam Furniture Industry (2012–2018)",
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_trends_over_time.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig3_trends_over_time.png")

# ── Figure 4: ln(Output) vs each predictor (scatter) ────────────────────────
predictors = ['lnLabor', 'lnCapital', 'Leverage', 'lnWage', 'lnSize']
pred_in_df = [p for p in predictors if p in df.columns]

fig, axes = plt.subplots(1, len(pred_in_df), figsize=(16, 4))
for ax, pred in zip(axes, pred_in_df):
    sample = df[['lnOutput', pred]].dropna().sample(min(500, len(df)), random_state=42)
    ax.scatter(sample[pred], sample['lnOutput'], alpha=0.35, s=15, color='steelblue')
    # Fit line
    m, b = np.polyfit(sample[pred].dropna(), sample['lnOutput'].dropna(), 1)
    x_line = np.linspace(sample[pred].min(), sample[pred].max(), 100)
    ax.plot(x_line, m*x_line + b, color='red', linewidth=1.5, linestyle='--')
    ax.set_xlabel(LABELS.get(pred, pred))
    ax.set_ylabel('ln(Output)')
    ax.set_title(f'ln(Output) vs {LABELS.get(pred, pred)[:12]}', fontweight='bold')

plt.suptitle("Figure 4: ln(Output) vs Explanatory Variables (sample n=500)",
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig4_scatter_plots.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig4_scatter_plots.png")

# ── Figure 5: Ownership comparison ───────────────────────────────────────────
if 'ownership_label' in df.columns:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, var in zip(axes, ['Output', 'Labor', 'Leverage']):
        grp_data = [df.loc[df['ownership_label'] == g, var].dropna().values
                    for g in ['SOE', 'Private', 'FDI'] if g in df['ownership_label'].values]
        grp_labels = [g for g in ['SOE', 'Private', 'FDI'] if g in df['ownership_label'].values]
        ax.boxplot(grp_data, labels=grp_labels, patch_artist=True,
                   boxprops=dict(facecolor='steelblue', alpha=0.6),
                   medianprops=dict(color='darkred', linewidth=2))
        ax.set_title(f'{var} by Ownership', fontweight='bold')
        ax.set_ylabel(LABELS.get(var, var))
    plt.suptitle("Figure 5: Distribution by Firm Ownership — Vietnam Furniture Industry",
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_ownership_boxplots.png"), bbox_inches='tight')
    plt.close()
    print("  Saved: fig5_ownership_boxplots.png")

print("\nDone. Descriptive statistics complete.\n")
