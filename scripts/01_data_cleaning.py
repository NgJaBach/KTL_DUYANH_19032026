"""
Script 01: Data Cleaning & Preparation
=====================================
Topic: Factors affecting output of firms in Vietnam's furniture industry
Model: ln(Output) = β0 + β1*ln(Labor) + β2*ln(Capital) + β3*Leverage
                       + β4*ln(Wage) + β5*ln(Size) + ε

Data source: GSO Enterprise Survey 2012-2018 (2012_2018_1a+1am.dta)
Industry: Furniture manufacturing (VSIC 2-digit code = 31)
"""

import pandas as pd
import numpy as np
import pyreadstat
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "Data", "2012_2018_1a+1am.dta")
DATA_OUT = os.path.join(BASE_DIR, "output", "data", "furniture_cleaned.csv")
DATA_LOG  = os.path.join(BASE_DIR, "output", "data", "cleaning_log.txt")

os.makedirs(os.path.dirname(DATA_OUT), exist_ok=True)

log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)


# ── 1. Load raw data ─────────────────────────────────────────────────────────
log("=" * 60)
log("STEP 1: Loading raw data")
log("=" * 60)

COLS = [
    'ma_thue',         # Tax code (firm ID) — available for all years
    'madn',            # Master firm ID (only available for 2012, 2015)
    'year',            # Fiscal year
    'nganh_kd',        # Industry code (5-digit)
    'industry2digit',  # Industry code (2-digit string)
    'tinh',            # Province code
    'lhdn',            # Firm type (SOE / private / FDI)
    'firm_ownership',  # Ownership: 1=SOE, 2=private, 3=FDI
    'soe',             # =1 if SOE
    'private',         # =1 if private
    'fdi',             # =1 if FDI
    # Core model variables
    'net_sales',       # Net sales revenue → OUTPUT
    'labor',           # Number of employees → LABOR
    'end_fixed_assets',# Fixed assets → CAPITAL
    'liabilities',     # Total liabilities
    'end_assets',      # Total assets → SIZE
    'leverage',        # Liabilities / Total assets → LEVERAGE
    'avg_wage',        # Average wage per employee → WAGE
    # Supplementary
    'wage',            # Total wage bill
    'labor_cost',      # Total labor cost (wage + social insurance)
    'VA',              # Value added (robustness)
    'sales',           # Gross sales (robustness)
    'equity',          # Equity
    'capital',         # Total capital
    'firmage',         # Firm age
    'region',          # Geographic region
    'big_city',        # =1 if Hanoi/HCM/Haiphong/Danang/Cantho
    'export',          # Export value
    'import',          # Import value
]

df_raw, meta = pyreadstat.read_dta(DATA_RAW, usecols=COLS)
log(f"  Raw data loaded: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")

# ── 2. Filter furniture industry ─────────────────────────────────────────────
log("\nSTEP 2: Filtering furniture industry (VSIC 31)")
furn = df_raw[df_raw['industry2digit'] == '31'].copy()
log(f"  Furniture observations: {len(furn):,}")
log(f"  Unique firms:           {furn['madn'].nunique():,}")
log(f"  Year range:             {int(furn['year'].min())} – {int(furn['year'].max())}")
log(f"  Industry codes (5-digit): {furn['nganh_kd'].value_counts().to_dict()}")

# ── 3. Construct model variables ─────────────────────────────────────────────
log("\nSTEP 3: Constructing model variables")

# Firm ID — use ma_thue (tax code) which is available for all years
# madn is only populated for 2012 and 2015, so we use ma_thue as the panel identifier
furn = furn.dropna(subset=['ma_thue'])
furn['firm_id'] = furn['ma_thue'].astype(np.int64)
furn['year']    = furn['year'].astype(int)

# Output = net sales revenue (in million VND)
furn['Output']   = furn['net_sales']

# Labor = number of employees at year-end
furn['Labor']    = furn['labor']

# Capital = end-of-year fixed assets (in million VND)
furn['Capital']  = furn['end_fixed_assets']

# Leverage = liabilities / total assets (already computed)
furn['Leverage'] = furn['leverage']

# Wage = average wage per employee (million VND / year)
# Use pre-computed avg_wage; fall back to wage/labor if missing
computed_wage = (furn['wage'] / furn['labor'].replace(0, np.nan)).astype(float)
avg_w = furn['avg_wage'].astype(float)
furn['Wage'] = avg_w.where((avg_w > 0) & avg_w.notna(), other=computed_wage)

# Size = total assets (in million VND)
furn['Size']     = furn['end_assets']

log(f"  Variables created: Output, Labor, Capital, Leverage, Wage, Size")

# ── 4. Data cleaning ─────────────────────────────────────────────────────────
log("\nSTEP 4: Cleaning data")
n_before = len(furn)

# 4a. Drop if core variables are missing
core_vars = ['Output', 'Labor', 'Capital', 'Leverage', 'Wage', 'Size']
furn = furn.dropna(subset=core_vars)
log(f"  After dropping missing core vars: {len(furn):,} (dropped {n_before - len(furn)})")
n_before = len(furn)

# 4b. Drop non-positive values (required for log transform)
for var in ['Output', 'Labor', 'Capital', 'Wage', 'Size']:
    furn = furn[furn[var] > 0]
    n_dropped = n_before - len(furn)
    if n_dropped > 0:
        log(f"  Dropped {n_dropped} obs with {var} <= 0, remaining: {len(furn):,}")
    n_before = len(furn)

# 4c. Leverage must be in [0, 1] range (it's a ratio, but can exceed 1 for distressed firms)
#     Keep leverage >= 0 (drop negative, which is impossible for liabilities/assets)
furn = furn[furn['Leverage'] >= 0]
n_dropped = n_before - len(furn)
if n_dropped > 0:
    log(f"  Dropped {n_dropped} obs with Leverage < 0, remaining: {len(furn):,}")
n_before = len(furn)

log(f"  After cleaning: {len(furn):,} obs (dropped {n_before - len(furn)} total)")

# ── 5. Log-transform variables ───────────────────────────────────────────────
log("\nSTEP 5: Log-transforming variables")
for col in ['Output','Labor','Capital','Wage','Size','VA','end_assets']:
    furn[col] = pd.to_numeric(furn[col], errors='coerce')

furn['lnOutput']  = np.log(furn['Output'].astype(float))
furn['lnLabor']   = np.log(furn['Labor'].astype(float))
furn['lnCapital'] = np.log(furn['Capital'].astype(float))
furn['lnWage']    = np.log(furn['Wage'].astype(float))
furn['lnSize']    = np.log(furn['Size'].astype(float))
# Leverage is NOT log-transformed (it is already a ratio [0, 1+])

# Supplementary log vars
va_vals = pd.to_numeric(furn['VA'], errors='coerce')
furn['lnVA']    = np.where(va_vals > 0, np.log(va_vals), np.nan)
if 'sales' in furn.columns:
    furn['lnSales'] = np.log(pd.to_numeric(furn['sales'], errors='coerce').clip(lower=1))

log(f"  Log variables created: lnOutput, lnLabor, lnCapital, lnWage, lnSize")

# ── 6. Winsorize outliers at 1% – 99% ────────────────────────────────────────
log("\nSTEP 6: Winsorizing log variables at 1st–99th percentile")
log_vars = ['lnOutput', 'lnLabor', 'lnCapital', 'lnWage', 'lnSize', 'Leverage']

def winsorize(series, low=0.01, high=0.99):
    lo, hi = series.quantile([low, high])
    return series.clip(lo, hi)

for var in log_vars:
    furn[var] = winsorize(furn[var])
    log(f"  {var}: [{furn[var].min():.3f}, {furn[var].max():.3f}]")

# ── 7. Panel structure summary ───────────────────────────────────────────────
log("\nSTEP 7: Panel structure")
panel_obs = furn.groupby('firm_id')['year'].count()
log(f"  Total obs:          {len(furn):,}")
log(f"  Unique firms:       {furn['firm_id'].nunique():,}")
log(f"  Obs per firm (avg): {panel_obs.mean():.1f}")
log(f"  Obs per firm (min): {panel_obs.min()}")
log(f"  Obs per firm (max): {panel_obs.max()}")

year_dist = furn['year'].value_counts().sort_index()
log(f"\n  Observations by year:")
for yr, cnt in year_dist.items():
    log(f"    {yr}: {cnt:,}")

# Ownership breakdown
if 'firm_ownership' in furn.columns:
    own_map = {1: 'SOE', 2: 'Private', 3: 'FDI'}
    own_dist = furn['firm_ownership'].map(own_map).value_counts()
    log(f"\n  Ownership breakdown:")
    for k, v in own_dist.items():
        log(f"    {k}: {v:,} ({v/len(furn)*100:.1f}%)")

# ── 8. Save cleaned data ─────────────────────────────────────────────────────
log("\nSTEP 8: Saving cleaned data")
save_cols = [
    'firm_id', 'year', 'tinh', 'nganh_kd', 'firm_ownership', 'soe', 'private', 'fdi', 'ma_thue',
    'region', 'big_city', 'firmage',
    # Raw model vars
    'Output', 'Labor', 'Capital', 'Leverage', 'Wage', 'Size',
    # Log-transformed
    'lnOutput', 'lnLabor', 'lnCapital', 'lnWage', 'lnSize',
    # Supplementary
    'lnVA', 'lnSales', 'VA', 'equity', 'export', 'import',
]
save_cols = [c for c in save_cols if c in furn.columns]
furn[save_cols].to_csv(DATA_OUT, index=False)
log(f"  Saved: {DATA_OUT}")
log(f"  Shape: {furn[save_cols].shape}")

# Save log
with open(DATA_LOG, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
log(f"\n  Log saved: {DATA_LOG}")
log("\nDone. Data cleaning complete.\n")
