#!/usr/bin/env bash
# ============================================================
# run_pipeline.sh
# Full pipeline for Vietnam Furniture Industry — Econometric Analysis
# Run: bash run_pipeline.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Vietnam Furniture Industry — Econometric Analysis Pipeline"
echo "============================================================"
echo ""

# ── Create output directories ────────────────────────────────
mkdir -p output/data output/tables output/figures

echo "[1/4] Data Cleaning & Preparation..."
python scripts/01_data_cleaning.py
echo "      Done."
echo ""

echo "[2/4] Descriptive Statistics & EDA..."
python scripts/02_descriptive_stats.py
echo "      Done."
echo ""

echo "[3/4] Regression Analysis (OLS / FE / RE / Hausman)..."
python scripts/03_regression_analysis.py
echo "      Done."
echo ""

echo "[4/4] Visualization..."
python scripts/04_visualization.py
echo "      Done."
echo ""

echo "[5/5] Generating Vietnamese PDF Report..."
python scripts/05_generate_report_pdf.py
echo "      Done."
echo ""

# ── Optionally execute notebook ──────────────────────────────
if command -v jupyter &> /dev/null; then
    echo "[+] Converting notebook to HTML report..."
    jupyter nbconvert --to html --execute notebooks/analysis.ipynb \
        --output-dir output/ --output analysis_report.html 2>/dev/null && \
    echo "      Saved: output/analysis_report.html" || \
    echo "      (Notebook execution skipped — check Jupyter)"
fi

echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "  Output files:"
echo "    output/data/furniture_cleaned.csv       -- Cleaned dataset"
echo "    output/tables/                           -- Regression tables (CSV)"
echo "    output/figures/                          -- Charts & plots (PNG)"
echo "    output/BaoCao_NganhGoNoiThat.pdf         -- MAIN REPORT (PDF, Vietnamese)"
echo "    output/analysis_report.html              -- Interactive notebook report"
echo "============================================================"
