"""
Script 05: Tao bao cao PDF tieng Viet
=====================================
Bao cao: Cac nhan to anh huong den san luong doanh nghiep nganh go noi that Viet Nam
Su dung fpdf2 voi font Arial (ho tro tieng Viet)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from fpdf import FPDF, XPos, YPos
from fpdf.enums import XPos, YPos
import warnings
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_IN  = os.path.join(BASE_DIR, "output", "data",    "furniture_cleaned.csv")
TBL_DIR  = os.path.join(BASE_DIR, "output", "tables")
FIG_DIR  = os.path.join(BASE_DIR, "output", "figures")
PDF_OUT  = os.path.join(BASE_DIR, "output", "BaoCao_NganhGoNoiThat.pdf")

FONT_REG  = "C:/Windows/Fonts/arial.ttf"
FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
FONT_ITAL = "C:/Windows/Fonts/ariali.ttf"

# ── Load data & results ──────────────────────────────────────────────────────
df          = pd.read_csv(DATA_IN)
desc_raw    = pd.read_csv(os.path.join(TBL_DIR, "table1_descriptive_raw.csv"), index_col=0)
desc_log    = pd.read_csv(os.path.join(TBL_DIR, "table2_descriptive_log.csv"), index_col=0)
yr_tbl      = pd.read_csv(os.path.join(TBL_DIR, "table3_panel_by_year.csv"))
corr_tbl    = pd.read_csv(os.path.join(TBL_DIR, "table5_correlation.csv"), index_col=0)
reg_tbl     = pd.read_csv(os.path.join(TBL_DIR, "table6_regression_results.csv"))
hausman_tbl = pd.read_csv(os.path.join(TBL_DIR, "table7_hausman_test.csv"))
diag_tbl    = pd.read_csv(os.path.join(TBL_DIR, "table8_diagnostics.csv"))
vif_tbl     = pd.read_csv(os.path.join(TBL_DIR, "diag_vif.csv"))
conc_tbl    = pd.read_csv(os.path.join(TBL_DIR, "table_market_concentration.csv"))

# Market share per firm per year
df['market_share'] = df.groupby('year')['Output'].transform(lambda x: x / x.sum())

print("Du lieu va bang ket qua da load xong.")

# ═══════════════════════════════════════════════════════════════════════════
# Custom PDF class
# ═══════════════════════════════════════════════════════════════════════════

class ReportPDF(FPDF):
    def __init__(self):
        super().__init__(format='A4')
        self.set_auto_page_break(auto=True, margin=20)
        self.add_font("Arial",  "",  FONT_REG)
        self.add_font("Arial",  "B", FONT_BOLD)
        self.add_font("Arial",  "I", FONT_ITAL)
        self.set_margins(20, 15, 20)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Arial", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8,
            "Báo cáo Kinh Tế Lượng — Nhân tố ảnh hưởng đến sản lượng doanh nghiệp ngành gỗ nội thất Việt Nam",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-13)
        self.set_font("Arial", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Trang {self.page_no()}", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)

    # ── Helpers ──────────────────────────────────────────────────────────────
    def h1(self, text):
        self.set_font("Arial", "B", 15)
        self.set_fill_color(13, 71, 161)   # dark blue
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, text, fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def h2(self, text):
        self.set_font("Arial", "B", 12)
        self.set_text_color(13, 71, 161)
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)
        y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(3)

    def h3(self, text):
        self.set_font("Arial", "B", 10)
        self.set_text_color(33, 37, 41)
        self.cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def body(self, text, size=10, indent=0):
        self.set_font("Arial", "", size)
        if indent:
            self.set_x(self.l_margin + indent)
        self.multi_cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def table(self, headers, rows, col_widths=None, header_color=(13, 71, 161)):
        """Draw table with multi-line cell support."""
        usable = self.w - self.l_margin - self.r_margin
        if col_widths is None:
            col_widths = [usable / len(headers)] * len(headers)

        LH = 5.5  # line height per text line

        def calc_row_h(row, fsize):
            self.set_font("Arial", "", fsize)
            max_l = 1
            for cell, cw in zip(row, col_widths):
                txt = str(cell)
                parts = txt.split('\n')
                n = len(parts)
                for part in parts:
                    sw = self.get_string_width(part) + 2
                    if sw > cw - 2:
                        n += int(sw / (cw - 2))
                max_l = max(max_l, n)
            return max_l * LH + 3

        def draw_row_impl(row, fill_c, txt_c, fsize, bold):
            x0 = self.l_margin
            y0 = self.get_y()
            row_h = calc_row_h(row, fsize)
            if y0 + row_h > self.h - self.b_margin - 5:
                self.add_page()
                y0 = self.get_y()
            x = x0
            for cell, cw in zip(row, col_widths):
                self.set_fill_color(*fill_c)
                self.set_draw_color(180, 180, 180)
                self.rect(x, y0, cw, row_h, style='FD')
                self.set_font("Arial", "B" if bold else "", fsize)
                self.set_text_color(*txt_c)
                self.set_xy(x + 1, y0 + 1.5)
                self.multi_cell(cw - 2, LH, str(cell), border=0, fill=False,
                                align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
                x += cw
            self.set_xy(x0, y0 + row_h)
            self.set_text_color(0, 0, 0)
            self.set_draw_color(0, 0, 0)

        # Header
        draw_row_impl(headers, header_color, (255, 255, 255), 9, True)
        # Data rows
        for ri, row in enumerate(rows):
            fill_c = (240, 244, 255) if ri % 2 == 0 else (255, 255, 255)
            draw_row_impl(row, fill_c, (0, 0, 0), 8.5, False)
        self.ln(3)

    def insert_figure(self, path, caption="", w=170):
        if os.path.exists(path):
            avail_h = self.h - self.get_y() - self.b_margin - 15
            img_w = w
            # Estimate height from aspect ratio
            from PIL import Image as PILImage
            with PILImage.open(path) as im:
                ar = im.height / im.width
            img_h = img_w * ar
            if img_h > avail_h and avail_h > 40:
                img_h = avail_h
                img_w = img_h / ar
            if img_h > (self.h - 2 * self.b_margin - 30):
                self.add_page()
                img_w = w
                img_h = img_w * ar
            x = (self.w - img_w) / 2
            self.image(path, x=x, w=img_w, h=img_h)
            if caption:
                self.set_font("Arial", "I", 8.5)
                self.set_text_color(80, 80, 80)
                self.cell(0, 6, caption, align="C",
                          new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                self.set_text_color(0, 0, 0)
            self.ln(4)

    def highlight_box(self, text, bg=(232, 240, 254), border=(13, 71, 161)):
        self.set_fill_color(*bg)
        self.set_draw_color(*border)
        self.set_font("Arial", "", 9.5)
        self.set_line_width(0.5)
        self.multi_cell(0, 6.5, text, border=1, fill=True,
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_line_width(0.2)
        self.ln(3)

    def result_box(self, label, coeff, sig, interp, color=(22, 101, 52)):
        self.set_fill_color(240, 253, 244)
        self.set_draw_color(*color)
        self.set_line_width(0.5)
        x0 = self.l_margin
        w_total = self.w - self.l_margin - self.r_margin
        self.set_font("Arial", "B", 10)
        self.cell(w_total * 0.30, 7, label, border="LTB", fill=True)
        self.set_font("Arial", "B", 10)
        self.set_text_color(*color)
        self.cell(w_total * 0.18, 7, coeff + " " + sig, border="TB", fill=True, align="C")
        self.set_text_color(0, 0, 0)
        self.set_font("Arial", "", 9)
        self.cell(w_total * 0.52, 7, interp, border="TRB", fill=True)
        self.ln()
        self.set_line_width(0.2)


# ═══════════════════════════════════════════════════════════════════════════
# Build PDF
# ═══════════════════════════════════════════════════════════════════════════

pdf = ReportPDF()
pdf.set_title("Báo Cáo Kinh Tế Lượng — Ngành Gỗ Nội Thất Việt Nam")
pdf.set_author("Phân tích dữ liệu GSO Doanh nghiệp")

# ════════════════════════════════════════════════════════════════════════════
# TRANG BÌA
# ════════════════════════════════════════════════════════════════════════════
pdf.add_page()

pdf.ln(20)
pdf.set_font("Arial", "B", 20)
pdf.set_text_color(13, 71, 161)
pdf.multi_cell(0, 12, "CÁC NHÂN TỐ ẢNH HƯỞNG ĐẾN SẢN LƯỢNG\nDOANH NGHIỆP NGÀNH GỖ NỘI THẤT VIỆT NAM", align="C")
pdf.set_text_color(0, 0, 0)
pdf.ln(8)

pdf.set_font("Arial", "I", 13)
pdf.set_text_color(80, 80, 80)
pdf.multi_cell(0, 8, "Phân tích Kinh Tế Lượng — Dữ liệu bảng (2012–2018)", align="C")
pdf.set_text_color(0, 0, 0)
pdf.ln(10)

# Divider
pdf.set_draw_color(13, 71, 161)
pdf.set_line_width(1.5)
pdf.line(pdf.l_margin + 30, pdf.get_y(), pdf.w - pdf.r_margin - 30, pdf.get_y())
pdf.set_line_width(0.2)
pdf.ln(15)

# Model display
pdf.set_font("Arial", "B", 11)
pdf.set_fill_color(232, 240, 254)
pdf.set_draw_color(13, 71, 161)
pdf.set_line_width(0.5)
pdf.multi_cell(0, 8,
    "Mô hình:\n"
    "ln(Output_it) = beta_0 + beta_1·ln(Labor_it) + beta_2·ln(Capital_it) + beta_3·Leverage_it\n"
    "                   + beta_4·ln(Wage_it) + beta_5·ln(Size_it) + ε_it",
    border=1, fill=True, align="C")
pdf.set_line_width(0.2)
pdf.ln(12)

# Info grid
pdf.set_font("Arial", "", 10)
info_items = [
    ("Nguồn dữ liệu",   "Điều tra Doanh nghiệp GSO, Việt Nam"),
    ("Giai doan (7 nam)",        "2012 - 2018"),
    ("Ngành",            "VSIC 31 — Sản xuất giường, tủ, bàn, ghế"),
    ("Số quan sát",      f"{len(df):,} quan sát — {df['firm_id'].nunique():,} doanh nghiệp"),
    ("Phương pháp",      "Pooled OLS · Fixed Effects · Random Effects · Hausman Test"),
]
col_w = [55, 115]
for label, val in info_items:
    pdf.set_fill_color(13, 71, 161)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 9.5)
    pdf.cell(col_w[0], 8, "  " + label, border=1, fill=True)
    pdf.set_fill_color(245, 248, 255)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "", 9.5)
    pdf.cell(col_w[1], 8, "  " + val, border=1, fill=True)
    pdf.ln()
pdf.ln(10)

pdf.set_font("Arial", "I", 9)
pdf.set_text_color(120, 120, 120)
pdf.cell(0, 6, "Phân tích được thực hiện bằng Python (pandas, statsmodels, linearmodels)", align="C",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_text_color(0, 0, 0)

# ════════════════════════════════════════════════════════════════════════════
# MỤC LỤC
# ════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1("MỤC LỤC")
toc = [
    ("I.",     "Gioi thieu va mo hinh nghien cuu"),
    ("II.",    "Mo ta du lieu"),
    ("III.",   "Phan tich kham pha (EDA)"),
    ("III.5.", "Thi phan, CR4 va HHI — Cau truc thi truong"),
    ("IV.",    "Ket qua uoc luong mo hinh (OLS / FE / RE)"),
    ("V.",     "Lua chon mo hinh — Kiem dinh Hausman"),
    ("VI.",    "Kiem dinh chan doan (VIF, BP, DW)"),
    ("VII.",   "Kiem tra tinh ben vung (Robustness Check)"),
    ("VIII.",  "Thao luan ket qua"),
    ("IX.",    "Ket luan va ham y chinh sach"),
]
pdf.ln(3)
for num, title in toc:
    pdf.set_font("Arial", "B", 11)
    pdf.cell(15, 8, num)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(pdf.l_margin, pdf.get_y() - 1, pdf.w - pdf.r_margin, pdf.get_y() - 1)
pdf.ln(8)

# ════════════════════════════════════════════════════════════════════════════
# I. GIỚI THIỆU VÀ MÔ HÌNH
# ════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1("I. GIỚI THIỆU VÀ MÔ HÌNH NGHIÊN CỨU")

pdf.h2("1.1. Giới thiệu")
pdf.body(
    "Ngành sản xuất gỗ và nội thất (VSIC 31) là một trong những ngành chế biến "
    "chế tạo quan trọng tại Việt Nam, đóng góp đáng kể vào kim ngạch xuất khẩu "
    "và tạo việc làm. Nghiên cứu này phân tích các nhân tố ảnh hưởng đến sản "
    "lượng (doanh thu thuần) của các doanh nghiệp trong ngành, sử dụng bộ dữ "
    "liệu bảng (panel data) từ Điều tra Doanh nghiệp của Tổng cục Thống kê "
    "(GSO) giai đoạn 2012–2018."
)
pdf.body(
    "Bộ dữ liệu bao gồm 2.294 quan sát từ 590 doanh nghiệp độc nhất, tạo thành "
    "bảng không cân bằng (unbalanced panel) với trung bình 3,9 năm quan sát mỗi doanh nghiệp. "
    "Cấu trúc dữ liệu bảng cho phép kiểm soát các đặc điểm không quan sát được "
    "cố định theo doanh nghiệp, giảm thiểu độ chệch ước lượng."
)

pdf.h2("1.2. Mô hình kinh tế lượng")
pdf.highlight_box(
    "ln(Output_it) = beta_0 + beta_1·ln(Labor_it) + beta_2·ln(Capital_it) + beta_3·Leverage_it\n"
    "                   + beta_4·ln(Wage_it) + beta_5·ln(Size_it) + ε_it\n\n"
    "Trong đó:  i = doanh nghiệp,  t = năm,  ε = sai số ngẫu nhiên"
)

# Variable table
pdf.h3("Định nghĩa biến số")
var_headers = ["Biến (ký hiệu)", "Loại", "Định nghĩa", "Đơn vị"]
var_rows = [
    ["ln(Output_it)",  "Phụ thuộc (Y)",  "Log doanh thu thuần",                   "ln(triệu VND)"],
    ["ln(Labor_it)",   "beta_1",          "Log số lao động cuối năm",              "ln(người)"],
    ["ln(Capital_it)", "beta_2",          "Log tài sản cố định cuối năm",          "ln(triệu VND)"],
    ["Leverage_it",    "beta_3",          "Tỷ lệ nợ / tổng tài sản",              "tỷ lệ >= 0"],
    ["ln(Wage_it)",    "beta_4",          "Log lương bình quân / lao động",        "ln(triệu VND)"],
    ["ln(Size_it)",    "beta_5",          "Log tổng tài sản (quy mô DN)",          "ln(triệu VND)"],
]
pdf.table(var_headers, var_rows, col_widths=[38, 28, 82, 22])

pdf.h2("1.3. Chiến lược ước lượng")
pdf.body(
    "Ba ước lượng viên được áp dụng và so sánh:\n"
    "  (1) Pooled OLS — Mô hình cơ sở, bỏ qua dị biệt không quan sát được.\n"
    "  (2) Fixed Effects (FE) — Kiểm soát các đặc điểm cố định của doanh nghiệp "
    "      bằng cách khử sai số nhóm qua demeaning (ước lượng within).\n"
    "  (3) Random Effects (RE) — Giả định hiệu ứng nhóm là ngẫu nhiên và không "
    "      tương quan với biến giải thích.\n"
    "Kiểm định Hausman được sử dụng để lựa chọn giữa FE và RE. "
    "Tất cả các mô hình sử dụng sai số chuẩn robust để xử lý phương sai sai số thay đổi."
)

# ════════════════════════════════════════════════════════════════════════════
# II. MÔ TẢ DỮ LIỆU
# ════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1("II. MÔ TẢ DỮ LIỆU")

pdf.h2("2.1. Cấu trúc bảng dữ liệu")

# Year table
yr = yr_tbl.copy()
yr['year'] = yr['year'].astype(int)
pdf.table(
    ["Năm", "Quan sát", "DN độc nhất", "TB Doanh thu\n(triệu VND)", "TB Lao\nđộng", "TB Đòn\nbẩy"],
    [[str(int(r.year)), str(int(r.N_obs)), str(int(r.N_firms)),
      f"{r.Mean_Output:,.0f}", f"{r.Mean_Labor:.0f}", f"{r.Mean_Leverage:.3f}"]
     for _, r in yr.iterrows()],
    col_widths=[14, 22, 25, 45, 22, 22]
)

pdf.body(
    "Dữ liệu bao gồm 2.294 quan sát từ 590 doanh nghiệp thuộc ngành VSIC 31 "
    "trong giai đoạn 2012–2018. Đây là bảng không cân bằng với trung bình 3,9 năm "
    "quan sát mỗi doanh nghiệp. Xu hướng cho thấy doanh thu bình quân tăng dần "
    "từ 87.704 triệu đồng (2012) lên 263.792 triệu đồng (2018), phản ánh sự "
    "tăng trưởng đáng kể của ngành."
)

pdf.h2("2.2. Thống kê mô tả")
pdf.h3("Bảng 1 — Thống kê mô tả các biến gốc (trước khi lấy log)")

raw_headers = ["Biến", "N", "Trung bình", "ĐLC", "Nhỏ nhất", "Trung vị", "Lớn nhất"]
raw_idx_names = {
    "Output (net sales, mil VND)":   "Sản lượng (tr. VND)",
    "Labor (employees)":             "Lao động (người)",
    "Capital (fixed assets, mil VND)":"Vốn cố định (tr. VND)",
    "Leverage (liab/assets)":        "Đòn bẩy",
    "Wage (mil VND/employee)":       "Lương BQ (tr. VND/người)",
    "Size (total assets, mil VND)":  "Quy mô (tr. VND)",
}
raw_rows_table = []
for idx in desc_raw.index:
    r = desc_raw.loc[idx]
    label = raw_idx_names.get(idx, idx)
    raw_rows_table.append([
        label,
        f"{r['N']:,.0f}",
        f"{r['Mean']:,.2f}",
        f"{r['Std Dev']:,.2f}",
        f"{r['Min']:,.2f}",
        f"{r['Median']:,.2f}",
        f"{r['Max']:,.2f}",
    ])
pdf.table(raw_headers, raw_rows_table, col_widths=[52, 15, 28, 28, 22, 22, 23])

pdf.h3("Bảng 2 — Thống kê mô tả các biến log-transform")
log_headers = ["Biến", "N", "Trung bình", "ĐLC", "Nhỏ nhất", "Trung vị", "Lớn nhất"]
log_idx_names = {
    "ln(Output)":             "ln(Sản lượng)",
    "ln(Labor)":              "ln(Lao động)",
    "ln(Capital)":            "ln(Vốn cố định)",
    "Leverage (liab/assets)": "Đòn bẩy",
    "ln(Wage)":               "ln(Lương BQ)",
    "ln(Size)":               "ln(Quy mô)",
}
log_rows_table = []
for idx in desc_log.index:
    r = desc_log.loc[idx]
    label = log_idx_names.get(idx, idx)
    log_rows_table.append([
        label,
        f"{r['N']:,.0f}",
        f"{r['Mean']:.4f}",
        f"{r['Std Dev']:.4f}",
        f"{r['Min']:.4f}",
        f"{r['Median']:.4f}",
        f"{r['Max']:.4f}",
    ])
pdf.table(log_headers, log_rows_table, col_widths=[52, 15, 28, 28, 22, 22, 23])

pdf.body(
    "Nhận xét: Doanh thu bình quân là 168.039 triệu đồng với độ lệch chuẩn rất lớn "
    "(357.229 tr. VND), phản ánh sự phân tán cao giữa các doanh nghiệp. Tỷ lệ đòn bẩy "
    "trung bình là 0,68 cho thấy phần lớn tài sản được tài trợ bởi nợ. "
    "Sau khi lấy log, các biến có phân phối gần chuẩn hơn, phù hợp cho phân tích hồi quy."
)

# ════════════════════════════════════════════════════════════════════════════
# III. PHÂN TÍCH KHÁM PHÁ
# ════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1("III. PHÂN TÍCH KHÁM PHÁ (EDA)")

pdf.h2("3.1. Phân phối các biến mô hình")
pdf.insert_figure(
    os.path.join(FIG_DIR, "fig1_distributions.png"),
    "Hình 1: Phân phối các biến mô hình (sau log-transform), ngành gỗ nội thất 2012–2018",
    w=168
)
pdf.body(
    "Sau khi lấy logarithm tự nhiên và winsorize tại phân vị 1%–99%, "
    "các biến có phân phối gần đối xứng, phù hợp để ứng dụng trong mô hình hồi quy tuyến tính. "
    "ln(Lao động) có đỉnh nhọn (leptokurtic), phản ánh sự hiện diện đông đảo "
    "của các doanh nghiệp quy mô vừa và nhỏ."
)

pdf.h2("3.2. Ma trận tương quan")
pdf.insert_figure(
    os.path.join(FIG_DIR, "fig2_correlation_heatmap.png"),
    "Hình 2: Ma trận tương quan Pearson giữa các biến (log-transform)",
    w=110
)

# Correlation table
corr_headers = ["Biến"] + [c[:12] for c in corr_tbl.columns]
corr_rows = []
corr_label_map = {
    "ln(Output)": "ln(Sản lượng)",
    "ln(Labor)": "ln(Lao động)",
    "ln(Capital)": "ln(Vốn)",
    "Leverage (liab/assets)": "Đòn bẩy",
    "ln(Wage)": "ln(Lương)",
    "ln(Size)": "ln(Quy mô)",
}
for idx in corr_tbl.index:
    row = [corr_label_map.get(idx, idx[:14])]
    for val in corr_tbl.loc[idx]:
        row.append(f"{val:.3f}")
    corr_rows.append(row)
n_cols = len(corr_headers)
usable = pdf.w - pdf.l_margin - pdf.r_margin
col_ws = [38] + [(usable - 38) / (n_cols - 1)] * (n_cols - 1)
pdf.table(corr_headers, corr_rows, col_widths=col_ws)

pdf.body(
    "Nhận xét tương quan:\n"
    "  - ln(Lao động) có tương quan rất cao với ln(Sản lượng) (r = 0,891), khẳng định vai trò "
    "    then chốt của lao động trong ngành thâm dụng lao động này.\n"
    "  - ln(Vốn) và ln(Quy mô) có tương quan cao với nhau (r = 0,884) do vốn cố định "
    "    là thành phần của tổng tài sản, dẫn đến hiện tượng đa cộng tuyến.\n"
    "  - Đòn bẩy có tương quan thấp với các biến còn lại, không gây lo ngại về đa cộng tuyến."
)

pdf.add_page()
pdf.h2("3.3. Xu hướng theo thời gian")
pdf.insert_figure(
    os.path.join(FIG_DIR, "fig3_trends_over_time.png"),
    "Hình 3: Xu hướng của sản lượng, lao động và đòn bẩy theo năm (2012–2018)",
    w=168
)
pdf.body(
    "Xu hướng nổi bật trong giai đoạn 2012–2018:\n"
    "  - Sản lượng bình quân tăng gần 3 lần từ 87.704 triệu VND (2012) lên 263.792 triệu VND (2018), "
    "    thể hiện sự tăng trưởng mạnh mẽ của ngành gỗ nội thất Việt Nam.\n"
    "  - Số lao động bình quân tăng từ 251 người (2012) lên 489 người (2018), "
    "    phản ánh xu hướng mở rộng quy mô doanh nghiệp.\n"
    "  - Đòn bẩy tài chính khá ổn định trong giai đoạn 2012–2017 (0,63–0,72), "
    "    nhưng tăng lên 0,82 vào năm 2018."
)

pdf.h2("3.4. Mối quan hệ ln(Sản lượng) với các biến giải thích")
pdf.insert_figure(
    os.path.join(FIG_DIR, "fig4_scatter_plots.png"),
    "Hình 4: Biểu đồ phân tán ln(Sản lượng) theo từng biến giải thích",
    w=168
)
pdf.body(
    "Quan hệ tuyến tính dương rõ ràng giữa ln(Sản lượng) và ln(Lao động), "
    "ln(Quy mô), ln(Vốn), ln(Lương). Đòn bẩy có tương quan dương yếu. "
    "Các đường xu hướng đều có độ dốc dương, xác nhận hướng tác động kỳ vọng."
)

# ════════════════════════════════════════════════════════════════════════════
# III.5  THI PHAN, CR4, HHI
# ════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1("III.5. THI PHAN, CR4 VA HHI — CAU TRUC THI TRUONG")

pdf.h2("3.5.1. Giai thich chi so")
pdf.body(
    "Truoc khi uoc luong mo hinh, can phan tich cau truc thi truong cua nganh go "
    "noi that de hieu boi canh canh tranh. Ba chi so duoc su dung:\n\n"
    "  - THI PHAN (Market Share): Doanh thu DN / Tong doanh thu nganh trong nam.\n"
    "    Phan anh vi the canh tranh cua tung doanh nghiep.\n\n"
    "  - CR4 (Concentration Ratio 4): Tong thi phan cua 4 DN lon nhat trong nam.\n"
    "    CR4 < 40% = thi truong canh tranh;  40-60% = oligopoly;  >60% = doc quyen nhom.\n\n"
    "  - HHI (Herfindahl-Hirschman Index): Tong binh phuong thi phan x 10.000.\n"
    "    HHI < 1.500 = thi truong canh tranh;  1.500-2.500 = tap trung vua;\n"
    "    HHI > 2.500 = tap trung cao (theo chuan Hoa Ky / EU)."
)

pdf.h2("3.5.2. Ket qua tinh toan CR4 va HHI (2012-2018)")

conc_headers = ["Nam", "So DN", "CR4 (%)", "HHI", "Danh gia CR4", "Danh gia HHI"]
conc_rows_tbl = []
for _, r in conc_tbl.iterrows():
    cr4_pct = r['CR4']
    hhi_val = r['HHI']
    cr4_label = "Canh tranh" if cr4_pct < 40 else ("Oligopoly" if cr4_pct < 60 else "Doc quyen nhom")
    hhi_label  = "Canh tranh" if hhi_val < 1500 else ("Tap trung vua" if hhi_val < 2500 else "Tap trung cao")
    conc_rows_tbl.append([
        str(int(r['year'])),
        str(int(r['N_firms'])),
        f"{cr4_pct:.2f}%",
        f"{hhi_val:.1f}",
        cr4_label,
        hhi_label,
    ])
pdf.table(conc_headers, conc_rows_tbl, col_widths=[14, 18, 22, 22, 37, 37])

pdf.highlight_box(
    "Ket luan cau truc thi truong: CR4 dao dong 14,6% - 22,0% va HHI tu 123 - 208,\n"
    "deu rat thap so voi nguong canh tranh. Nganh go noi that VSIC 31 co cau truc\n"
    "THI TRUONG CANH TRANH CAO — khong co DN nao co vi the doc quyen.\n"
    "Tuy nhien, xu huong tap trung tang nhe tu 2012 den 2018 (CR4 tang 6,1 diem %,\n"
    "HHI tang 74 diem), cho thay qua trinh tich tu san xuat dang dien ra.",
    bg=(232, 240, 254), border=(13, 71, 161)
)

pdf.h2("3.5.3. Bieu do cau truc thi truong")
pdf.insert_figure(
    os.path.join(FIG_DIR, "fig_market_concentration.png"),
    "Hinh: CR4, HHI va so DN nganh go noi that Viet Nam 2012-2018",
    w=168
)
pdf.body(
    "Nhan xet tu bieu do:\n"
    "  - CR4 tang lien tuc tu 15,9% (2012) len 22,0% (2018), du van o muc canh tranh.\n"
    "  - HHI tang tu 134 (2012) len 208 (2018), phan anh xu huong tap trung nhe.\n"
    "  - So DN trong mau giam tu 523 (2012) xuong 230 (2018), co the phan anh\n"
    "    qua trinh rut lui cua DN nho va gia nhap cua DN lon hon."
)

pdf.h2("3.5.4. Thi phan top 10 doanh nghiep lon nhat")
pdf.insert_figure(
    os.path.join(FIG_DIR, "fig_top10_market_share.png"),
    "Hinh: Phan phoi thi phan Top 10 DN lon nhat moi nam (2012-2018)",
    w=155
)
pdf.body(
    "Thi phan cua DN hang dau dao dong 3-6%, cho thay khong co DN nao chiem vi tri\n"
    "ap dao. Duong cong thi phan co do doc kha thoai — phu hop voi thi truong\n"
    "co nhieu DN nho va vua canh tranh. Den 2018, DN lon nhat chiem 6,2% thi phan,\n"
    "tang nhe so voi 5,0% nam 2012."
)

# ════════════════════════════════════════════════════════════════════════════
# IV. KẾT QUẢ HỒI QUY
# ════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1("IV. KẾT QUẢ HỒI QUY")

pdf.h2("4.1. Bảng kết quả tổng hợp")
pdf.body(
    "Bảng dưới đây trình bày hệ số ước lượng và sai số chuẩn (trong ngoặc) "
    "từ 4 mô hình. Tất cả mô hình dùng sai số chuẩn robust. "
    "Ký hiệu mức ý nghĩa: *** p<0,01  ** p<0,05  * p<0,10."
)

# Compact regression table
reg_headers = ["Biến", "Pooled OLS", "FE 1-chiều", "FE 2-chiều (*)", "RE"]
reg_var_map = {
    "Constant":   "Hằng số",
    "ln(Labor)":  "ln(Lao động)",
    "ln(Capital)":"ln(Vốn cố định)",
    "Leverage":   "Đòn bẩy",
    "ln(Wage)":   "ln(Lương BQ)",
    "ln(Size)":   "ln(Quy mô)",
}

# Build rows from reg_tbl
coef_rows = []
for var in ["Constant", "ln(Labor)", "ln(Capital)", "Leverage", "ln(Wage)", "ln(Size)"]:
    row_data = reg_tbl[reg_tbl["Variable"] == var]
    if len(row_data) == 0:
        continue
    r = row_data.iloc[0]
    def fmt_cell(col_c, col_se):
        c = str(r.get(col_c, "")).strip()
        s = str(r.get(col_se, "")).strip()
        if c == "nan" or c == "NaN" or c == "":
            return "—"
        return f"{c}\n{s}"
    coef_rows.append([
        reg_var_map.get(var, var),
        fmt_cell("Coef (OLS)",  "SE (OLS)"),
        fmt_cell("Coef (FE1W)", "SE (FE1W)"),
        fmt_cell("Coef (FE2W)", "SE (FE2W)"),
        fmt_cell("Coef (RE)",   "SE (RE)"),
    ])

# Stat rows
r2_ols  = reg_tbl[reg_tbl["Variable"]=="R² (within)"].iloc[0]["Coef (OLS)"]
r2_fe1  = reg_tbl[reg_tbl["Variable"]=="R² (within)"].iloc[1]["Coef (FE1W)"]
r2_fe2  = reg_tbl[reg_tbl["Variable"]=="R² (within)"].iloc[2]["Coef (FE2W)"]
r2_re   = reg_tbl[reg_tbl["Variable"]=="R² (within)"].iloc[3]["Coef (RE)"]
nobs    = reg_tbl[reg_tbl["Variable"]=="Observations"].iloc[0]
h_pval  = reg_tbl[reg_tbl["Variable"]=="Hausman p-value"].iloc[0]

coef_rows.append(["R² (within)",
    str(r2_ols), str(r2_fe1), str(r2_fe2), str(r2_re)])
coef_rows.append(["Số quan sát",
    str(nobs.get("Coef (OLS)","")), str(nobs.get("Coef (FE1W)","")),
    str(nobs.get("Coef (FE2W)","")), str(nobs.get("Coef (RE)",""))])
coef_rows.append(["Hausman p-value",
    "—", str(h_pval.get("Coef (FE1W)","")), "—", "—"])

usable = pdf.w - pdf.l_margin - pdf.r_margin
pdf.table(reg_headers, coef_rows, col_widths=[38, 37, 37, 37, 21])

pdf.set_font("Arial", "I", 8)
pdf.body("(*) Mô hình ưu tiên theo kiểm định Hausman (p < 0,001). "
         "Sai số chuẩn robust trong ngoặc. *** p<0,01, ** p<0,05, * p<0,10.")

pdf.h2("4.2. Biểu đồ so sánh hệ số các mô hình")
pdf.insert_figure(
    os.path.join(FIG_DIR, "fig6_coefficient_plot.png"),
    "Hình 6: Hệ số ước lượng và khoảng tin cậy 95% từ 4 mô hình",
    w=155
)
pdf.body(
    "Biểu đồ cho thấy các hệ số tương đối nhất quán qua các mô hình về chiều tác động. "
    "Pooled OLS có xu hướng phóng đại hệ số lao động và quy mô do bỏ sót đặc điểm "
    "doanh nghiệp không quan sát được. Mô hình FE (2-chiều) cho hệ số thận trọng hơn "
    "sau khi kiểm soát hiệu ứng doanh nghiệp và thời gian."
)

# ════════════════════════════════════════════════════════════════════════════
# V. LỰA CHỌN MÔ HÌNH — KIỂM ĐỊNH HAUSMAN
# ════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1("V. LỰA CHỌN MÔ HÌNH — KIỂM ĐỊNH HAUSMAN")

pdf.h2("5.1. Lý thuyết")
pdf.body(
    "Kiểm định Hausman (1978) kiểm tra giả thuyết H0: hiệu ứng nhóm không tương quan "
    "với các biến giải thích (ủng hộ RE) so với H1: có tương quan (ủng hộ FE). "
    "Thống kê kiểm định tuân theo phân phối chi2 với bậc tự do bằng số biến giải thích."
)

pdf.h2("5.2. Kết quả kiểm định Hausman")
h = hausman_tbl.iloc[0]
hausman_rows = [
    ["Thống kê H (chi2)", f"{float(h['Statistic']):.4f}"],
    ["Bậc tự do (df)", str(int(h['DF']))],
    ["P-value", f"{float(h['p-value']):.4f}"],
    ["Kết luận", "Bác bỏ H0 — Sử dụng Fixed Effects"],
]
pdf.table(["Chỉ tiêu", "Giá trị"], hausman_rows, col_widths=[90, 80])

pdf.highlight_box(
    "Kết luận: Hausman H = 41,9956 ~ chi2(5), p-value < 0,0001\n"
    "=> Bác bỏ H0 ở mức ý nghĩa 1% — Hiệu ứng doanh nghiệp tương quan với "
    "biến giải thích. MÔ HÌNH FIXED EFFECTS LÀ PHÙ HỢP.",
    bg=(255, 243, 224), border=(230, 81, 0)
)

pdf.body(
    "Kết quả kiểm định Hausman cho thấy không thể giả định hiệu ứng riêng của "
    "doanh nghiệp là ngẫu nhiên và độc lập với các biến giải thích. "
    "Điều này có nghĩa là các đặc điểm cố định của doanh nghiệp "
    "(ví dụ: năng lực quản lý, thương hiệu, vị trí địa lý) tương quan với "
    "quy mô vốn, lao động và tiền lương. Mô hình FE (2-chiều) được chọn làm "
    "mô hình ưu tiên, kiểm soát cả hiệu ứng doanh nghiệp và hiệu ứng thời gian."
)

# ════════════════════════════════════════════════════════════════════════════
# VI. KIỂM ĐỊNH CHẨN ĐOÁN
# ════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1("VI. KIỂM ĐỊNH CHẨN ĐOÁN")

pdf.h2("6.1. Phương sai sai số thay đổi và tự tương quan")
diag_rows = [
    ["Breusch-Pagan (phương sai sai số thay đổi)",
     f"LM = {float(diag_tbl.iloc[0]['Statistic']):.4f}",
     f"p = {float(str(diag_tbl.iloc[0]['p-value']).replace('nan','').strip() or '0'):.4f}",
     "Phát hiện — dùng SE robust"],
    ["Durbin-Watson (tự tương quan dương)",
     f"DW = {float(diag_tbl.iloc[1]['Statistic']):.4f}", "—",
     "Có tự tương quan dương"],
]
pdf.table(
    ["Kiểm định", "Thống kê", "P-value", "Kết luận"],
    diag_rows,
    col_widths=[75, 28, 22, 45]
)
pdf.body(
    "Kiểm định Breusch-Pagan bác bỏ giả thuyết phương sai đồng nhất (p < 0,001), "
    "xác nhận cần dùng sai số chuẩn robust. Chỉ số Durbin-Watson = 1,21 < 2 "
    "cho thấy tự tương quan dương trong phần dư, điều này phổ biến với dữ liệu bảng "
    "và được xử lý bằng sai số chuẩn cluster-robust trong mô hình FE."
)

pdf.h2("6.2. Đa cộng tuyến (VIF)")
vif_rows = [[r['Variable'], f"{r['VIF']:.3f}",
             "Cao — do Capital thuộc Size" if r['VIF'] > 100 else
             ("Chấp nhận được" if r['VIF'] < 10 else "Trung bình")]
            for _, r in vif_tbl.iterrows()]
pdf.table(["Biến", "VIF", "Đánh giá"], vif_rows, col_widths=[60, 30, 80])
pdf.body(
    "VIF rất cao của ln(Vốn) (164) và ln(Quy mô) (172) do vốn cố định là thành "
    "phần của tổng tài sản — đây là đặc điểm cấu trúc, không phải lỗi mô hình. "
    "Trong ước lượng FE (within estimator), đa cộng tuyến không làm chệch hệ số "
    "nhưng có thể làm tăng sai số chuẩn. Kết quả ổn định qua các mô hình khác nhau "
    "cho thấy tác động này không đáng kể trong trường hợp này."
)

pdf.h2("6.3. Phần dư mô hình FE (2-chiều)")
pdf.insert_figure(
    os.path.join(FIG_DIR, "fig7_residual_diagnostics.png"),
    "Hình 7: Phần dư vs Giá trị dự báo | Q-Q Plot | Phân phối phần dư — Mô hình FE 2-chiều",
    w=168
)
pdf.body(
    "(a) Phần dư phân tán đều quanh 0, không có xu hướng hệ thống rõ ràng.\n"
    "(b) Q-Q plot cho thấy phân phối phần dư gần chuẩn ở vùng trung tâm, "
    "    nhưng có đuôi nặng ở hai đầu (kurtosis cao) — đặc trưng của dữ liệu doanh nghiệp.\n"
    "(c) Histogram phần dư có dạng gần chuẩn, xác nhận mô hình phù hợp."
)

# ════════════════════════════════════════════════════════════════════════════
# VII. KIỂM TRA TÍNH BỀN VỮNG
# ════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1("VII. KIỂM TRA TÍNH BỀN VỮNG (ROBUSTNESS CHECK)")

pdf.h2("7.1. Thay biến phụ thuộc: ln(Giá trị gia tăng) thay vì ln(Doanh thu)")
pdf.body(
    "Để kiểm tra tính ổn định của kết quả, mô hình FE (2-chiều) được ước lượng lại "
    "với ln(Giá trị gia tăng — VA) làm biến phụ thuộc thay vì ln(Doanh thu thuần). "
    "Giá trị gia tăng = Lợi nhuận trước thuế + Chi phí lao động + Khấu hao, "
    "phản ánh đóng góp thực sự của doanh nghiệp vào nền kinh tế."
)

rob_rows = [
    ["ln(Lao động)",    "0,5296***",  "(0,0430)",  "0,4841***", "(0,0497)"],
    ["ln(Vốn cố định)", "0,0793**",   "(0,0346)",  "0,1117**",  "(0,0400)"],
    ["Đòn bẩy",         "-0,2054***", "(0,0735)",  "-0,0274",   "(0,0591)"],
    ["ln(Lương BQ)",    "0,7472***",  "(0,0458)",  "0,3564***", "(0,0490)"],
    ["ln(Quy mô)",      "0,1777***",  "(0,0436)",  "0,2572***", "(0,0511)"],
    ["R² (within)",     "0,5479",     "",          "0,3378",    ""],
    ["Số quan sát",     "2.185",      "",          "2.294",     ""],
]
pdf.table(
    ["Biến", "FE — ln(VA)", "SE", "FE — ln(DT thuần)", "SE"],
    rob_rows,
    col_widths=[45, 34, 20, 44, 27]
)

pdf.highlight_box(
    "Kết luận: Kết quả về chiều và mức ý nghĩa thống kê của ln(Lao động), "
    "ln(Vốn), ln(Lương) và ln(Quy mô) nhất quán qua cả hai biến phụ thuộc. "
    "Điểm khác biệt đáng chú ý: Đòn bẩy có tác động âm và có ý nghĩa (he so = -0,205***) "
    "khi dùng VA — gợi ý rằng nợ cao làm giảm giá trị gia tăng, dù không ảnh hưởng "
    "đến doanh thu thuần. Kết quả mô hình chính là bền vững.",
    bg=(240, 253, 244), border=(22, 101, 52)
)

pdf.h2("7.2. Hiệu ứng thời gian cố định")
pdf.insert_figure(
    os.path.join(FIG_DIR, "fig9_year_fixed_effects.png"),
    "Hình 9: Hiệu ứng thời gian cố định — tác động của năm lên ln(Sản lượng)",
    w=130
)
pdf.body(
    "Hiệu ứng thời gian (năm) dương và tăng dần qua các năm, phản ánh xu hướng "
    "tăng trưởng chung của ngành nội thất Việt Nam độc lập với các biến giải thích. "
    "Năm 2018 có hiệu ứng cao nhất, phù hợp với dữ liệu xu hướng (Hình 3)."
)

# ════════════════════════════════════════════════════════════════════════════
# VIII. THẢO LUẬN KẾT QUẢ
# ════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1("VIII. THẢO LUẬN KẾT QUẢ")

pdf.h2("Mô hình ưu tiên: Fixed Effects 2-chiều (FE 2-way)")
pdf.body(
    "Dựa trên kiểm định Hausman (H = 41,99, p < 0,001), mô hình Fixed Effects "
    "là phù hợp nhất. Mô hình này loại bỏ hiệu ứng cố định của từng doanh nghiệp "
    "(năng lực quản lý, công nghệ riêng, thương hiệu...) và từng năm (chu kỳ kinh "
    "tế, chính sách), cho phép ước lượng tác động nhân quả trong nội bộ doanh nghiệp."
)

pdf.h2("Kết quả chi tiết từng biến")

# Individual variable result boxes
pdf.result_box("ln(Lao động)", "beta_1 = 0,484", "***",
               "1% tăng lao động => +0,48% sản lượng",
               color=(13, 71, 161))
pdf.ln(1)
pdf.result_box("ln(Vốn cố định)", "beta_2 = 0,112", "**",
               "1% tăng tài sản cố định => +0,11% sản lượng",
               color=(46, 125, 50))
pdf.ln(1)
pdf.result_box("Đòn bẩy", "beta_3 = -0,027", "(n.s.)",
               "Tỷ lệ nợ không ảnh hưởng đáng kể đến sản lượng",
               color=(130, 130, 130))
pdf.ln(1)
pdf.result_box("ln(Lương BQ)", "beta_4 = 0,356", "***",
               "1% tăng lương => +0,36% sản lượng (hiệu ứng kỹ năng)",
               color=(183, 28, 28))
pdf.ln(1)
pdf.result_box("ln(Quy mô)", "beta_5 = 0,257", "***",
               "1% tăng quy mô => +0,26% sản lượng (lợi thế quy mô)",
               color=(123, 31, 162))
pdf.ln(6)

pdf.h3("Giải thích chi tiết:")
pdf.body(
    "(1) LƯƠNG — Yếu tố nổi bật nhất bên cạnh lao động:\n"
    "    Tác động dương của lương (beta_4 = 0,356***) phản ánh hiệu ứng vốn con người: "
    "    doanh nghiệp trả lương cao hơn thu hút lao động có kỹ năng tốt hơn, "
    "    từ đó nâng cao năng suất và sản lượng. Đây là tín hiệu quan trọng: "
    "    đầu tư vào chất lượng lao động mang lại lợi nhuận.\n\n"
    "(2) LƯƠNG > VỐN (beta_4 > beta_2):\n"
    "    Trong mô hình FE, hệ số lương lớn hơn hệ số vốn cố định, "
    "    khẳng định đây là ngành thâm dụng lao động kỹ năng hơn là thâm dụng vốn.\n\n"
    "(3) ĐÒN BẨY không có ý nghĩa thống kê:\n"
    "    Cấu trúc tài chính (tỷ lệ nợ) không tác động đáng kể đến sản lượng "
    "    trong ngắn hạn. Tuy nhiên, khi dùng VA làm biến phụ thuộc, đòn bẩy "
    "    có tác động âm (he so = -0,205***), gợi ý rằng nợ cao làm giảm hiệu quả "
    "    sản xuất dù không làm giảm doanh thu.\n\n"
    "(4) HIỆU SUẤT THEO QUY MÔ (Returns to Scale):\n"
    "    Tổng hệ số lao động và vốn (0,484 + 0,112 = 0,596) < 1,\n"
    "    cho thấy HIỆU SUẤT GIẢM DẦN THEO QUY MÔ trong chiều within. "
    "    Điều này hợp lý: khi doanh nghiệp mở rộng nhanh, hiệu quả biên giảm do "
    "    khó khăn trong quản lý và phối hợp."
)

# ════════════════════════════════════════════════════════════════════════════
# IX. KẾT LUẬN VÀ HÀM Ý CHÍNH SÁCH
# ════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1("IX. KẾT LUẬN VÀ HÀM Ý CHÍNH SÁCH")

pdf.h2("9.1. Tóm tắt phát hiện chính")
findings = [
    "Lao động là nhân tố quan trọng nhất: elasticity = 0,484 — ngành gỗ nội thất là ngành thâm dụng lao động.",
    "Chất lượng lao động (đại diện bởi lương bình quân) có tác động đáng kể: he so = 0,356 — kỹ năng quan trọng hơn số lượng.",
    "Vốn cố định có tác động khiêm tốn (he so = 0,112), nhỏ hơn nhiều so với lao động.",
    "Đòn bẩy tài chính không ảnh hưởng đáng kể đến doanh thu, nhưng làm giảm giá trị gia tăng.",
    "Quy mô doanh nghiệp có lợi thế (he so = 0,257), nhưng hiệu suất theo quy mô là giảm dần (0,596 < 1).",
    "Kiểm định Hausman xác nhận Fixed Effects là mô hình phù hợp — đặc điểm doanh nghiệp không quan sát được là quan trọng.",
]
for i, f in enumerate(findings, 1):
    pdf.set_font("Arial", "B", 9.5)
    pdf.set_x(pdf.l_margin + 3)
    pdf.cell(8, 6.5, f"({i})")
    pdf.set_font("Arial", "", 9.5)
    pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 11, 6.5, f,
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)

pdf.h2("9.2. Hàm ý chính sách")
policies = [
    ("Đầu tư vào đào tạo lao động",
     "Kết quả cho thấy nâng cao kỹ năng lao động (phản ánh qua lương) có tác động "
     "mạnh đến sản lượng. Chính phủ và doanh nghiệp cần ưu tiên đào tạo nghề "
     "và nâng cao tay nghề cho công nhân ngành gỗ."),
    ("Kiểm soát đòn bẩy tài chính",
     "Dù không ảnh hưởng đến doanh thu, tỷ lệ nợ cao làm giảm giá trị gia tăng. "
     "Doanh nghiệp nên duy trì cấu trúc vốn lành mạnh, hạn chế vay nợ quá mức."),
    ("Tận dụng lợi thế quy mô có kiểm soát",
     "Lợi thế quy mô tồn tại (beta_5 = 0,257) nhưng hiệu suất giảm dần theo quy mô. "
     "Chiến lược mở rộng cần đi kèm với nâng cao năng lực quản lý để duy trì hiệu quả."),
    ("Ưu tiên đầu tư máy móc thiết bị",
     "Hệ số vốn cố định dương (beta_2 = 0,112) cho thấy đầu tư vào máy móc, công nghệ "
     "có lợi cho sản lượng. Tuy nhiên lợi nhuận biên thấp hơn so với đầu tư nhân lực."),
]
for title, detail in policies:
    pdf.set_font("Arial", "B", 10)
    pdf.set_fill_color(232, 240, 254)
    pdf.set_draw_color(13, 71, 161)
    pdf.set_line_width(0.4)
    pdf.cell(0, 7, "  " + title, border="L", fill=True,
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_line_width(0.2)
    pdf.set_font("Arial", "", 9.5)
    pdf.set_x(pdf.l_margin + 5)
    pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 5, 6, detail,
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

pdf.h2("9.3. Hạn chế và hướng nghiên cứu tiếp theo")
pdf.body(
    "Nghiên cứu này có một số hạn chế:\n"
    "  - Dữ liệu bảng không cân bằng: nhiều doanh nghiệp chỉ xuất hiện 1–2 năm, "
    "    hạn chế khả năng khai thác biến thiên theo thời gian.\n"
    "  - Đa cộng tuyến giữa ln(Vốn) và ln(Quy mô) làm tăng sai số chuẩn, "
    "    dù không ảnh hưởng đến tính nhất quán của ước lượng FE.\n"
    "  - Chưa kiểm soát được các cú sốc ngoại sinh (thay đổi giá nguyên liệu, "
    "    chính sách thuế xuất khẩu gỗ).\n\n"
    "Hướng nghiên cứu tiếp theo:\n"
    "  - Phân tích theo phân khúc doanh nghiệp (nhỏ/vừa/lớn, FDI/tư nhân)\n"
    "  - Kiểm tra tính phi tuyến (mô hình ngưỡng — threshold regression)\n"
    "  - Mở rộng sang dữ liệu 2019–2023 để nắm bắt tác động COVID-19"
)

# ════════════════════════════════════════════════════════════════════════════
# PHỤ LỤC — Thực tế vs Dự báo
# ════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1("PHỤ LỤC — Biểu đồ bổ sung")

pdf.h2("A1. Sản lượng thực tế vs Dự báo")
pdf.insert_figure(
    os.path.join(FIG_DIR, "fig8_actual_vs_predicted.png"),
    "Hình 8: ln(Sản lượng) thực tế vs dự báo từ mô hình FE (2-chiều), R² within = 0,338",
    w=120
)

pdf.h2("A2. Phân bổ quan sát theo số năm quan sát")
pdf.insert_figure(
    os.path.join(FIG_DIR, "fig10_panel_balance.png"),
    "Hình 10: Số doanh nghiệp theo số năm có mặt trong mẫu",
    w=130
)

pdf.h2("A3. Phân bố theo hình thức sở hữu")
pdf.insert_figure(
    os.path.join(FIG_DIR, "fig5_ownership_boxplots.png"),
    "Hình 5: So sánh sản lượng, lao động và đòn bẩy theo hình thức sở hữu",
    w=155
)

# ── Save PDF ─────────────────────────────────────────────────────────────────
pdf.output(PDF_OUT)
print(f"\nDone. PDF da luu tai: {PDF_OUT}")
print(f"  Kich thuoc: {os.path.getsize(PDF_OUT) / 1024 / 1024:.2f} MB")
