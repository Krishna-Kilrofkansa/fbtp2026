"""
generate_pdf.py — Generate executive_summary.pdf

A tight 2-3 page judge-skimmable PDF with:
- Problem reframing
- Three methods + key results
- Stated assumptions
- Actionable recommendations
"""

from fpdf import FPDF
from pathlib import Path


class SummaryPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Flipkart Grid 7.0 - PS2: Traffic Event Intelligence", align="R")
        self.ln(10)
    
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")
    
    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(50, 50, 80)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(99, 102, 241)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)
    
    def subsection(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(70, 70, 100)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
    
    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(2)
    
    def bullet(self, text, bold_prefix=""):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        x = self.get_x()
        self.cell(5)
        self.cell(4, 5.5, "-")  # bullet
        if bold_prefix:
            self.set_font("Helvetica", "B", 10)
            self.write(5.5, bold_prefix + " ")
            self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, text)
        self.ln(1)
    
    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [(self.w - self.l_margin - self.r_margin) / len(headers)] * len(headers)
        
        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(240, 240, 250)
        self.set_text_color(50, 50, 80)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        
        # Rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 40, 40)
        for row in rows:
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), border=1, align="C")
            self.ln()
        self.ln(3)


def generate_pdf():
    pdf = SummaryPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    
    # ── Page 1 ──
    pdf.add_page()
    
    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(50, 50, 80)
    pdf.cell(0, 12, "BTP Traffic Event Intelligence", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(100, 100, 120)
    pdf.cell(0, 8, "Predictive Analytics for Bengaluru Traffic Police", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 6, "Flipkart Grid 7.0 - Problem Statement 2", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)
    
    # Problem Reframing
    pdf.section_title("1. Problem Reframing")
    pdf.body_text(
        "The dataset is labeled 'Rallies, Festivals & Traffic Management' but contains "
        "8,173 rows of incident logs: 94% are unplanned disruptions (vehicle breakdowns, "
        "potholes, accidents), not rallies or festivals. Only 191 rows are true public events. "
        "We reframe honestly: every logged disruption is an 'event' consuming traffic-management "
        "resources, and we predict its impact and resourcing need."
    )
    
    pdf.body_text(
        "Dataset: 7,944 clean events (Nov 2023 - Apr 2024), 23 corridors, "
        "46 original columns expanded to 51 engineered features. Key challenge: "
        "937 events (11.8%) are right-censored (still active at data snapshot)."
    )
    
    # Methods
    pdf.section_title("2. Three Complementary Methods")
    
    pdf.subsection("Method A: Gradient-Boosted Classifiers")
    pdf.body_text(
        "Two LightGBM classifiers predict event priority (high/low) and road closure "
        "(yes/no). Priority model achieves ROC-AUC 0.9995, F1 0.9993. Road closure "
        "model achieves PR-AUC 0.9946 despite 92:8 class imbalance. The near-perfect "
        "metrics are not overfitting - corridor_clean alone nearly determines priority "
        "because BTP has different operating rules per corridor."
    )
    
    pdf.subsection("Method B: Survival Analysis (Censoring-Aware)")
    pdf.body_text(
        "Log-Normal Accelerated Failure Time model predicts event duration while correctly "
        "handling 937 right-censored events. C-index 0.7076, median absolute error 1.59h, "
        "calibration 75.3%. Key finding: vehicle breakdowns resolve in ~42 min while water "
        "logging and potholes often exceed the 24h observation window."
    )
    
    pdf.subsection("Method C: Spatio-Temporal Risk Forecasting")
    pdf.body_text(
        "Corridor x time-slot grid (22 corridors x 6 daily slots) with LightGBM Poisson "
        "objective. Complete grid includes zero-event cells (critical for count data). "
        "Enables proactive deployment: 'Mysore Road 20:00-24:00 expects 2.1 events - "
        "pre-position 6 personnel.'"
    )
    
    pdf.add_table(
        ["Method", "Model", "Primary Metric", "Value"],
        [
            ["A: Classification", "LightGBM", "ROC-AUC (Priority)", "0.9995"],
            ["A: Classification", "LightGBM", "PR-AUC (Closure)", "0.9946"],
            ["B: Survival", "Log-Normal AFT", "C-index", "0.7076"],
            ["C: Risk Forecast", "LightGBM Poisson", "RMSE", "1.510"],
        ],
        [35, 40, 50, 35]
    )
    
    # ── Page 2 ──
    pdf.section_title("3. Novelty Layers")
    
    pdf.subsection("Composite Impact Score (0-100)")
    pdf.body_text(
        "No severity metric exists in the data. We construct a transparent weighted proxy: "
        "duration (30%), priority (20%), road closure (20%), cause severity (15%), "
        "corridor importance (10%), rush hour (5%). Mean score: 43.0, max: 89.4. "
        "Four categories: Low (2%), Medium (79%), High (18%), Critical (0.8%)."
    )
    
    pdf.subsection("DBSCAN Hotspot Discovery")
    pdf.body_text(
        "38% of events (3,025) are off-grid. DBSCAN clustering (eps=500m, min_samples=5) "
        "reveals 79 unofficial hotspots. Notable: a 45-event tree_fall cluster near MSR Nagar "
        "suggests a specific vegetation/infrastructure issue BTP could prevent proactively."
    )
    
    pdf.subsection("Post-Event Learning Loop")
    pdf.body_text(
        "The problem statement says 'no post-event learning system.' We build one using "
        "EMA risk tracking (alpha=0.1) and Bayesian Beta closure probability. Key insight: "
        "Bannerghata Road drifted +7.4 risk points (biggest riser), suggesting BTP should "
        "reallocate resources from improving corridors (ORR East 1: -5.3) to rising ones."
    )
    
    # Key Findings
    pdf.section_title("4. Key Findings & Recommendations")
    
    pdf.bullet("Pre-position on Mysore Road & Bannerghata Road during 20:00-24:00 (highest risk)", "Deploy:")
    pdf.bullet("45-event tree_fall cluster near MSR Nagar - vegetation/infrastructure investigation", "Investigate:")
    pdf.bullet("From ORR East 1 to Bannerghata Road based on risk drift analysis", "Shift resources:")
    pdf.bullet("The post-event learning loop for continuously updated risk estimates", "Adopt:")
    pdf.bullet("The impact simulator for real-time resource allocation decisions", "Use:")
    
    pdf.ln(3)
    
    # Assumptions
    pdf.section_title("5. Stated Assumptions")
    pdf.bullet("No external data used - all analysis from provided dataset only", "Rule compliance:")
    pdf.bullet("Resource recommendations are calibrated heuristics, not learned from data", "Resources:")
    pdf.bullet("Timestamps are IST (confirmed by peak hour analysis)", "Timezone:")
    pdf.bullet("Corridor importance proxied by event density (no traffic volume data)", "Proxies:")
    
    pdf.ln(3)
    
    # Tech Stack
    pdf.section_title("6. Technical Stack & Deliverables")
    
    pdf.add_table(
        ["Deliverable", "Technology", "Purpose"],
        [
            ["pipeline.ipynb", "Python/Jupyter", "Code + documentation"],
            ["dashboard.py", "Streamlit/Plotly", "Interactive 6-tab dashboard"],
            ["react-dashboard/", "React/Vite", "Static results showcase"],
            ["models/", "LightGBM/lifelines", "Saved models & metrics"],
            ["Maps", "MapMyIndia (Mappls)", "Spatial visualization"],
        ],
        [45, 40, 75]
    )
    
    pdf.body_text(
        "Pipeline modules: data_loader.py, model_classification.py, model_survival.py, "
        "model_risk_forecast.py, impact_score.py, hotspot_discovery.py, post_event_learning.py. "
        "Total: ~2,500 lines of documented Python across 7 modules + 2 dashboard implementations."
    )
    
    # Save
    out_path = Path(__file__).parent / "executive_summary.pdf"
    pdf.output(str(out_path))
    print(f"PDF saved to {out_path}")
    return out_path


if __name__ == "__main__":
    generate_pdf()
