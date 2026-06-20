"""
generate_pdf.py - Generate executive_summary.pdf

A tight 3-page judge-skimmable PDF with:
- Problem reframing & data leakage discovery
- Three methods + semantic priority model + key results
- Stated assumptions
- Latest features: tail risk caveats, corridor multipliers, planned event playbooks, surge detection
- Actionable recommendations & technical stack
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
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(50, 50, 80)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(99, 102, 241)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)
    
    def subsection(self, title):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(70, 70, 100)
        self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)
    
    def body_text(self, text):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5, text)
        self.ln(2)
    
    def bullet(self, text, bold_prefix=""):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(40, 40, 40)
        x = self.get_x()
        self.cell(4)
        self.cell(3, 5, "-")  # bullet
        if bold_prefix:
            self.set_font("Helvetica", "B", 9.5)
            self.write(5, bold_prefix + " ")
            self.set_font("Helvetica", "", 9.5)
        self.multi_cell(0, 5, text)
        self.ln(1)
    
    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [(self.w - self.l_margin - self.r_margin) / len(headers)] * len(headers)
        
        # Header
        self.set_font("Helvetica", "B", 8.5)
        self.set_fill_color(240, 240, 250)
        self.set_text_color(50, 50, 80)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 6.5, h, border=1, fill=True, align="C")
        self.ln()
        
        # Rows
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(40, 40, 40)
        for row in rows:
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 5.5, str(cell), border=1, align="C")
            self.ln()
        self.ln(2.5)


def generate_pdf():
    pdf = SummaryPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # ── Page 1 ──
    pdf.add_page()
    
    # Title Block
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(50, 50, 80)
    pdf.cell(0, 10, "BTP Traffic Event Intelligence System", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(99, 102, 241)
    pdf.cell(0, 6, "Proactive Dispatch and Resource Optimization Engine", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "I", 9.5)
    pdf.set_text_color(100, 100, 120)
    pdf.cell(0, 5, "Flipkart Grid 7.0 - Problem Statement 2 - Executive Summary", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)
    
    # Problem Reframing
    pdf.section_title("1. Problem Reframing and Data Discovery")
    pdf.body_text(
        "The baseline dataset (8,173 raw incident logs) contains 94.1% unplanned road disruptions "
        "(vehicle breakdowns, potholes, water logging, accidents) rather than planned public events. "
        "We reframe the task: every logged incident is an 'event' consuming traffic management "
        "resources, requiring modeling of impact, duration, and road closure probability."
    )
    
    pdf.body_text(
        "Data Quality Discoveries and Leakage Fixes:\n"
        "- PRIORITY LEAKAGE: The original priority classifier achieved ROC-AUC 0.9995 because BTP's internal "
        "routing rule assigns 'High' priority to all named corridors and 'Low' priority to non-corridors. "
        "This is spatial memorization, not prediction. We reframed this as a 'Spatial Rule Discovery' and built "
        "a separate, honest 'Semantic Priority Classifier' excluding spatial fields to predict priority pre-dispatch.\n"
        "- DURATION BIAS: 55.8% of resolved events lack timestamps. Missingness is highly biased - congestion "
        "is 4.1x more likely to miss timestamps than accidents. Raw duration averages are severely underestimated."
    )
    
    # Methods
    pdf.section_title("2. Core Predictive Methodology")
    
    pdf.subsection("Method A: Classification Engine (Priority and Road Closure)")
    pdf.body_text(
        "- Semantic Priority Model: Uses only event-level variables (cause, vehicle type, severity, time) to "
        "predict priority. Achieves an honest ROC-AUC of 0.6256 and F1 of 0.7649, providing a reliable pre-dispatch signal.\n"
        "- Road Closure Model: LightGBM model with scale_pos_weight = 13.6 to handle extreme class imbalance (9.1% closures). "
        "Achieves ROC-AUC of 0.9997, PR-AUC of 0.9946, and F1 of 0.9935, predicting which incidents will shut down lanes."
    )
    
    pdf.subsection("Method B: Censoring-Aware Survival Analysis")
    pdf.body_text(
        "- Log-Normal AFT model predicts the resolution duration of active incidents while properly handling "
        "937 right-censored (still active) events. C-index: 0.7076. Mean Absolute Error: 1.59 hours.\n"
        "- Tail Risk Mitigation: In simulator operations, duration estimates for low-coverage event causes (construction, "
        "congestion) are adjusted upwards by +75% to counter missing-data bias, marking them as lower bounds."
    )
    
    pdf.subsection("Method C: Spatio-Temporal Risk Forecasting")
    pdf.body_text(
        "- LightGBM Poisson Regressor trained on a 22-corridor x 6-time-slot daily grid (132 cells) including zero-incident periods. "
        "Predicts the expected rate of concurrent traffic events in upcoming periods (RMSE: 1.510). Helps shift BTP from "
        "reactive dispatching to proactive officer pre-positioning."
    )
    
    pdf.add_table(
        ["Method", "Model Type", "Primary Evaluation Metric", "Value", "Status"],
        [
            ["A: Priority (Spatial)", "LightGBM Classifier", "ROC-AUC / F1 Score", "0.9995 / 0.9993", "Rule Discovery"],
            ["A: Priority (Semantic)", "LightGBM Classifier", "ROC-AUC / F1 Score", "0.6256 / 0.7649", "Leakage-Free"],
            ["A: Road Closure", "LightGBM Classifier", "PR-AUC / F1 Score", "0.9946 / 0.9935", "Robust"],
            ["B: Event Duration", "Log-Normal AFT", "Concordance Index (C-Index)", "0.7076", "Censoring-Aware"],
            ["C: Risk Forecasting", "Poisson LightGBM", "Root Mean Square Error (RMSE)", "1.510", "Zero-Inflated"],
        ],
        [32, 42, 50, 32, 29]
    )
    
    # ── Page 2 ──
    pdf.add_page()
    
    # Novelty Layers
    pdf.section_title("3. Operational Novelty Layers")
    
    pdf.subsection("Composite Impact Score (0 to 100)")
    pdf.body_text(
        "Since the raw logs lack an explicit severity rank, we engineered a weighted multi-dimensional "
        "severity proxy: Event Duration (30%), Road Closure (20%), Priority (20%), Cause Severity (15%), "
        "Corridor Load (10%), and Rush Hour (5%). The score segments events into four tiers: Low (2.0%), "
        "Medium (78.9%), High (18.3%), and Critical (0.8%). This serves as the foundation for resource allocation."
    )
    
    pdf.subsection("Corridor-Specific Operational Multipliers")
    pdf.body_text(
        "To replace static, uniform resource allocations (e.g., standard response cards), we implemented "
        "operational multipliers (1.1x to 1.5x) based on historical corridor load. High-risk corridors like "
        "Varthur Road (11.8% closure rate) and Mysore Road (11.2% closure rate) receive a 1.5x multiplier "
        "on personnel and barricades, whereas light-load corridors (e.g., Magadi Road) use a 1.1x baseline."
    )
    
    pdf.subsection("Proactive Planned Event Playbook")
    pdf.body_text(
        "Planned events represent 5.9% of the dataset but are 5.4x more likely to cause road closures than "
        "unplanned events (36.2% vs 6.7%). We mapped out a tactical playbook for BTP dispatchers:\n"
        "- VIP Movement: Pre-deploy 16 personnel + barricades 2 hours in advance (80.0% historical closure rate).\n"
        "- Processions / Rallies: Pre-deploy 6 personnel 1.5 hours in advance (42.1% closure rate).\n"
        "- Public Events: Pre-deploy 10 personnel 1.5 hours in advance (46.4% closure rate).\n"
        "- Construction: Maintain 4 personnel ongoing (29.9% closure rate)."
    )
    
    pdf.subsection("Spatio-Temporal Surge Detection")
    pdf.body_text(
        "Concurrent event spikes can overwhelm local police stations. We implemented a threshold-based "
        "surge detection engine identifying periods where active events exceed 3x the corridor's baseline. "
        "Historical logs reveal 159 high-load hours, peaking at 56 simultaneous events across the city. "
        "The system triggers dispatch alerts when a corridor's active load breaches its 90th percentile."
    )
    
    pdf.subsection("Unsupervised Hotspot Discovery (DBSCAN)")
    pdf.body_text(
        "Using spatial coordinates (latitude and longitude), DBSCAN clustering (eps=500m, min_samples=5) "
        "identified 79 unofficial incident hotspots outside the 23 named corridors. A high-density cluster of "
        "45 tree-fall incidents near MSR Nagar highlights localized infrastructure vulnerabilities that warrant "
        "preventative municipal trimming."
    )
    
    pdf.subsection("Post-Event Learning & Risk Drift Tracker")
    pdf.body_text(
        "A recursive Bayesian learning loop updates corridor risk profiles after every event resolution. "
        "We track risk drift over time: Bannerghatta Road represents the highest upward risk drift (+7.4 risk points) "
        "while ORR East 1 showed the highest recovery (-5.3 risk points). This allows BTP to dynamically redirect "
        "personnel from stable zones to rising risk corridors."
    )
    
    # ── Page 3 ──
    pdf.add_page()
    
    # Key Recommendations
    pdf.section_title("4. Strategic Recommendations for BTP Deployment")
    
    pdf.bullet(
        "Shift 10-15% of tactical traffic personnel from ORR East 1 (improving risk) to "
        "Bannerghatta Road and Mysore Road (rising risk profiles and high historical closure rates).",
        "Dynamic Reallocation:"
    )
    pdf.bullet(
        "Prioritize officer pre-positioning on Mysore Road and Outer Ring Road corridors between 20:00 "
        "and 24:00 daily, where the Poisson model forecasts a peak average event rate of 2.1 simultaneous disruptions.",
        "Spatio-Temporal Pre-positioning:"
    )
    pdf.bullet(
        "Integrate the Planned Event Playbook into BTP's central ASTraM dispatch system to mandate pre-dispatch "
        "barricade staging 1.5 to 2 hours before VIP movements and public rallies.",
        "Staged Deployment:"
    )
    pdf.bullet(
        "Use the 75% duration inflation adjustment for construction and congestion events to communicate honest, "
        "conservative clearance times to commuter navigation systems (Google Maps, Mappls) instead of biased raw averages.",
        "Public Communication:"
    )
    pdf.bullet(
        "Collaborate with the BBMP forest department to run targeted pruning campaigns at the discovered "
        "DBSCAN tree-fall hotspot near MSR Nagar (45 historic blockages) to eliminate risk at the source.",
        "Targeted Infrastructure Fixes:"
    )
    
    pdf.ln(3)
    
    # Stated Assumptions
    pdf.section_title("5. Stated Assumptions & Constraints")
    pdf.bullet("The analysis relies entirely on the provided 8,173 anonymized historical incident records.", "Data Boundary:")
    pdf.bullet("Event timestamps are processed as Indian Standard Time (IST), verified by midnight-hour activity drop.", "Time Zone:")
    pdf.bullet("Corridor importance is proxied by historical event density (due to lack of physical vehicle count sensors).", "Volume Proxy:")
    pdf.bullet("Personnel and barricade requirements are calibrated heuristics based on impact severity tiers.", "Resource Scaling:")
    
    pdf.ln(3)
    
    # Technical Stack & Deliverables
    pdf.section_title("6. Deliverables & Technical Architecture")
    pdf.add_table(
        ["Component", "Tech Stack", "Operational Role"],
        [
            ["pipeline.ipynb", "Python, Jupyter", "End-to-end data cleaning, training, and evaluations"],
            ["model_classification.py", "LightGBM, Scikit-Learn", "Saves models/priority_model_semantic.txt and closure models"],
            ["model_survival.py", "Lifelines, Log-Normal", "Generates censored-aware duration predictions"],
            ["model_risk_forecast.py", "LightGBM Poisson", "Calculates spatio-temporal risk densities for grid cells"],
            ["dashboard.py", "Streamlit, Plotly, Folium", "Interactive operator console for real-time dispatch simulator"],
            ["react-dashboard/", "React 18, Vite, CSS, Tailwind", "Premium presentation board visualizing results and metrics"],
        ],
        [40, 48, 87]
    )
    
    pdf.body_text(
        "The codebase encompasses ~2,500 lines of production-grade Python across 7 modules, "
        "complete with a local Streamlit dashboard for dispatcher simulation, and a high-performance "
        "Vite-based React single-page application showcasing the interactive results."
    )
    
    # Save PDF
    out_path = Path(__file__).parent / "executive_summary.pdf"
    pdf.output(str(out_path))
    print(f"PDF successfully compiled and saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    generate_pdf()
