# BTP Traffic Event Intelligence System
### Event-Driven Congestion (Planned & Unplanned) (Flipkart GridLock 2.0 — Problem Statement 2)

This repository contains the complete implementation of the **Traffic Event Intelligence System** designed for the Bengaluru Traffic Police (BTP). The system transitions BTP operations from **reactive incident responses** to **proactive resource deployment and simulator-driven dispatching**.

---

## 1. Problem Reframing & Critical Discoveries

The provided dataset contains 8,173 raw incident logs. Through exploratory data analysis, we established two crucial dataset anomalies and reframed the modeling strategy accordingly:

### A. The Priority Model Data Leakage (FATAL Fix)
*   **Discovery**: The baseline priority model had an AUC of **0.9995**. Analysis revealed this was due to **data leakage**: BTP's ASTraM system assigns "High" priority to all named corridors (100% of 3,124 corridor events) and "Low" priority to all non-corridors. The model had memorized a spatial routing rule rather than predicting actual priority.
*   **Resolution**: We separated the priority classifier into two components:
    1.  **Spatial Rule Discovery**: A classifier (AUC 0.9995) showing the exact routing rule BTP uses.
    2.  **Semantic Priority Classifier**: A new, honest model built entirely without spatial fields (`corridor_clean`, `zone`, `police_station`, etc.) using only event cause, vehicle type, and time features. It achieves a realistic, leakage-free **ROC-AUC of 0.6256** and **F1 of 0.7649**, predicting priority *pre-dispatch* (before the location is verified).

### B. Duration Missing Data Bias (CRITICAL Fix)
*   **Discovery**: $55.8\%$ of resolved events lack resolution timestamps. This missingness is heavily biased—congestion is **4.1x** and accidents are **2.5x** more likely to have missing timestamps. A raw analysis would heavily underestimate duration tail-risks.
*   **Resolution**: 
    *   We built a **Censoring-Aware Survival Analysis** model using right-censoring.
    *   We added explicit "Tail Risk Caveats" across the models tab.
    *   In simulator operations, duration estimates for low-coverage causes (congestion, construction) are adjusted upward by **+75%** to represent conservative lower-bounds.

### C. Monthly Event Volume Escalation (+98.7%)
*   **Discovery**: Monthly event volume surged by **+98.7%** from November (972 events) to March (1,931 events), with road closure rates rising from $5.2\%$ to $9.1\%$. The current resource allocation models must adapt to this escalating traffic load.

---

## 2. Core Predictive Methodology

The intelligence system is built on **three complementary modeling methods**:

```
                  ┌───────────────────────────────────────────────┐
                  │          BTP RAW INCIDENT STREAM              │
                  └──────────────────────┬────────────────────────┘
                                         │
                 ┌───────────────────────┼───────────────────────┐
                 ▼                       ▼                       ▼
      ┌──────────────────────┐┌──────────────────────┐┌──────────────────────┐
      │       METHOD A       ││       METHOD B       ││       METHOD C       │
      │Classification Engine ││  Survival Analysis   ││Risk Grid Forecasting │
      │  Priority (Semantic) ││   (Right-Censored)   ││ (LightGBM Poisson)  │
      │  Road Closure (LGBM) ││  (Log-Normal AFT)   ││ (Corridor x Slot)   │
      └──────────────────────┘└──────────────────────┘└──────────────────────┘
```

### Method A: Classification Engine
*   **Semantic Priority Classifier**: Predicts priority pre-dispatch (ROC-AUC: 0.6256, F1: 0.7649).
*   **Road Closure Classifier**: Predicts if an event will necessitate a road closure. Uses a scale parameter of 13.6 to handle severe class imbalance ($9.1\%$ closures). Achieves a **ROC-AUC of 0.9997** and a **PR-AUC of 0.9946**.

### Method B: Censoring-Aware Survival Analysis
*   Predicts event resolution time while properly accounting for 937 active (right-censored) events using a **Log-Normal Accelerated Failure Time (AFT)** model.
*   **Concordance Index (C-Index)**: **0.7076**.
*   **Median Absolute Error**: **1.59 hours** (observed events).
*   *Key Insight*: Vehicle breakdowns resolve quickly (median 42 minutes), whereas potholes and water logging frequently exceed 24 hours.

### Method C: Spatio-Temporal Risk Forecasting
*   Models traffic density using a Poisson objective on a daily grid (22 corridors x 6 daily time slots). Includes zero-incident periods to avoid positive-class prediction bias.
*   **Validation RMSE**: **1.510** (outperforming the historical average baseline).
*   *Operational Application*: Predicts upcoming hourly event spikes (e.g., Mysore Road expects 2.1 concurrent events between 20:00 and 24:00) so supervisors can proactively position officers.

---

## 3. Operational Novelty Layers

To make the predictive models actionable for BTP, we implemented several operational layers:

### 1. Composite Impact Score (0 to 100)
A multi-dimensional scoring proxy that evaluates event severity:
$$\text{Impact} = 30\%\text{ Duration} + 20\%\text{ Closure} + 20\%\text{ Priority} + 15\%\text{ Cause} + 10\%\text{ Corridor} + 5\%\text{ RushHour}$$
Events are categorized into: **Low** ($2.0\%$), **Medium** ($78.9\%$), **High** ($18.3\%$), and **Critical** ($0.8\%$).

### 2. Corridor-Specific Resource Multipliers
Replaces static resource allocations with corridor-specific multipliers (1.1x to 1.5x) based on historic closure and event frequencies:
*   *Varthur Road* ($11.8\%$ closures) $\rightarrow$ **1.5x barricade/officer multiplier**.
*   *Mysore Road* ($11.2\%$ closures) $\rightarrow$ **1.5x multiplier**.
*   *Magadi Road* ($4.1\%$ closures) $\rightarrow$ **1.1x multiplier**.

### 3. Proactive Planned Event Playbook
Planned events represent only $5.9\%$ of occurrences but are **5.4x** more likely to cause road closures than unplanned incidents ($36.2\%$ vs $6.7\%$). The system defines tactical playbooks:
*   **VIP Movement** (80.0% closure rate) $\rightarrow$ Pre-deploy 16 officers + barricades 2.0h before.
*   **Processions / Rallies** (42.1% closure rate) $\rightarrow$ Pre-deploy 6 officers 1.5h before.
*   **Public Events** (46.4% closure rate) $\rightarrow$ Pre-deploy 10 officers 1.5h before.

### 4. Unsupervised Hotspot Discovery (DBSCAN)
Clustering off-grid coordinates (latitude/longitude) with DBSCAN ($eps=500m$, $min\_samples=5$) revealed **79 unofficial hotspots**. 
*   *Highlight*: Discovered a cluster of **45 tree falls** near MSR Nagar, indicating a localized infrastructure vulnerability that BBMP can address proactively by pruning branches.

### 5. Post-Event Learning Loop
A dynamic Bayesian updating tracker (detailed in `learning.md`) that updates corridor risk scores and closure probabilities in real-time as tickets are closed.
*   *Risk Drift*: Identified **Bannerghatta Road** as the fastest-growing risk zone ($+7.4$ drift) and **ORR East 1** as the fastest-improving zone ($-5.3$ drift), telling BTP to reallocate patrol officers.

---

## 4. Codebase Architecture

```
├── Astram event data_anonymized.csv  # Raw BTP incident logs
├── data_loader.py                    # Cleaning, feature engineering, and train/test splitting
├── model_classification.py          # Priority and road closure classifiers (LightGBM)
├── model_survival.py                # Censoring-aware Log-Normal AFT model (lifelines)
├── model_risk_forecast.py           # Spatio-temporal Poisson risk grid model
├── impact_score.py                  # Composite impact scores and resource recommendations
├── hotspot_discovery.py             # DBSCAN spatial clustering for off-grid hotpots
├── post_event_learning.py           # Bayesian updating loop and risk drift tracker
├── dashboard.py                     # Streamlit application (dispatcher simulator + interactive map)
├── generate_pdf.py                  # Compiles the 3-page executive summary PDF
├── executive_summary.pdf            # Compiled executive summary document
├── learning.md                      # post-event learning loop documentation
├── requirements.txt                 # Python dependencies
└── react-dashboard/                 # React showcase board (Vite, CSS, interactive visualizations)
    ├── src/
    │   ├── App.jsx                  # Main interface (Overview, Models, Corridors, Simulator tabs)
    │   ├── data.js                  # Precompiled model outputs and stats
    │   └── index.css                # Custom styling system
    └── index.html
```

---

## 5. Getting Started & Setup

### Prerequisites
*   Python 3.8+
*   Node.js (for React dashboard)

### Python Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

Verify/Run the pipeline modules:
```bash
# Train & compare priority (spatial vs semantic) and road closure classifiers:
python model_classification.py

# Train censoring-aware survival duration models:
python model_survival.py

# Run spatio-temporal risk forecasting:
python model_risk_forecast.py

# Test post-event learning loop and generate risk drift records:
python post_event_learning.py

# Re-generate the executive summary PDF:
python generate_pdf.py
```

### Launching the Dashboards

#### 1. Interactive Dispatch Simulator (Streamlit)
Features a live dispatcher simulator, interactive GIS maps, and hotspot analyses.
```bash
streamlit run dashboard.py
```

#### 2. Presentation Board (React/Vite)
A highly polished single-page dashboard showcasing model statistics, priority reframing, and playbook visualizers.
```bash
cd react-dashboard
npm install
npm run dev
```
Open [http://localhost:5174/](http://localhost:5174/) to view the web dashboard.
To build a production build:
```bash
npm run build
```
