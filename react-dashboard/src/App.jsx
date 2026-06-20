import { useState } from 'react'
import './App.css'
import { RESULTS } from './data.js'

function App() {
  const [activeTab, setActiveTab] = useState('overview')

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'eventmap', label: 'Event Map' },
    { id: 'models', label: 'Models' },
    { id: 'corridors', label: 'Corridors' },
    { id: 'hotspots', label: 'Hotspots' },
    { id: 'learning', label: 'Learning' },
    { id: 'simulator', label: 'Simulator' },
  ]

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div>
            <h1>BTP Traffic Event Intelligence</h1>
            <p className="subtitle">Predictive Analytics for Bengaluru Traffic Police - Flipkart Grid 7.0 PS2</p>
          </div>
        </div>
      </header>

      <nav className="tab-nav">
        {tabs.map(tab => (
          <button key={tab.id} className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
                  onClick={() => setActiveTab(tab.id)}>{tab.label}</button>
        ))}
      </nav>

      <main className="main">
        {activeTab === 'overview' && <OverviewTab />}
        {activeTab === 'eventmap' && <EventMapTab />}
        {activeTab === 'models' && <ModelsTab />}
        {activeTab === 'corridors' && <CorridorsTab />}
        {activeTab === 'hotspots' && <HotspotsTab />}
        {activeTab === 'learning' && <LearningTab />}
        {activeTab === 'simulator' && <SimulatorTab />}
      </main>

      <footer className="footer">
        <p>Built for Flipkart Grid 7.0 | Powered by LightGBM, lifelines, OpenStreetMap</p>
      </footer>
    </div>
  )
}

function KPICard({ label, value, delta, color }) {
  return (
    <div className="kpi-card">
      <span className="kpi-label">{label}</span>
      <span className="kpi-value">{value}</span>
      {delta && <span className="kpi-delta">{delta}</span>}
    </div>
  )
}

function BarChart({ data, maxVal, color = '#6366f1' }) {
  const max = maxVal || Math.max(...data.map(d => d.value))
  return (
    <div className="bar-chart">
      {data.map((d, i) => (
        <div key={i} className="bar-row">
          <span className="bar-label">{d.label}</span>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${(d.value / max) * 100}%`, background: color }} />
          </div>
          <span className="bar-value">{typeof d.value === 'number' ? d.value.toLocaleString() : d.value}</span>
        </div>
      ))}
    </div>
  )
}

function ProgressRing({ value, max = 100, label, color = '#6366f1' }) {
  const pct = (value / max) * 100
  const radius = 40
  const circ = 2 * Math.PI * radius
  const offset = circ - (pct / 100) * circ
  return (
    <div className="progress-ring-container">
      <svg width="100" height="100" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r={radius} fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="6" />
        <circle cx="50" cy="50" r={radius} fill="none" stroke={color} strokeWidth="6"
                strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round"
                transform="rotate(-90 50 50)" style={{ transition: 'stroke-dashoffset 1s ease' }} />
        <text x="50" y="48" textAnchor="middle" fill="white" fontSize="16" fontWeight="700">
          {typeof value === 'number' ? value.toFixed(value < 10 ? 2 : 0) : value}
        </text>
        <text x="50" y="65" textAnchor="middle" fill="rgba(255,255,255,0.5)" fontSize="9">{label}</text>
      </svg>
    </div>
  )
}

// ════════════════════════════════════════════════════════════
// OVERVIEW TAB — with monthly trend + surge detection
// ════════════════════════════════════════════════════════════

function OverviewTab() {
  const d = RESULTS
  return (
    <div className="tab-content">
      <div className="kpi-grid">
        <KPICard label="Total Events" value="7,944" delta="Nov 2023 - Apr 2024" />
        <KPICard label="High Priority" value="4,921" delta="62%" />
        <KPICard label="Road Closures" value="673" delta="8.5%" />
        <KPICard label="Avg Impact" value="43.0" delta="Max: 89.4" />
        <KPICard label="Corridors" value="23" delta="22 named + off-grid" />
        <KPICard label="Censored Events" value="937" delta="11.8% active" />
      </div>

      {/* ─── Monthly Escalation Trend (Fix 6) ─── */}
      <div className="card">
        <h3>📈 Monthly Event Escalation — The Urgency Case</h3>
        <p className="card-desc">{d.escalationNote}</p>
        <div className="trend-chart">
          {d.monthlyTrend.map((m, i) => {
            const maxEvents = Math.max(...d.monthlyTrend.map(x => x.events))
            const isPartial = m.month.includes('Apr')
            return (
              <div key={i} className="trend-bar-col">
                <div className="trend-bar-wrapper">
                  <div className="trend-bar"
                       style={{
                         height: `${(m.events / maxEvents) * 140}px`,
                         background: isPartial ? 'rgba(239,68,68,0.4)' : `hsl(${260 - i * 20}, 70%, 55%)`,
                         border: isPartial ? '2px dashed #ef4444' : 'none'
                       }}>
                    <span className="trend-bar-value">{m.events.toLocaleString()}</span>
                  </div>
                </div>
                <span className="trend-label">{m.month.split(' ')[0]}</span>
                <span className="trend-sublabel">{m.closureRate}% closure</span>
              </div>
            )
          })}
        </div>
        <div className="escalation-callout">
          <strong>⚠️ +98.7% volume increase</strong> in 5 months. BTP's current resource model was calibrated for ~970 monthly events but now faces ~1,930. This system is needed now.
        </div>
      </div>

      <div className="card-grid-2">
        <div className="card">
          <h3>Events by Cause</h3>
          <BarChart data={d.causeCounts} color="#6366f1" />
        </div>
        <div className="card">
          <h3>Impact Categories</h3>
          <div className="impact-pills">
            {d.impactCategories.map((c, i) => (
              <div key={i} className={`impact-pill ${c.category.toLowerCase()}`}>
                <span className="pill-label">{c.category}</span>
                <span className="pill-count">{c.count.toLocaleString()}</span>
                <span className="pill-pct">{c.pct}%</span>
              </div>
            ))}
          </div>
          <div className="weight-breakdown">
            <h4>Score Components</h4>
            {d.weights.map((w, i) => (
              <div key={i} className="weight-row">
                <span>{w.name}</span>
                <div className="weight-bar"><div style={{ width: `${w.value * 100}%` }} /></div>
                <span>{(w.value * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ─── Bimodal Hour Pattern ─── */}
      <div className="card">
        <h3>Bimodal Hour Pattern (IST) - Peak at 9pm + 4-6am</h3>
        <div className="hour-chart">
          {d.hourlyPattern.map((h, i) => (
            <div key={i} className="hour-bar-container">
              <div className="hour-bar"
                   style={{ height: `${(h.count / Math.max(...d.hourlyPattern.map(x => x.count))) * 120}px`,
                            background: [19,20,21,22,4,5,6].includes(h.hour) ? '#ef4444' : '#6366f1' }}
                   title={`${h.hour}:00 - ${h.count} events`} />
              <span className="hour-label">{h.hour}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ─── Surge Detection (Fix 5) ─── */}
      <div className="card">
        <h3>🚨 Surge Detection — Concurrent Event Overload</h3>
        <p className="card-desc">Hours where event load exceeded 3× the hourly baseline. When BTP is stretched across named corridors, these mass-overload hours are when the system breaks.</p>
        <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
          <KPICard label="High-Load Hours" value="159" delta="3+ events/hour" />
          <KPICard label="Peak Simultaneous" value="56" delta="Non-Corridor zone" />
          <KPICard label="Hosur Road Peak" value="22" delta="3× weekly avg in 1hr" />
        </div>
        <div className="card-grid-2" style={{ marginBottom: 0 }}>
          <div>
            <h4 style={{ fontSize: '0.9rem', color: '#a5b4fc', marginBottom: '12px' }}>Corridor Surge Summary</h4>
            <table className="metrics-table">
              <thead><tr><th>Corridor</th><th>High-Load Hours</th><th>Peak</th></tr></thead>
              <tbody>
                {d.surgeCorridors.map((s, i) => (
                  <tr key={i}>
                    <td>{s.corridor}</td><td>{s.highLoadHours}</td>
                    <td><span className={`type-badge ${s.peakSimultaneous > 20 ? 'slow' : s.peakSimultaneous > 10 ? 'medium' : 'fast'}`}>{s.peakSimultaneous}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div>
            <h4 style={{ fontSize: '0.9rem', color: '#a5b4fc', marginBottom: '12px' }}>Top Surge Events</h4>
            <table className="metrics-table">
              <thead><tr><th>Date</th><th>Corridor</th><th>Count</th></tr></thead>
              <tbody>
                {d.topSurgeEvents.slice(0, 6).map((s, i) => (
                  <tr key={i} title={s.cause}>
                    <td>{s.date} {s.hour}</td><td>{s.corridor}</td>
                    <td><strong>{s.simultaneous}</strong></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}

// ════════════════════════════════════════════════════════════
// EVENT MAP TAB (unchanged)
// ════════════════════════════════════════════════════════════

function EventMapTab() {
  const d = RESULTS
  const [mapLayer, setMapLayer] = useState('Individual Events')
  const [selectedCauses, setSelectedCauses] = useState(['vehicle_breakdown','others','pot_holes','construction','water_logging'])
  const [priorityFilter, setPriorityFilter] = useState('All')

  const toggleCause = (cause) => {
    setSelectedCauses(prev =>
      prev.includes(cause) ? prev.filter(c => c !== cause) : [...prev, cause]
    )
  }

  // Filter events
  const filtered = (d.mapEvents || []).filter(e => {
    if (!selectedCauses.includes(e.cause)) return false
    if (priorityFilter === 'High Only' && e.priority !== 'high') return false
    if (priorityFilter === 'Low Only' && e.priority !== 'low') return false
    return true
  })

  // Build map HTML
  const buildMapHtml = () => {
    let markersJs = ''

    if (mapLayer === 'Individual Events') {
      markersJs = filtered.map(e => {
        const color = e.priority === 'high' ? '#ef4444' : '#6366f1'
        return `L.circleMarker([${e.lat},${e.lng}],{radius:4,fillColor:'${color}',color:'${color}',weight:1,opacity:0.8,fillOpacity:0.5}).addTo(map).bindPopup('<b>${e.cause.replace(/_/g,' ')}</b><br>Priority: ${e.priority}<br>Impact: ${e.impact}<br>Corridor: ${e.corridor}');`
      }).join('\n')
    } else if (mapLayer === 'Hotspot Clusters') {
      markersJs = (d.hotspots || []).map(h => {
        const color = h.risk > 150 ? '#ef4444' : h.risk > 80 ? '#f97316' : h.risk > 40 ? '#eab308' : '#6366f1'
        const radius = Math.max(Math.min(h.events / 5, 40), 8)
        return `L.circleMarker([${h.lat},${h.lng}],{radius:${radius},fillColor:'${color}',color:'${color}',weight:2,opacity:0.9,fillOpacity:0.35}).addTo(map).bindPopup('<b>${h.location}</b><br>Events: ${h.events}<br>Cause: ${h.cause}<br>Risk: ${h.risk}');`
      }).join('\n')
    } else {
      // Heatmap via leaflet-heat
      const heatData = filtered.map(e => `[${e.lat},${e.lng},${e.impact/100}]`).join(',')
      markersJs = `
        var script = document.createElement('script');
        script.src = 'https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js';
        script.onload = function() { L.heatLayer([${heatData}],{radius:20,blur:15,maxZoom:13}).addTo(map); };
        document.head.appendChild(script);
      `
    }

    return `<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"><\/script>
<style>html,body,#map{margin:0;padding:0;width:100%;height:100%;}</style>
</head><body>
<div id="map"></div>
<script>
var map = L.map('map').setView([12.9716, 77.5946], 11);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{
  attribution:'&copy; OpenStreetMap contributors',maxZoom:18
}).addTo(map);
${markersJs}
<\/script>
</body></html>`
  }

  return (
    <div className="tab-content">
      <h2 className="section-title">Event Map - Bengaluru Traffic Police</h2>
      <div className="map-controls">
        <div className="map-control-group">
          <label>Map Layer</label>
          <select value={mapLayer} onChange={e => setMapLayer(e.target.value)}>
            <option>Individual Events</option>
            <option>Event Heatmap</option>
            <option>Hotspot Clusters</option>
          </select>
        </div>
        <div className="map-control-group">
          <label>Priority</label>
          <select value={priorityFilter} onChange={e => setPriorityFilter(e.target.value)}>
            <option>All</option>
            <option>High Only</option>
            <option>Low Only</option>
          </select>
        </div>
        <div className="map-control-group cause-filters">
          <label>Filter by Cause</label>
          <div className="cause-checkboxes">
            {(d.allCauses || []).map(c => (
              <label key={c} className={`cause-chip ${selectedCauses.includes(c) ? 'active' : ''}`}>
                <input type="checkbox" checked={selectedCauses.includes(c)} onChange={() => toggleCause(c)} />
                {c.replace(/_/g,' ')}
              </label>
            ))}
          </div>
        </div>
      </div>
      <p className="map-count">Showing {filtered.length} of {(d.mapEvents||[]).length} events (300 sampled from 7,944)</p>
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <iframe
          key={mapLayer + priorityFilter + selectedCauses.join(',')}
          srcDoc={buildMapHtml()}
          style={{ width: '100%', height: '550px', border: 'none', borderRadius: '16px' }}
          title="Event Map"
        />
      </div>
      <div className="map-legend">
        <span className="legend-item"><span className="legend-dot" style={{background:'#ef4444'}} /> High Priority</span>
        <span className="legend-item"><span className="legend-dot" style={{background:'#6366f1'}} /> Low Priority</span>
      </div>
    </div>
  )
}

// ════════════════════════════════════════════════════════════
// MODELS TAB — reframed priority + tail risk caveat
// ════════════════════════════════════════════════════════════

function ModelsTab() {
  const d = RESULTS
  return (
    <div className="tab-content">
      <h2 className="section-title">Three Complementary Methods</h2>
      <div className="card-grid-3">
        {/* ─── Method A: REFRAMED (Fix 1) ─── */}
        <div className="card method-card">
          <div className="method-badge">Method A</div>
          <h3>Gradient-Boosted Classifiers</h3>
          <p>LightGBM classifiers — Priority (semantic) and Road Closure prediction</p>

          {/* Semantic Priority Model — the honest one */}
          <h4 style={{ color: '#a5b4fc', fontSize: '0.85rem', marginBottom: '8px' }}>A1: Semantic Priority Classifier</h4>
          <div className="metric-rings">
            <ProgressRing value={d.semanticPriorityMetrics.roc_auc} max={1} label="ROC-AUC" color="#6366f1" />
            <ProgressRing value={d.semanticPriorityMetrics.f1} max={1} label="F1 Score" color="#8b5cf6" />
          </div>
          <table className="metrics-table">
            <thead><tr><th>Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>F1</th></tr></thead>
            <tbody>
              <tr><td>Priority (semantic)</td><td>{d.semanticPriorityMetrics.roc_auc}</td><td>{d.semanticPriorityMetrics.pr_auc}</td><td>{d.semanticPriorityMetrics.f1}</td></tr>
              <tr><td>Closure</td><td>0.9997</td><td>0.9946</td><td>0.9935</td></tr>
            </tbody>
          </table>

          {/* Dataset Discovery */}
          <div className="key-insight" style={{ background: 'rgba(239,68,68,0.08)', borderColor: '#ef4444', marginTop: '12px' }}>
            <strong style={{ color: '#fca5a5' }}>🔍 Dataset Discovery:</strong> {d.priorityModelFraming.discovery}
          </div>
          <div className="key-insight" style={{ marginTop: '8px' }}>
            <strong>💡 Our Value:</strong> {d.priorityModelFraming.value}
          </div>
        </div>

        {/* ─── Method B: Survival (unchanged, strong) ─── */}
        <div className="card method-card">
          <div className="method-badge survival">Method B</div>
          <h3>Survival Analysis</h3>
          <p>Censoring-aware duration prediction (Log-Normal AFT)</p>
          <div className="metric-rings">
            <ProgressRing value={0.7076} max={1} label="C-index" color="#22c55e" />
            <ProgressRing value={75.3} max={100} label="Calibration" color="#eab308" />
          </div>
          <table className="metrics-table">
            <thead><tr><th>Model</th><th>C-index</th><th>Med AE</th><th>AIC</th></tr></thead>
            <tbody>
              <tr><td>Weibull</td><td>0.7054</td><td>1.69h</td><td>7,884</td></tr>
              <tr className="best-row"><td>LogNormal *</td><td>0.7076</td><td>1.59h</td><td>8,406</td></tr>
              <tr><td>Cox PH</td><td>0.7075</td><td>-</td><td>-</td></tr>
            </tbody>
          </table>
          <div className="key-insight">
            <strong>Key Differentiator:</strong> 937 active events treated as right-censored,
            not dropped. Most teams will bias toward short durations.
          </div>
        </div>

        {/* ─── Method C: Risk (unchanged, strong) ─── */}
        <div className="card method-card">
          <div className="method-badge risk">Method C</div>
          <h3>Spatio-Temporal Risk</h3>
          <p>Corridor × time-slot forecasting (LightGBM Poisson)</p>
          <div className="metric-rings">
            <ProgressRing value={0.460} max={1} label="MAE" color="#ef4444" />
            <ProgressRing value={1.510} max={3} label="RMSE" color="#f97316" />
          </div>
          <table className="metrics-table">
            <thead><tr><th>Feature</th><th colSpan="3">Importance (gain)</th></tr></thead>
            <tbody>
              {d.riskFeatures.map((f, i) => (
                <tr key={i}>
                  <td>{f.name}</td>
                  <td colSpan="3">
                    <div className="feature-bar"><div style={{ width: `${(f.importance / d.riskFeatures[0].importance) * 100}%` }} /></div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="key-insight">
            <strong>Value:</strong> Enables pre-positioning: "Deploy extra patrol on Bannerghata Rd evening shift - 3.7x baseline risk."
          </div>
        </div>
      </div>

      {/* ─── Spatial Rule Discovery Detail (Fix 1 continued) ─── */}
      <div className="card">
        <h3>🔍 Dataset Discovery — BTP Priority is a Spatial Routing Rule</h3>
        <p className="card-desc">
          Every named corridor is ~100% High priority. Non-corridor is ~100% Low. The original model with AUC 0.9995 was memorizing
          this rule, not predicting risk. We removed spatial features and built a model that learns from <strong>event semantics</strong>.
        </p>
        <div className="card-grid-2" style={{ marginBottom: 0 }}>
          <div>
            <h4 style={{ fontSize: '0.85rem', color: '#fca5a5', marginBottom: '10px' }}>Corridor → Priority (Deterministic)</h4>
            <table className="metrics-table">
              <thead><tr><th>Corridor</th><th>Events</th><th>% High Priority</th></tr></thead>
              <tbody>
                {d.corridorDeterminism.map((c, i) => (
                  <tr key={i} className={c.highPct === 0 ? 'low-row' : c.highPct < 100 ? 'med-row' : ''}>
                    <td>{c.corridor}</td><td>{c.events}</td>
                    <td style={{ color: c.highPct === 100 ? '#fca5a5' : c.highPct === 0 ? '#86efac' : '#fde047' }}>
                      {c.highPct}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div>
            <h4 style={{ fontSize: '0.85rem', color: '#86efac', marginBottom: '10px' }}>Event Cause → Priority (Real Signal)</h4>
            <table className="metrics-table">
              <thead><tr><th>Cause</th><th>Events</th><th>% High</th><th>Signal</th></tr></thead>
              <tbody>
                {d.eventCausePriorityRates.map((c, i) => (
                  <tr key={i}>
                    <td>{c.cause}</td><td>{c.total}</td><td>{c.highPct}%</td>
                    <td><span className={`type-badge ${c.signal === 'strong' ? 'slow' : c.signal === 'moderate' ? 'medium' : 'fast'}`}>{c.signal}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* ─── Tail Risk Caveat (Fix 2) ─── */}
      <div className="card">
        <h3>⚠️ Duration Estimate Confidence — Tail Risk Caveat</h3>
        <p className="card-desc">{d.tailRiskNote.summary}</p>
        <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)', marginBottom: '16px' }}>
          <KPICard label="Closed Events" value={d.tailRiskNote.totalClosed.toLocaleString()} />
          <KPICard label="With Timestamps" value={d.tailRiskNote.withTimestamp.toLocaleString()} delta={`${(100 - d.tailRiskNote.missingPct).toFixed(1)}% coverage`} />
          <KPICard label="Missing Duration" value={`${d.tailRiskNote.missingPct}%`} delta="Systematically biased" />
        </div>
        <table className="metrics-table">
          <thead><tr><th>Cause</th><th>Closed</th><th>With Timestamp</th><th>Coverage</th><th>Bias Factor</th><th>Confidence</th></tr></thead>
          <tbody>
            {d.durationCoverage.map((c, i) => {
              const confColor = { good: '#86efac', moderate: '#fde047', low: '#fdba74', very_low: '#fca5a5', none: '#ef4444' }
              return (
                <tr key={i}>
                  <td>{c.cause}</td>
                  <td>{c.closedTotal}</td>
                  <td>{c.withTimestamp}</td>
                  <td>{c.coverage}%</td>
                  <td>{c.bias < 100 ? `${c.bias}×` : '∞'}</td>
                  <td><span className={`type-badge ${c.confidence === 'good' ? 'fast' : c.confidence === 'moderate' ? 'medium' : 'slow'}`}>{c.confidence.replace('_', ' ')}</span></td>
                </tr>
              )
            })}
          </tbody>
        </table>
        <div className="key-insight" style={{ background: 'rgba(239,68,68,0.08)', borderColor: '#ef4444', marginTop: '12px' }}>
          <strong style={{ color: '#fca5a5' }}>⚠️ Caveat:</strong> {d.tailRiskNote.caveat}
        </div>
      </div>
    </div>
  )
}

// ════════════════════════════════════════════════════════════
// CORRIDORS TAB — with Planned Event Playbook + Diversions
// ════════════════════════════════════════════════════════════

function CorridorsTab() {
  const d = RESULTS
  return (
    <div className="tab-content">
      <h2 className="section-title">Corridor Risk and Resource Analysis</h2>
      <div className="card">
        <h3>Top 10 Corridors by Mean Impact</h3>
        <BarChart data={d.corridorRisk} color="#f97316" maxVal={55} />
      </div>
      <div className="card-grid-2">
        <div className="card">
          <h3>Resource Recommendations per Category</h3>
          <table className="metrics-table">
            <thead><tr><th>Category</th><th>Personnel</th><th>Barricades</th><th>Vehicles</th><th>Cost (INR)</th></tr></thead>
            <tbody>
              <tr className="low-row"><td>Low</td><td>2</td><td>0</td><td>0</td><td>2,000</td></tr>
              <tr className="med-row"><td>Medium</td><td>4</td><td>4</td><td>1</td><td>8,000</td></tr>
              <tr className="high-row"><td>High</td><td>8</td><td>8</td><td>2</td><td>20,000</td></tr>
              <tr className="crit-row"><td>Critical</td><td>15</td><td>12</td><td>3</td><td>50,000</td></tr>
            </tbody>
          </table>
          <p className="disclaimer">Note: Base heuristic — adjusted by corridor-specific multipliers in the Simulator</p>
        </div>
        <div className="card">
          <h3>Duration by Cause (Kaplan-Meier)</h3>
          <table className="metrics-table">
            <thead><tr><th>Cause</th><th>Median</th><th>Censored</th><th>Type</th></tr></thead>
            <tbody>
              {d.kmResults.map((k, i) => (
                <tr key={i}>
                  <td>{k.cause}</td><td>{k.median}</td><td>{k.censored}</td>
                  <td><span className={`type-badge ${k.type}`}>{k.type}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ─── Planned Event Playbook (Fix 4) ─── */}
      <div className="card">
        <h3>📋 Planned Event Playbook — Pre-Deployment Protocol</h3>
        <p className="card-desc">
          {d.plannedVsUnplanned.insight}. Planned events ({d.plannedVsUnplanned.planned.total}) have a{' '}
          <strong>{d.plannedVsUnplanned.planned.closureRate}%</strong> road closure rate vs{' '}
          <strong>{d.plannedVsUnplanned.unplanned.closureRate}%</strong> for unplanned — a{' '}
          <strong>{d.plannedVsUnplanned.ratio}× difference</strong>.
          These are events BTP knows about in advance — forecasting has maximum operational value here.
        </p>
        <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)', marginBottom: '16px' }}>
          <KPICard label="Planned Events" value={d.plannedVsUnplanned.planned.total} />
          <KPICard label="Planned Closures" value={d.plannedVsUnplanned.planned.closures} delta={`${d.plannedVsUnplanned.planned.closureRate}%`} />
          <KPICard label="Closure Rate Gap" value={`${d.plannedVsUnplanned.ratio}×`} delta="Planned vs Unplanned" />
          <KPICard label="Unplanned Closures" value={d.plannedVsUnplanned.unplanned.closures} delta={`${d.plannedVsUnplanned.unplanned.closureRate}%`} />
        </div>
        <table className="metrics-table">
          <thead>
            <tr>
              <th>Event Type</th><th>Count</th><th>Closure Rate</th><th>Expected Duration</th>
              <th>Advance Deploy</th><th>Personnel</th><th>Barricades</th><th>Protocol</th>
            </tr>
          </thead>
          <tbody>
            {d.plannedEventPlaybook.map((p, i) => (
              <tr key={i} className={p.closureRate > 50 ? 'high-row' : p.closureRate > 35 ? 'med-row' : ''}>
                <td><strong>{p.cause}</strong></td>
                <td>{p.totalPlanned}</td>
                <td style={{ color: p.closureRate > 50 ? '#fca5a5' : p.closureRate > 35 ? '#fdba74' : '#fde047' }}>
                  {p.closureRate}%
                </td>
                <td>{p.expectedDurationH}h</td>
                <td>{p.advanceDeployH}h before</td>
                <td>{p.personnel}</td>
                <td>{p.barricades}</td>
                <td style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{p.notes}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="key-insight" style={{ marginTop: '12px' }}>
          <strong>PS Alignment:</strong> This directly answers the problem statement requirement for "manpower and barricading plans"
          with data-backed numbers derived from 467 planned events in the dataset.
        </div>
      </div>

      {/* ─── Corridor Profiles (Fix 3 data) ─── */}
      <div className="card">
        <h3>📊 Corridor Profiles — Operational Metrics</h3>
        <p className="card-desc">Each corridor has unique operational characteristics. CBD 2's long resolution (2.54h median) means more personnel tied up per incident than Tumkur Road (0.88h). Same impact score requires very different resource postures.</p>
        <table className="metrics-table">
          <thead>
            <tr><th>Corridor</th><th>Events/Week</th><th>Med. Resolution</th><th>Closure Rate</th><th>Load Tier</th><th>Resource Multiplier</th></tr>
          </thead>
          <tbody>
            {d.corridorProfiles.map((c, i) => (
              <tr key={i}>
                <td>{c.corridor}</td>
                <td>{c.eventsPerWeek}</td>
                <td>{c.medianResHours}h</td>
                <td style={{ color: c.closureRate > 10 ? '#fca5a5' : c.closureRate > 5 ? '#fde047' : '#86efac' }}>{c.closureRate}%</td>
                <td><span className={`type-badge ${c.loadTier === 'heavy' ? 'slow' : c.loadTier === 'medium' ? 'medium' : 'fast'}`}>{c.loadTier}</span></td>
                <td><strong>{c.multiplier}×</strong></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* ─── Diversion Corridor Table (Fix 7) ─── */}
      <div className="card">
        <h3>🔀 Diversion Corridor Adjacency</h3>
        <p className="card-desc">When a corridor is blocked, these are the recommended alternate routes based on spatial adjacency. This directly addresses the PS requirement for "diversion plans."</p>
        <table className="metrics-table">
          <thead><tr><th>Blocked Corridor</th><th>Alternate Route 1</th><th>Alternate Route 2</th><th>Alternate Route 3</th></tr></thead>
          <tbody>
            {Object.entries(d.diversionTable).slice(0, 12).map(([corridor, alts], i) => (
              <tr key={i}>
                <td><strong>{corridor.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}</strong></td>
                {alts.map((a, j) => <td key={j}>{a}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ════════════════════════════════════════════════════════════
// HOTSPOTS TAB (unchanged — this is a strong section)
// ════════════════════════════════════════════════════════════

function LeafletMap({ hotspots }) {
  const mapHtml = `
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
      <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
      <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"><\/script>
      <style>html,body,#map{margin:0;padding:0;width:100%;height:100%;}</style>
    </head>
    <body>
      <div id="map"></div>
      <script>
        var map = L.map('map').setView([12.9716, 77.5946], 11);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          attribution: '&copy; OpenStreetMap contributors',
          maxZoom: 18
        }).addTo(map);
        var hotspots = ${JSON.stringify(hotspots)};
        hotspots.forEach(function(h) {
          var color = h.risk > 150 ? '#ef4444' : h.risk > 80 ? '#f97316' : h.risk > 40 ? '#eab308' : '#6366f1';
          var radius = Math.max(Math.min(h.events / 5, 40), 8);
          L.circleMarker([h.lat, h.lng], {
            radius: radius,
            fillColor: color,
            color: color,
            weight: 2,
            opacity: 0.9,
            fillOpacity: 0.35
          }).addTo(map).bindPopup(
            '<b>' + h.location + '</b><br>' +
            'Events: ' + h.events + '<br>' +
            'Top Cause: ' + h.cause + '<br>' +
            'Risk Score: ' + h.risk + '<br>' +
            'Radius: ' + h.radius
          );
        });
      <\/script>
    </body>
    </html>
  `
  return (
    <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
      <h3 style={{ padding: '16px 24px 0' }}>Hotspot Map - Bengaluru</h3>
      <iframe
        srcDoc={mapHtml}
        style={{ width: '100%', height: '450px', border: 'none', borderRadius: '0 0 16px 16px' }}
        title="Hotspot Map"
      />
    </div>
  )
}

function HotspotsTab() {
  const d = RESULTS
  return (
    <div className="tab-content">
      <h2 className="section-title">Hidden Hotspot Discovery (DBSCAN)</h2>
      <p className="section-desc">
        38% of events (3,025) are off-grid - not on any of BTP's 22 named corridors.
        DBSCAN clustering reveals <strong>79 unofficial hotspots</strong> the corridor list misses.
      </p>
      <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
        <KPICard label="Non-Corridor Events" value="3,025" />
        <KPICard label="Hotspots Found" value="79" />
        <KPICard label="Clustered" value="90.3%" delta="2,733 events" />
        <KPICard label="Noise Points" value="292" delta="9.7%" />
      </div>
      <LeafletMap hotspots={d.hotspots} />
      <div className="card">
        <h3>Top 10 Hotspots</h3>
        <table className="metrics-table">
          <thead><tr><th>#</th><th>Location</th><th>Events</th><th>Top Cause</th><th>Radius</th><th>Risk</th></tr></thead>
          <tbody>
            {d.hotspots.map((h, i) => (
              <tr key={i}>
                <td>{i + 1}</td><td>{h.location}</td><td>{h.events}</td><td>{h.cause}</td><td>{h.radius}</td>
                <td><div className="risk-badge" style={{ background: h.risk > 200 ? 'rgba(239,68,68,0.15)' : 'rgba(99,102,241,0.15)' }}>{h.risk}</div></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ════════════════════════════════════════════════════════════
// LEARNING TAB (unchanged — this is a strong section)
// ════════════════════════════════════════════════════════════

function LearningTab() {
  const d = RESULTS
  return (
    <div className="tab-content">
      <h2 className="section-title">Post-Event Learning Loop</h2>
      <p className="section-desc">
        The problem statement says "no post-event learning system." We built exactly that:
        per-corridor EMA risk + Bayesian Beta closure probability, updated after each event.
      </p>
      <div className="card-grid-2">
        <div className="card">
          <h3>Risk Risers (Train to Test)</h3>
          <p className="card-desc">Corridors that got MORE dangerous over time</p>
          {d.riskDrift.risers.map((r, i) => (
            <div key={i} className="drift-row riser">
              <span className="drift-name">{r.corridor}</span>
              <span className="drift-values">{r.from} to {r.to}</span>
              <span className="drift-delta">+{r.drift}</span>
            </div>
          ))}
        </div>
        <div className="card">
          <h3>Risk Fallers (Train to Test)</h3>
          <p className="card-desc">Corridors that improved over time</p>
          {d.riskDrift.fallers.map((r, i) => (
            <div key={i} className="drift-row faller">
              <span className="drift-name">{r.corridor}</span>
              <span className="drift-values">{r.from} to {r.to}</span>
              <span className="drift-delta">{r.drift}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="card">
        <h3>How It Works</h3>
        <div className="method-flow">
          <div className="flow-step"><h4>Event Resolved</h4><p>BTP closes/resolves an event with outcome data</p></div>
          <div className="flow-arrow">&rarr;</div>
          <div className="flow-step"><h4>EMA Update</h4><p>risk_t = 0.1 × severity + 0.9 × risk_(t-1)</p></div>
          <div className="flow-arrow">&rarr;</div>
          <div className="flow-step"><h4>Bayesian Update</h4><p>Beta(a, b) posterior for closure probability</p></div>
          <div className="flow-arrow">&rarr;</div>
          <div className="flow-step"><h4>Adjust Deployment</h4><p>Reallocate resources to rising-risk corridors</p></div>
        </div>
      </div>
    </div>
  )
}

// ════════════════════════════════════════════════════════════
// SIMULATOR TAB — corridor multipliers + diversion suggestion
// ════════════════════════════════════════════════════════════

function SimulatorTab() {
  const d = RESULTS
  const [cause, setCause] = useState('vehicle_breakdown')
  const [corridor, setCorridor] = useState('mysore road')
  const [hour, setHour] = useState(20)
  const [priority, setPriority] = useState('High')
  const [closure, setClosure] = useState(false)

  const causes = ['vehicle_breakdown','accident','water_logging','pot_holes','construction','tree_fall','congestion','others','vip_movement','procession','public_event']
  const corridors = ['mysore road','bellary road 1','bellary road 2','tumkur road','hosur road','bannerghata road','magadi road','orr north 1','orr north 2','orr east 1','orr east 2','orr west 1','old madras road','varthur road','cbd 2','hennur main road','west of chord road','old airport road']

  const CAUSE_SEV = {accident:1.0,water_logging:0.7,pot_holes:0.4,construction:0.6,vehicle_breakdown:0.35,tree_fall:0.8,congestion:0.55,others:0.5,vip_movement:0.85,procession:0.8,public_event:0.9}
  const CAUSE_DUR = {vehicle_breakdown:42,accident:42,water_logging:720,pot_holes:1440,construction:600,tree_fall:480,congestion:120,others:120,vip_movement:150,procession:180,public_event:360}

  // Corridor-specific multiplier (Fix 3)
  const corridorProfile = d.corridorProfiles.find(
    c => c.corridor.toLowerCase() === corridor
  ) || { multiplier: 1.0, eventsPerWeek: 0, medianResHours: 0, closureRate: 0, loadTier: 'unknown' }
  const corridorMultiplier = corridorProfile.multiplier

  // Auto-calculate on every state change
  const durEst = CAUSE_DUR[cause] || 120
  const durScore = Math.log1p(durEst) / Math.log1p(1440)
  const priScore = priority === 'High' ? 1 : 0
  const cloScore = closure ? 1 : 0
  const rushScore = [7,8,9,17,18,19,20].includes(hour) ? 1 : 0
  const causeScore = CAUSE_SEV[cause] || 0.5
  const corImportance = 0.5 // static proxy

  const components = [
    { name: 'Duration (30%)', weight: 0.30, raw: durScore },
    { name: 'Priority (20%)', weight: 0.20, raw: priScore },
    { name: 'Road Closure (20%)', weight: 0.20, raw: cloScore },
    { name: 'Cause Severity (15%)', weight: 0.15, raw: causeScore },
    { name: 'Corridor Importance (10%)', weight: 0.10, raw: corImportance },
    { name: 'Rush Hour (5%)', weight: 0.05, raw: rushScore },
  ]

  const impact = components.reduce((sum, c) => sum + c.weight * c.raw, 0) * 100
  const cat = impact > 75 ? 'Critical' : impact > 50 ? 'High' : impact > 25 ? 'Medium' : 'Low'
  const resMap = {Low:[2,0,0,2000],Medium:[4,4,1,8000],High:[8,8,2,20000],Critical:[15,12,3,50000]}
  let [p,b,v,c_cost] = resMap[cat]

  // Apply multipliers
  if (rushScore) p = Math.round(p*1.5)
  if (closure) { p = Math.round(p*1.3); b = Math.round(b*1.3) }
  // Corridor-specific multiplier (Fix 3)
  p = Math.round(p * corridorMultiplier)
  b = Math.round(b * corridorMultiplier)

  // Diversion data (Fix 7)
  const diversions = d.diversionTable[corridor] || []

  // Tail risk duration adjustment (Fix 2)
  const durationCoverageEntry = d.durationCoverage.find(
    dc => dc.cause.toLowerCase().replace(/ /g, '_') === cause
  )
  const isLowCoverage = durationCoverageEntry && durationCoverageEntry.coverage < 30
  const adjustedDurEst = isLowCoverage ? Math.round(durEst * 1.75) : durEst

  return (
    <div className="tab-content">
      <h2 className="section-title">Event Impact Simulator</h2>
      <p className="section-desc">Adjust any input below - results update instantly. Resource recommendations are now corridor-aware.</p>
      <div className="card sim-card">
        <div className="sim-grid">
          <div className="sim-field">
            <label>Event Cause</label>
            <select value={cause} onChange={e => setCause(e.target.value)}>
              {causes.map(c => <option key={c} value={c}>{c.replace(/_/g,' ')}</option>)}
            </select>
          </div>
          <div className="sim-field">
            <label>Corridor</label>
            <select value={corridor} onChange={e => setCorridor(e.target.value)}>
              {corridors.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div className="sim-field">
            <label>Hour (IST): {hour}:00 {rushScore ? '(RUSH HOUR)' : '(off-peak)'}</label>
            <input type="range" min="0" max="23" value={hour} onChange={e => setHour(+e.target.value)} />
          </div>
          <div className="sim-field">
            <label>Priority</label>
            <select value={priority} onChange={e => setPriority(e.target.value)}>
              <option>High</option><option>Low</option>
            </select>
          </div>
          <div className="sim-field">
            <label><input type="checkbox" checked={closure} onChange={e => setClosure(e.target.checked)} /> Road Closure Required</label>
          </div>
        </div>

        <div className="sim-result">
          <div className={`impact-badge ${cat.toLowerCase()}`}>{cat} Impact - {impact.toFixed(1)}/100</div>

          {/* Corridor Profile Card (Fix 3) */}
          <div className="corridor-profile-card" style={{
            display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '8px',
            padding: '12px', background: 'rgba(99,102,241,0.06)', borderRadius: '8px',
            marginBottom: '16px', fontSize: '0.8rem', textAlign: 'center'
          }}>
            <div><span style={{ color: 'var(--text-muted)', display: 'block' }}>Events/Week</span><strong>{corridorProfile.eventsPerWeek}</strong></div>
            <div><span style={{ color: 'var(--text-muted)', display: 'block' }}>Med. Resolution</span><strong>{corridorProfile.medianResHours}h</strong></div>
            <div><span style={{ color: 'var(--text-muted)', display: 'block' }}>Closure Rate</span><strong>{corridorProfile.closureRate}%</strong></div>
            <div><span style={{ color: 'var(--text-muted)', display: 'block' }}>Load Tier</span><strong>{corridorProfile.loadTier}</strong></div>
            <div><span style={{ color: 'var(--text-muted)', display: 'block' }}>Resource ×</span><strong style={{ color: corridorMultiplier > 1.2 ? '#fca5a5' : '#86efac' }}>{corridorMultiplier}×</strong></div>
          </div>

          <div className="sim-kpis">
            <KPICard label="Impact Score" value={`${impact.toFixed(1)}/100`} />
            <KPICard label="Est. Duration" value={`${Math.floor(adjustedDurEst/60)}h ${adjustedDurEst%60}m`} delta={isLowCoverage ? '⚠️ adjusted +75%' : ''} />
            <KPICard label="Personnel" value={p} delta={corridorMultiplier > 1 ? `×${corridorMultiplier} corridor` : ''} />
            <KPICard label="Barricades" value={b} delta={corridorMultiplier > 1 ? `×${corridorMultiplier} corridor` : ''} />
            <KPICard label="Vehicles" value={v} />
            <KPICard label="Est. Cost" value={`₹${c_cost.toLocaleString()}`} />
          </div>

          {/* Tail Risk Warning (Fix 2) */}
          {isLowCoverage && (
            <div className="key-insight" style={{ background: 'rgba(239,68,68,0.08)', borderColor: '#ef4444', marginBottom: '16px' }}>
              <strong style={{ color: '#fca5a5' }}>⚠️ Low-Coverage Cause:</strong> Only {durationCoverageEntry.coverage}% of closed {cause.replace(/_/g,' ')} events have duration data.
              Estimate adjusted upward by 75% as a conservative lower bound.
            </div>
          )}

          <h4 style={{marginBottom:'12px', color:'#a5b4fc'}}>Score Breakdown</h4>
          <table className="metrics-table">
            <thead><tr><th>Component</th><th>Weight</th><th>Raw Value</th><th>Contribution</th></tr></thead>
            <tbody>
              {components.map((comp, i) => (
                <tr key={i}>
                  <td>{comp.name}</td>
                  <td>{(comp.weight*100).toFixed(0)}%</td>
                  <td>{comp.raw.toFixed(2)}</td>
                  <td>{(comp.weight*comp.raw*100).toFixed(1)}</td>
                </tr>
              ))}
              <tr style={{fontWeight:700, borderTop:'2px solid rgba(255,255,255,0.1)'}}>
                <td>Total</td><td>100%</td><td></td><td>{impact.toFixed(1)}</td>
              </tr>
            </tbody>
          </table>

          <div className="deployment-rec" style={{marginTop:'16px'}}>
            <h4>Deployment Recommendation</h4>
            <p>Deploy <strong>{p} personnel</strong> to {corridor} with <strong>{b} barricades</strong> and <strong>{v} patrol vehicle(s)</strong>.</p>
            <p>Estimated resolution: ~{adjustedDurEst} min ({(adjustedDurEst/60).toFixed(1)}h) | Rush hour: <strong>{rushScore ? 'YES (hours 7-9, 17-20)' : 'No - off-peak'}</strong></p>
            {corridorMultiplier > 1 && (
              <p style={{ color: '#fdba74' }}>📊 Corridor multiplier applied: <strong>{corridorMultiplier}×</strong> ({corridorProfile.loadTier} load, {corridorProfile.closureRate}% closure rate)</p>
            )}
          </div>

          {/* Diversion Suggestion (Fix 7) */}
          {diversions.length > 0 && (
            <div className="deployment-rec" style={{ marginTop: '12px', borderLeftColor: '#22c55e' }}>
              <h4 style={{ color: '#86efac' }}>🔀 Diversion Plan</h4>
              <p>If <strong>{corridor}</strong> is blocked, recommended alternate routes:</p>
              <div style={{ display: 'flex', gap: '8px', marginTop: '8px', flexWrap: 'wrap' }}>
                {diversions.map((alt, i) => (
                  <span key={i} style={{
                    padding: '6px 14px', background: 'rgba(34,197,94,0.1)', border: '1px solid rgba(34,197,94,0.3)',
                    borderRadius: '20px', fontSize: '0.85rem', fontWeight: 500
                  }}>
                    {i === 0 ? '🥇' : i === 1 ? '🥈' : '🥉'} {alt}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
