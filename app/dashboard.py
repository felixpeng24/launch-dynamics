"""Streamlit dashboard for interactive landing dynamics simulation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from landing_sim.simulation import run_simulation, load_config
from landing_sim.monte_carlo import run_monte_carlo
from landing_sim.loads import compute_statistics, compute_load_envelope, format_statistics_table
from landing_sim import ai_analyst

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Landing Dynamics | Starship-Class Simulation",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/SpaceX-Logo.svg/220px-SpaceX-Logo.svg.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme Constants ──────────────────────────────────────────
COLORS = {
    "bg": "#0E1117",
    "surface": "#1A1D23",
    "accent": "#00A3E0",
    "accent_dim": "#006B94",
    "success": "#00C853",
    "warning": "#FF6D00",
    "danger": "#FF3131",
    "text": "#FFFFFF",
    "text_dim": "#8B949E",
    "border": "#30363D",
    "chart_1": "#00A3E0",
    "chart_2": "#00C853",
    "chart_3": "#FF6D00",
    "chart_4": "#A855F7",
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,29,35,0.8)",
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
        colorway=[COLORS["chart_1"], COLORS["chart_2"], COLORS["chart_3"], COLORS["chart_4"]],
        margin=dict(l=40, r=20, t=50, b=40),
    )
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif !important; }

    /* Hide default streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1A1D23 0%, #21252B 100%);
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] {
        color: #8B949E !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 1.8rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricDelta"] {
        color: #00C853 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background: #1A1D23;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid #30363D;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 24px;
        color: #8B949E;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #00A3E0 !important;
        color: white !important;
    }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00A3E0 0%, #0080B3 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
        padding: 8px 24px !important;
        transition: all 0.2s !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 0 20px rgba(0, 163, 224, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button {
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
        background: #1A1D23 !important;
        color: #C9D1D9 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #161B22 !important;
        border-right: 1px solid #30363D;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #00A3E0 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 24px;
    }

    /* Cards / containers */
    [data-testid="stExpander"] {
        background: #1A1D23;
        border: 1px solid #30363D;
        border-radius: 12px;
    }

    /* Code blocks */
    .stCodeBlock { border-radius: 8px !important; }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00A3E0, #00C853) !important;
    }

    /* Info / warning boxes */
    .stAlert { border-radius: 8px !important; }

    /* Hero section divider */
    .hero-divider {
        height: 2px;
        background: linear-gradient(90deg, #00A3E0, transparent);
        margin: 8px 0 24px 0;
        border: none;
    }

    /* Status indicator */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    .status-dot.active { background: #00C853; }
    .status-dot.idle { background: #8B949E; }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Stat grid */
    .stat-label { color: #8B949E; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 2px; }
    .stat-value { color: #FFFFFF; font-size: 1.4rem; font-weight: 600; }
    .stat-unit { color: #8B949E; font-size: 0.85rem; margin-left: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Load Config ──────────────────────────────────────────────
config = load_config()


# ── Hero Header ──────────────────────────────────────────────
hero_col1, hero_col2 = st.columns([3, 1])
with hero_col1:
    st.markdown(
        '<h1 style="margin-bottom:0; font-weight:700; font-size:2.2rem;">'
        'Landing Dynamics Simulation</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#8B949E; font-size:1rem; margin-top:4px;">'
        '3-DOF Propulsive Landing &bull; Nonlinear Contact &bull; Monte Carlo Analysis'
        '</p>',
        unsafe_allow_html=True,
    )
with hero_col2:
    st.markdown(
        '<div style="text-align:right; padding-top:12px;">'
        '<span style="color:#8B949E; font-size:0.8rem;">VEHICLE MODEL</span><br>'
        '<span style="color:#FFFFFF; font-size:1.1rem; font-weight:600;">Starship-Class</span><br>'
        '<span style="color:#00A3E0; font-size:0.85rem;">105,000 kg &bull; 50 m &bull; 6 legs</span>'
        '</div>',
        unsafe_allow_html=True,
    )
st.markdown('<div class="hero-divider"></div>', unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="text-align:center; padding:16px 0 8px 0;">'
        '<span style="font-size:1.3rem; font-weight:700; color:#FFFFFF;">FLIGHT CONFIG</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("### Vehicle")
    mass = st.number_input("Dry Mass (kg)", value=config["vehicle"]["dry_mass"],
                           step=1000.0, format="%.0f")
    prop = st.number_input("Landing Propellant (kg)",
                           value=config["vehicle"]["landing_propellant"],
                           step=100.0, format="%.0f")
    config["vehicle"]["dry_mass"] = mass
    config["vehicle"]["landing_propellant"] = prop

    st.markdown("### Propulsion")
    thrust_mode = st.selectbox("Thrust Mode", ["simple", "multi_engine"],
                               format_func=lambda x: "Single Engine" if x == "simple" else "Multi-Engine (3x)")
    config["thrust"]["mode"] = thrust_mode
    max_thrust = st.number_input("Max Thrust / Engine (MN)",
                                 value=config["thrust"]["simple"]["max_thrust"] / 1e6,
                                 step=0.1, format="%.1f")
    config["thrust"]["simple"]["max_thrust"] = max_thrust * 1e6
    config["thrust"]["multi_engine"]["thrust_per_engine"] = max_thrust * 1e6

    st.markdown("### Contact Model")
    stiffness_exp = st.slider("Stiffness (10^x N/m^n)", 6.0, 10.0,
                              np.log10(config["contact"]["stiffness"]), 0.1)
    config["contact"]["stiffness"] = 10 ** stiffness_exp
    damping = st.number_input("Damping (N*s/m)", value=config["contact"]["damping"],
                              step=50000.0, format="%.0f")
    config["contact"]["damping"] = damping

    st.markdown("### Initial State")
    ic_alt = st.number_input("Altitude (m)", value=config["initial_conditions"]["altitude"],
                             step=10.0)
    ic_vz = st.number_input("Descent Rate (m/s)",
                            value=config["initial_conditions"]["vertical_velocity"],
                            step=1.0)
    config["initial_conditions"]["altitude"] = ic_alt
    config["initial_conditions"]["vertical_velocity"] = ic_vz

    st.markdown("---")
    st.markdown(
        '<p style="color:#8B949E; font-size:0.7rem; text-align:center;">'
        'Parameters based on publicly available data</p>',
        unsafe_allow_html=True,
    )


# ── Plotly helper ────────────────────────────────────────────
def apply_dark_theme(fig, height=500):
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        height=height,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text_dim"], size=11),
        ),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig


# ── Tabs ─────────────────────────────────────────────────────
tab_single, tab_mc, tab_ai = st.tabs(["SINGLE CASE", "MONTE CARLO", "AI ANALYST"])

# ==============================================================
# TAB 1: SINGLE CASE
# ==============================================================
with tab_single:
    st.markdown("")

    ctrl_col, spacer, info_col = st.columns([1.2, 0.1, 0.7])

    with ctrl_col:
        st.markdown(
            '<p style="color:#8B949E; font-size:0.75rem; text-transform:uppercase; '
            'letter-spacing:0.1em; margin-bottom:12px;">Touchdown Dispersions</p>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        pitch_deg = c1.number_input("Pitch Angle (deg)", value=0.0, step=0.5,
                                    help="Initial pitch offset from vertical")
        lat_vel = c2.number_input("Lateral Velocity (m/s)", value=0.0, step=0.1,
                                  help="Initial lateral drift rate")
        run_single = st.button("EXECUTE SIMULATION", type="primary", use_container_width=True)

    with info_col:
        total_mass = config["vehicle"]["dry_mass"] + config["vehicle"]["landing_propellant"]
        twr = (config["thrust"]["simple"]["max_thrust"]) / (total_mass * 9.80665)
        st.markdown(
            f'<div style="background:#1A1D23; border:1px solid #30363D; border-radius:12px; padding:16px;">'
            f'<p class="stat-label">Total Mass</p>'
            f'<p class="stat-value">{total_mass/1000:.0f}<span class="stat-unit">t</span></p>'
            f'<p class="stat-label" style="margin-top:12px;">Thrust-to-Weight</p>'
            f'<p class="stat-value">{twr:.2f}</p>'
            f'<p class="stat-label" style="margin-top:12px;">Burn Altitude</p>'
            f'<p class="stat-value">{config["initial_conditions"]["altitude"]:.0f}'
            f'<span class="stat-unit">m</span></p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if run_single:
        ic_over = {
            "pitch_angle": np.radians(pitch_deg),
            "lateral_velocity": lat_vel,
        }
        with st.spinner("Integrating equations of motion..."):
            result = run_simulation(config, ic_overrides=ic_over)

        # Status banner
        if result.settled:
            st.success("NOMINAL TOUCHDOWN  |  Vehicle settled successfully")
        else:
            st.warning("SIMULATION COMPLETE  |  Vehicle did not fully settle")

        # Metrics row
        if result.touchdown_idx is not None and "axial" in result.total_forces:
            m1, m2, m3, m4 = st.columns(4)
            td_vz = result.z_dot[result.touchdown_idx]
            m1.metric("Touchdown Velocity", f"{td_vz:.2f} m/s")
            m2.metric("Peak Axial Force", f"{np.max(result.total_forces['axial'])/1e6:.2f} MN")
            m3.metric("Peak Lateral Force",
                      f"{np.max(np.abs(result.total_forces['lateral']))/1e6:.3f} MN")
            m4.metric("Peak Moment",
                      f"{np.max(np.abs(result.total_forces['moment']))/1e6:.2f} MN-m")

        # Time history
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=("CG Altitude", "Vertical Velocity", "Contact Forces",
                          "Lateral Position", "Lateral Velocity", "Pitch Angle"),
            vertical_spacing=0.12, horizontal_spacing=0.08,
        )

        traces = [
            (result.t, result.z, "Altitude (m)", 1, 1, COLORS["chart_1"]),
            (result.t, result.z_dot, "Vz (m/s)", 1, 2, COLORS["chart_1"]),
            (result.t, result.x, "x (m)", 2, 1, COLORS["chart_3"]),
            (result.t, result.x_dot, "Vx (m/s)", 2, 2, COLORS["chart_3"]),
            (result.t, np.degrees(result.theta), "Pitch (deg)", 2, 3, COLORS["chart_2"]),
        ]
        for x, y, name, row, col, color in traces:
            fig.add_trace(go.Scatter(x=x, y=y, name=name,
                                     line=dict(color=color, width=2)), row=row, col=col)

        if "axial" in result.total_forces:
            fig.add_trace(go.Scatter(
                x=result.t, y=result.total_forces["axial"] / 1e6,
                name="Axial (MN)", line=dict(color=COLORS["chart_1"], width=2),
                fill="tozeroy", fillcolor="rgba(0,163,224,0.15)"),
                row=1, col=3)
            fig.add_trace(go.Scatter(
                x=result.t, y=np.abs(result.total_forces["lateral"]) / 1e6,
                name="|Lateral| (MN)", line=dict(color=COLORS["chart_3"], width=2)),
                row=1, col=3)

        apply_dark_theme(fig, height=550)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================
# TAB 2: MONTE CARLO
# ==============================================================
with tab_mc:
    st.markdown("")

    ctrl_col, spacer, info_col = st.columns([1.2, 0.1, 0.7])

    with ctrl_col:
        st.markdown(
            '<p style="color:#8B949E; font-size:0.75rem; text-transform:uppercase; '
            'letter-spacing:0.1em; margin-bottom:12px;">Campaign Configuration</p>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        n_cases = c1.number_input("Number of Cases", value=100, min_value=10,
                                  max_value=10000, step=50)
        n_workers = c2.number_input("Parallel Workers", value=1, min_value=1, max_value=16)
        run_mc = st.button("LAUNCH MONTE CARLO", type="primary", use_container_width=True)

    with info_col:
        disp = config["monte_carlo"]["dispersions"]
        st.markdown(
            f'<div style="background:#1A1D23; border:1px solid #30363D; border-radius:12px; padding:16px;">'
            f'<p class="stat-label">Dispersion Parameters</p>'
            f'<p class="stat-value">9</p>'
            f'<p class="stat-label" style="margin-top:12px;">Sampling Method</p>'
            f'<p class="stat-value" style="font-size:1.1rem;">Latin Hypercube</p>'
            f'<p class="stat-label" style="margin-top:12px;">Velocity 3-sigma</p>'
            f'<p class="stat-value">{disp["vertical_velocity"]["sigma"]*3:.1f}'
            f'<span class="stat-unit">m/s</span></p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if run_mc:
        progress = st.progress(0, text="Initializing Monte Carlo campaign...")

        def update_progress(done, total):
            pct = done / total
            progress.progress(pct, text=f"Simulating case {done:,} / {total:,}  "
                              f"({pct*100:.0f}%)")

        mc_results = run_monte_carlo(config, n_cases=n_cases,
                                     n_workers=n_workers,
                                     progress_callback=update_progress)
        progress.empty()

        st.session_state["mc_results"] = mc_results
        stats = compute_statistics(mc_results)
        st.session_state["mc_stats"] = stats
        settled_pct = np.mean(mc_results.settled_flags) * 100

        # Summary metrics
        st.markdown(
            '<p style="color:#8B949E; font-size:0.75rem; text-transform:uppercase; '
            'letter-spacing:0.1em; margin:24px 0 12px 0;">Results Summary</p>',
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Cases Run", f"{n_cases:,}")
        m2.metric("Mean Peak Axial", f"{stats['peak_axial_force'].mean/1e6:.2f} MN")
        m3.metric("P95 Peak Axial", f"{stats['peak_axial_force'].p95/1e6:.2f} MN")
        m4.metric("Max Peak Axial", f"{stats['peak_axial_force'].max_val/1e6:.2f} MN")
        m5.metric("Mean TD Velocity", f"{stats['touchdown_vz'].mean:.2f} m/s")

        # Full statistics table
        with st.expander("Full Statistics Table", expanded=False):
            st.code(format_statistics_table(stats))

        # Charts in 2x2 grid
        st.markdown(
            '<p style="color:#8B949E; font-size:0.75rem; text-transform:uppercase; '
            'letter-spacing:0.1em; margin:24px 0 12px 0;">Load Distributions</p>',
            unsafe_allow_html=True,
        )

        # Histograms
        fig_hist = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                "Peak Axial Force (MN)", "Peak Lateral Force (MN)", "Peak Moment (MN-m)"),
        )

        for i, (data, color, name) in enumerate([
            (mc_results.peak_axial / 1e6, COLORS["chart_1"], "Axial"),
            (mc_results.peak_lateral / 1e6, COLORS["chart_3"], "Lateral"),
            (mc_results.peak_moment / 1e6, COLORS["chart_2"], "Moment"),
        ]):
            fig_hist.add_trace(go.Histogram(
                x=data, nbinsx=40, name=name,
                marker=dict(color=color, line=dict(color="rgba(255,255,255,0.1)", width=0.5)),
                opacity=0.85,
            ), row=1, col=i + 1)
            # Add mean line
            fig_hist.add_vline(x=np.mean(data), line_dash="dash",
                              line_color=COLORS["text"], line_width=1,
                              row=1, col=i + 1)
            # Add P95 line
            fig_hist.add_vline(x=np.percentile(data, 95), line_dash="dot",
                              line_color=COLORS["warning"], line_width=1,
                              row=1, col=i + 1)

        apply_dark_theme(fig_hist, height=350)
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Load envelope and scatter side by side
        chart_l, chart_r = st.columns(2)

        with chart_l:
            st.markdown(
                '<p style="color:#8B949E; font-size:0.75rem; text-transform:uppercase; '
                'letter-spacing:0.1em;">Load Envelope</p>',
                unsafe_allow_html=True,
            )
            envelope = compute_load_envelope(mc_results.peak_axial, mc_results.peak_lateral,
                                             "Peak Axial", "Peak Lateral")
            fig_env = go.Figure()
            fig_env.add_trace(go.Scatter(
                x=envelope.x_values / 1e6, y=envelope.y_values / 1e6,
                mode="markers",
                marker=dict(size=4, color=COLORS["chart_1"], opacity=0.4),
                name="Cases",
            ))
            if envelope.hull_vertices is not None:
                hull = np.vstack([envelope.hull_vertices, envelope.hull_vertices[0]])
                fig_env.add_trace(go.Scatter(
                    x=hull[:, 0] / 1e6, y=hull[:, 1] / 1e6,
                    mode="lines", line=dict(color=COLORS["warning"], width=2.5),
                    name="Convex Hull",
                ))
            fig_env.update_layout(
                xaxis_title="Peak Axial Force (MN)",
                yaxis_title="Peak Lateral Force (MN)",
            )
            apply_dark_theme(fig_env, height=450)
            st.plotly_chart(fig_env, use_container_width=True)

        with chart_r:
            st.markdown(
                '<p style="color:#8B949E; font-size:0.75rem; text-transform:uppercase; '
                'letter-spacing:0.1em;">Velocity vs. Peak Load</p>',
                unsafe_allow_html=True,
            )
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=mc_results.touchdown_vz,
                y=mc_results.peak_axial / 1e6,
                mode="markers",
                marker=dict(
                    size=5,
                    color=np.degrees(mc_results.touchdown_theta),
                    colorscale=[[0, "#00A3E0"], [0.5, "#FFFFFF"], [1, "#FF6D00"]],
                    colorbar=dict(
                        title=dict(text="Pitch (deg)", font=dict(color=COLORS["text_dim"])),
                        tickfont=dict(color=COLORS["text_dim"]),
                    ),
                    opacity=0.7,
                ),
                name="Cases",
            ))
            fig_scatter.update_layout(
                xaxis_title="Touchdown Vz (m/s)",
                yaxis_title="Peak Axial Force (MN)",
            )
            apply_dark_theme(fig_scatter, height=450)
            st.plotly_chart(fig_scatter, use_container_width=True)

# ==============================================================
# TAB 3: AI ANALYST
# ==============================================================
with tab_ai:
    st.markdown("")

    if not ai_analyst.is_available():
        st.markdown(
            '<div style="background:#1A1D23; border:1px solid #30363D; border-radius:12px; '
            'padding:32px; text-align:center;">'
            '<p style="font-size:1.5rem; margin-bottom:8px;">AI Analysis Unavailable</p>'
            '<p style="color:#8B949E;">Set <code>OPENAI_API_KEY</code> as an environment '
            'variable or in Streamlit secrets to enable GPT-4o powered results analysis.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    elif "mc_stats" not in st.session_state:
        st.markdown(
            '<div style="background:#1A1D23; border:1px solid #30363D; border-radius:12px; '
            'padding:32px; text-align:center;">'
            '<p style="font-size:1.5rem; margin-bottom:8px;">No Results to Analyze</p>'
            '<p style="color:#8B949E;">Run a Monte Carlo campaign first, then return here '
            'to ask the AI analyst about your results.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        stats = st.session_state["mc_stats"]
        mc_results = st.session_state["mc_results"]
        settled_pct = np.mean(mc_results.settled_flags) * 100
        context = ai_analyst.serialize_results(stats, mc_results.n_cases, settled_pct)

        st.markdown(
            '<p style="color:#8B949E; font-size:0.75rem; text-transform:uppercase; '
            'letter-spacing:0.1em; margin-bottom:16px;">Suggested Questions</p>',
            unsafe_allow_html=True,
        )

        suggested = ai_analyst.get_default_questions()
        cols = st.columns(3)
        for i, q in enumerate(suggested[:6]):
            if cols[i % 3].button(q, key=f"suggest_{i}", use_container_width=True):
                st.session_state["ai_question"] = q

        st.markdown("")
        question = st.text_input(
            "Ask the analyst:",
            value=st.session_state.get("ai_question", ""),
            placeholder="e.g., Which parameters most influence the peak axial force?",
        )

        if question:
            if st.button("ANALYZE", type="primary"):
                with st.spinner("AI analyst is reviewing the data..."):
                    response = ai_analyst.query(question, context)

                st.markdown(
                    '<div style="background:#1A1D23; border-left:3px solid #00A3E0; '
                    'border-radius:0 12px 12px 0; padding:20px; margin-top:16px;">',
                    unsafe_allow_html=True,
                )
                st.markdown(response)
                st.markdown('</div>', unsafe_allow_html=True)
