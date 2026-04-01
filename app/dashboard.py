"""Streamlit dashboard for interactive landing dynamics simulation."""

import sys
import base64
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Load hero image as base64 for CSS embedding
_ASSET_DIR = Path(__file__).parent / "assets"
_HERO_B64 = ""
_hero_path = _ASSET_DIR / "starship.jpg"
if _hero_path.exists():
    _HERO_B64 = base64.b64encode(_hero_path.read_bytes()).decode()

from landing_sim.simulation import run_simulation, load_config
from landing_sim.monte_carlo import run_monte_carlo
from landing_sim.loads import compute_statistics, compute_load_envelope, format_statistics_table
from landing_sim import ai_analyst

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Starship Landing Dynamics",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System ────────────────────────────────────────────
# SpaceX uses D-DIN. Closest free alternative: Barlow (geometric sans-serif)
# Monochrome palette: true black, white, grays. No accent colors.

COLORS = {
    "bg": "#000000",
    "surface": "#0A0A0A",
    "rule": "rgba(255,255,255,0.12)",
    "text": "#FFFFFF",
    "text_secondary": "rgba(255,255,255,0.6)",
    "text_tertiary": "rgba(255,255,255,0.35)",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="rgba(255,255,255,0.6)", family="Barlow, sans-serif", size=12),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.06)",
        zerolinecolor="rgba(255,255,255,0.08)",
        linecolor="rgba(255,255,255,0.12)",
        tickfont=dict(color="rgba(255,255,255,0.4)"),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.06)",
        zerolinecolor="rgba(255,255,255,0.08)",
        linecolor="rgba(255,255,255,0.12)",
        tickfont=dict(color="rgba(255,255,255,0.4)"),
    ),
    colorway=["#FFFFFF", "rgba(255,255,255,0.5)", "rgba(255,255,255,0.3)"],
    margin=dict(l=48, r=16, t=40, b=40),
)


def style_fig(fig, height=480):
    fig.update_layout(**PLOTLY_LAYOUT, height=height, showlegend=True,
                      legend=dict(bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="rgba(255,255,255,0.5)", size=11)))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)",
                     linecolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)",
                     linecolor="rgba(255,255,255,0.12)")
    return fig


# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@300;400;500;600;700;800;900&family=Barlow+Condensed:wght@400;500;600;700&display=swap');

    /* ── Base ── */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background: #000000 !important;
        color: #FFFFFF !important;
        font-family: 'Barlow', sans-serif !important;
    }
    * { font-family: 'Barlow', sans-serif !important; }

    /* Hide chrome but keep sidebar toggle */
    #MainMenu, footer { visibility: hidden; }
    [data-testid="stToolbar"] { display: none !important; }

    /* Sidebar toggle button */
    [data-testid="stSidebarCollapsedControl"] {
        visibility: visible !important;
        display: block !important;
        color: #FFFFFF !important;
    }
    [data-testid="stSidebarCollapsedControl"] button {
        color: #FFFFFF !important;
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 0 !important;
    }
    [data-testid="stSidebarCollapsedControl"] svg {
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }

    /* ── Typography ── */
    h1 {
        font-family: 'Barlow Condensed', sans-serif !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        font-size: 3rem !important;
        line-height: 1.0 !important;
    }
    h2 {
        font-family: 'Barlow Condensed', sans-serif !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        font-size: 1.4rem !important;
        color: rgba(255,255,255,0.5) !important;
    }
    h3 {
        font-family: 'Barlow', sans-serif !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
        font-size: 0.7rem !important;
        color: rgba(255,255,255,0.35) !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: transparent !important;
        border: none !important;
        border-top: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 0 !important;
        padding: 20px 0 !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Barlow', sans-serif !important;
        color: rgba(255,255,255,0.4) !important;
        font-size: 0.7rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Barlow', sans-serif !important;
        color: #FFFFFF !important;
        font-size: 1.6rem !important;
        font-weight: 300 !important;
        letter-spacing: 0.02em !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background: transparent !important;
        border-bottom: 1px solid rgba(255,255,255,0.12);
        border-radius: 0 !important;
        padding: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 0 !important;
        padding: 12px 32px !important;
        color: rgba(255,255,255,0.35) !important;
        font-family: 'Barlow', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.14em !important;
        background: transparent !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
        background: transparent !important;
        border-bottom: 2px solid #FFFFFF !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background: #FFFFFF !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {
        background: #FFFFFF !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 0 !important;
        font-family: 'Barlow', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.14em !important;
        padding: 12px 40px !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button[kind="primary"]:hover {
        opacity: 0.85 !important;
        background: #FFFFFF !important;
        color: #000000 !important;
    }
    .stButton > button {
        background: transparent !important;
        color: rgba(255,255,255,0.6) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 0 !important;
        font-family: 'Barlow', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.7rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        padding: 8px 20px !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #000000 !important;
        border-right: 1px solid rgba(255,255,255,0.08) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h3 {
        margin-top: 28px !important;
    }

    /* ── Inputs ── */
    [data-testid="stNumberInput"] input,
    [data-baseweb="input"] input {
        background: transparent !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 0 !important;
        color: #FFFFFF !important;
        font-family: 'Barlow', sans-serif !important;
        font-weight: 400 !important;
    }
    [data-baseweb="select"] > div {
        background: transparent !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 0 !important;
    }
    label {
        font-family: 'Barlow', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.72rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        color: rgba(255,255,255,0.4) !important;
    }

    /* ── Slider ── */
    [data-baseweb="slider"] [role="slider"] {
        background: #FFFFFF !important;
        border: none !important;
    }
    [data-baseweb="slider"] [data-testid="stTickBar"] {
        background: rgba(255,255,255,0.12) !important;
    }

    /* ── Progress ── */
    .stProgress > div > div {
        background: #FFFFFF !important;
        border-radius: 0 !important;
    }
    .stProgress > div {
        background: rgba(255,255,255,0.08) !important;
        border-radius: 0 !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: transparent !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 0 !important;
    }

    /* ── Code blocks ── */
    .stCodeBlock {
        border-radius: 0 !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }

    /* ── Alerts ── */
    .stAlert {
        border-radius: 0 !important;
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
    }

    /* ── Dividers ── */
    hr {
        border-color: rgba(255,255,255,0.08) !important;
    }

    /* ── Stat row (custom) ── */
    .spec-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        padding: 18px 0;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .spec-label {
        font-family: 'Barlow', sans-serif;
        font-weight: 500;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: rgba(255,255,255,0.45);
    }
    .spec-value {
        font-family: 'Barlow', sans-serif;
        font-weight: 300;
        font-size: 1rem;
        color: #FFFFFF;
        letter-spacing: 0.02em;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Config ──────────────────────────────────────────────
config = load_config()


def spec_row(label, value):
    """Render a SpaceX-style spec row: LABEL ..... value"""
    return (f'<div class="spec-row">'
            f'<span class="spec-label">{label}</span>'
            f'<span class="spec-value">{value}</span>'
            f'</div>')


# ── Hero Section ─────────────────────────────────────────────
total_mass = config["vehicle"]["dry_mass"] + config["vehicle"]["landing_propellant"]

hero_bg = ""
if _HERO_B64:
    hero_bg = (
        f"background-image: "
        f"linear-gradient(90deg, rgba(0,0,0,0.92) 0%, rgba(0,0,0,0.7) 50%, rgba(0,0,0,0.2) 100%), "
        f"url('data:image/jpeg;base64,{_HERO_B64}');"
        f"background-size: cover;"
        f"background-position: right center;"
    )

_hero_style = (
    f"{hero_bg} margin:-1rem -1rem 0 -1rem; padding:80px 60px 60px 60px; "
    f"min-height:520px; display:flex; align-items:flex-end;"
)
_lbl = "font-size:0.65rem; text-transform:uppercase; letter-spacing:0.14em; color:rgba(255,255,255,0.35); margin-bottom:4px;"
_val = "font-size:1.1rem; font-weight:300; color:#FFF;"
_cell_t = "border-top:1px solid rgba(255,255,255,0.12); padding:14px 24px 14px 0;"
_cell_tr = "border-top:1px solid rgba(255,255,255,0.12); padding:14px 0 14px 24px;"
_cell_tb = "border-top:1px solid rgba(255,255,255,0.12); border-bottom:1px solid rgba(255,255,255,0.12); padding:14px 24px 14px 0;"
_cell_tbr = "border-top:1px solid rgba(255,255,255,0.12); border-bottom:1px solid rgba(255,255,255,0.12); padding:14px 0 14px 24px;"

_height = f'{config["vehicle"]["height"]:.0f}'
_diam = f'{config["vehicle"]["diameter"]:.0f}'
_mass_t = f'{total_mass/1000:.0f}'
_thrust_mn = f'{config["thrust"]["simple"]["max_thrust"]/1e6:.1f}'
_n_legs = f'{len(config["vehicle"]["legs"])}'

st.markdown(
    f'<div style="{_hero_style}">'
    f'<div style="max-width:560px;">'
    f'<p style="font-family:Barlow,sans-serif; font-weight:500; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.2em; color:rgba(255,255,255,0.4); margin-bottom:12px;">Landing Dynamics Simulation</p>'
    f'<h1 style="font-family:Barlow Condensed,sans-serif; font-weight:700; font-size:4.5rem; text-transform:uppercase; letter-spacing:0.02em; line-height:0.95; margin:0 0 32px 0; color:#FFFFFF;">STARSHIP</h1>'
    f'<p style="color:rgba(255,255,255,0.55); font-size:0.9rem; font-weight:300; line-height:1.75; margin-bottom:40px; max-width:440px;">3-DOF propulsive landing with nonlinear contact dynamics and Monte Carlo uncertainty quantification. Vehicle parameters approximate Starship from publicly available data.</p>'
    f'<div style="display:grid; grid-template-columns:1fr 1fr; gap:0;">'
    f'<div style="{_cell_t}"><div style="{_lbl}">Height</div><div style="{_val}">{_height} m</div></div>'
    f'<div style="{_cell_tr}"><div style="{_lbl}">Diameter</div><div style="{_val}">{_diam} m</div></div>'
    f'<div style="{_cell_t}"><div style="{_lbl}">Mass</div><div style="{_val}">{_mass_t} t</div></div>'
    f'<div style="{_cell_tr}"><div style="{_lbl}">Thrust</div><div style="{_val}">{_thrust_mn} MN</div></div>'
    f'<div style="{_cell_tb}"><div style="{_lbl}">Landing Legs</div><div style="{_val}">{_n_legs}</div></div>'
    f'<div style="{_cell_tbr}"><div style="{_lbl}">Degrees of Freedom</div><div style="{_val}">3</div></div>'
    f'</div></div></div>',
    unsafe_allow_html=True,
)

st.markdown("")
st.markdown("")


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="padding:20px 0 8px 0;">'
        '<span style="font-family:Barlow Condensed,sans-serif; font-size:1.1rem; '
        'font-weight:600; text-transform:uppercase; letter-spacing:0.12em; '
        'color:rgba(255,255,255,0.5);">Configuration</span>'
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
                               format_func=lambda x: "Single Engine" if x == "simple"
                               else "Multi-Engine")
    config["thrust"]["mode"] = thrust_mode
    max_thrust = st.number_input("Max Thrust / Engine (MN)",
                                 value=config["thrust"]["simple"]["max_thrust"] / 1e6,
                                 step=0.1, format="%.1f")
    config["thrust"]["simple"]["max_thrust"] = max_thrust * 1e6
    config["thrust"]["multi_engine"]["thrust_per_engine"] = max_thrust * 1e6

    st.markdown("### Contact")
    stiffness_exp = st.slider("Stiffness (10^x N/m^n)", 6.0, 10.0,
                              np.log10(config["contact"]["stiffness"]), 0.1)
    config["contact"]["stiffness"] = 10 ** stiffness_exp
    damping = st.number_input("Damping (N*s/m)", value=config["contact"]["damping"],
                              step=50000.0, format="%.0f")
    config["contact"]["damping"] = damping

    st.markdown("### Initial State")
    ic_alt = st.number_input("Altitude (m)",
                             value=config["initial_conditions"]["altitude"], step=10.0)
    ic_vz = st.number_input("Descent Rate (m/s)",
                            value=config["initial_conditions"]["vertical_velocity"], step=1.0)
    config["initial_conditions"]["altitude"] = ic_alt
    config["initial_conditions"]["vertical_velocity"] = ic_vz


# ── Tabs ─────────────────────────────────────────────────────
tab_single, tab_mc, tab_ai = st.tabs(["Single Case", "Monte Carlo", "AI Analyst"])


# ==============================================================
# TAB 1: SINGLE CASE
# ==============================================================
with tab_single:
    st.markdown("")
    st.markdown("## Touchdown Dispersions")
    st.markdown("")

    c1, c2, c3 = st.columns([1, 1, 1])
    pitch_deg = c1.number_input("Pitch Angle (deg)", value=0.0, step=0.5)
    lat_vel = c2.number_input("Lateral Velocity (m/s)", value=0.0, step=0.1)
    c3.markdown("")
    c3.markdown("")
    run_single = c3.button("Execute", type="primary", use_container_width=True)

    if run_single:
        ic_over = {
            "pitch_angle": np.radians(pitch_deg),
            "lateral_velocity": lat_vel,
        }
        with st.spinner(""):
            result = run_simulation(config, ic_overrides=ic_over)

        st.markdown("")

        # Results header
        if result.settled:
            st.markdown(
                '<p style="color:rgba(255,255,255,0.35); font-size:0.7rem; '
                'text-transform:uppercase; letter-spacing:0.14em;">'
                'Nominal Touchdown &mdash; Vehicle Settled</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p style="color:rgba(255,255,255,0.35); font-size:0.7rem; '
                'text-transform:uppercase; letter-spacing:0.14em;">'
                'Simulation Complete</p>',
                unsafe_allow_html=True,
            )

        # Metrics
        if result.touchdown_idx is not None and "axial" in result.total_forces:
            td_vz = result.z_dot[result.touchdown_idx]
            pk_ax = np.max(result.total_forces["axial"])
            pk_lat = np.max(np.abs(result.total_forces["lateral"]))
            pk_mom = np.max(np.abs(result.total_forces["moment"]))

            metrics_html = '<div style="margin:8px 0 40px 0;">'
            for label, val in [
                ("Touchdown Velocity", f"{td_vz:.2f} m/s"),
                ("Peak Axial Force", f"{pk_ax/1e6:.2f} MN"),
                ("Peak Lateral Force", f"{pk_lat/1e6:.3f} MN"),
                ("Peak Overturning Moment", f"{pk_mom/1e6:.2f} MN-m"),
            ]:
                metrics_html += spec_row(label, val)
            metrics_html += '</div>'
            st.markdown(metrics_html, unsafe_allow_html=True)

        # Charts
        st.markdown("## Time History")
        st.markdown("")

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=("Altitude", "Vertical Velocity", "Contact Forces",
                            "Lateral Position", "Lateral Velocity", "Pitch Angle"),
            vertical_spacing=0.14, horizontal_spacing=0.07,
        )

        trace_data = [
            (result.t, result.z, 1, 1),
            (result.t, result.z_dot, 1, 2),
            (result.t, result.x, 2, 1),
            (result.t, result.x_dot, 2, 2),
            (result.t, np.degrees(result.theta), 2, 3),
        ]
        for x, y, row, col in trace_data:
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines", showlegend=False,
                line=dict(color="#FFFFFF", width=1.5),
            ), row=row, col=col)

        if "axial" in result.total_forces:
            fig.add_trace(go.Scatter(
                x=result.t, y=result.total_forces["axial"] / 1e6,
                name="Axial", mode="lines",
                line=dict(color="#FFFFFF", width=1.5),
                fill="tozeroy", fillcolor="rgba(255,255,255,0.06)"),
                row=1, col=3)
            fig.add_trace(go.Scatter(
                x=result.t, y=np.abs(result.total_forces["lateral"]) / 1e6,
                name="Lateral", mode="lines",
                line=dict(color="rgba(255,255,255,0.4)", width=1.5)),
                row=1, col=3)

        style_fig(fig, height=520)
        fig.update_annotations(font=dict(
            color="rgba(255,255,255,0.35)", size=11,
            family="Barlow, sans-serif",
        ))
        st.plotly_chart(fig, use_container_width=True)


# ==============================================================
# TAB 2: MONTE CARLO
# ==============================================================
with tab_mc:
    st.markdown("")
    st.markdown("## Campaign")
    st.markdown("")

    c1, c2, c3 = st.columns([1, 1, 1])
    n_cases = c1.number_input("Number of Cases", value=100, min_value=10,
                              max_value=10000, step=50)
    n_workers = c2.number_input("Parallel Workers", value=1, min_value=1, max_value=16)
    c3.markdown("")
    c3.markdown("")
    run_mc = c3.button("Launch", type="primary", use_container_width=True,
                       key="mc_launch")

    if run_mc:
        progress = st.progress(0, text="")

        def update_progress(done, total):
            progress.progress(done / total,
                              text=f"Case {done:,} / {total:,}")

        mc_results = run_monte_carlo(config, n_cases=n_cases,
                                     n_workers=n_workers,
                                     progress_callback=update_progress)
        progress.empty()

        st.session_state["mc_results"] = mc_results
        stats = compute_statistics(mc_results)
        st.session_state["mc_stats"] = stats

        # Summary
        st.markdown("")
        st.markdown("## Results")

        summary_html = '<div style="margin:8px 0 40px 0;">'
        for label, val in [
            ("Cases", f"{n_cases:,}"),
            ("Mean Peak Axial", f'{stats["peak_axial_force"].mean/1e6:.2f} MN'),
            ("P95 Peak Axial", f'{stats["peak_axial_force"].p95/1e6:.2f} MN'),
            ("P99 Peak Axial", f'{stats["peak_axial_force"].p99/1e6:.2f} MN'),
            ("Max Peak Axial", f'{stats["peak_axial_force"].max_val/1e6:.2f} MN'),
            ("Mean Touchdown Velocity", f'{stats["touchdown_vz"].mean:.2f} m/s'),
        ]:
            summary_html += spec_row(label, val)
        summary_html += '</div>'
        st.markdown(summary_html, unsafe_allow_html=True)

        with st.expander("Full Statistics"):
            st.code(format_statistics_table(stats))

        # Histograms
        st.markdown("## Distributions")
        st.markdown("")

        fig_hist = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Peak Axial Force (MN)",
                            "Peak Lateral Force (MN)",
                            "Peak Moment (MN-m)"),
        )
        for i, data in enumerate([
            mc_results.peak_axial / 1e6,
            mc_results.peak_lateral / 1e6,
            mc_results.peak_moment / 1e6,
        ]):
            fig_hist.add_trace(go.Histogram(
                x=data, nbinsx=40, showlegend=False,
                marker=dict(color="rgba(255,255,255,0.25)",
                            line=dict(color="rgba(255,255,255,0.08)", width=0.5)),
            ), row=1, col=i + 1)
            fig_hist.add_vline(x=np.mean(data), line_dash="dash",
                               line_color="rgba(255,255,255,0.6)", line_width=1,
                               row=1, col=i + 1)
            fig_hist.add_vline(x=np.percentile(data, 95), line_dash="dot",
                               line_color="rgba(255,255,255,0.3)", line_width=1,
                               row=1, col=i + 1)

        style_fig(fig_hist, height=340)
        fig_hist.update_annotations(font=dict(
            color="rgba(255,255,255,0.35)", size=11,
            family="Barlow, sans-serif",
        ))
        st.plotly_chart(fig_hist, use_container_width=True)

        # Envelope + Scatter
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("## Load Envelope")
            envelope = compute_load_envelope(
                mc_results.peak_axial, mc_results.peak_lateral)

            fig_env = go.Figure()
            fig_env.add_trace(go.Scatter(
                x=envelope.x_values / 1e6, y=envelope.y_values / 1e6,
                mode="markers", name="Cases",
                marker=dict(size=3.5, color="rgba(255,255,255,0.3)"),
            ))
            if envelope.hull_vertices is not None:
                hull = np.vstack([envelope.hull_vertices, envelope.hull_vertices[0]])
                fig_env.add_trace(go.Scatter(
                    x=hull[:, 0] / 1e6, y=hull[:, 1] / 1e6,
                    mode="lines", name="Envelope",
                    line=dict(color="#FFFFFF", width=1.5),
                ))
            fig_env.update_layout(
                xaxis_title="Peak Axial (MN)",
                yaxis_title="Peak Lateral (MN)",
            )
            style_fig(fig_env, height=420)
            st.plotly_chart(fig_env, use_container_width=True)

        with col_r:
            st.markdown("## Velocity vs Load")
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=mc_results.touchdown_vz,
                y=mc_results.peak_axial / 1e6,
                mode="markers", name="Cases",
                marker=dict(
                    size=4, color=np.degrees(mc_results.touchdown_theta),
                    colorscale=[[0, "rgba(255,255,255,0.2)"],
                                [1, "#FFFFFF"]],
                    colorbar=dict(
                        title=dict(text="Pitch (deg)",
                                   font=dict(color="rgba(255,255,255,0.35)",
                                             size=10)),
                        tickfont=dict(color="rgba(255,255,255,0.35)", size=10),
                    ),
                    opacity=0.7,
                ),
            ))
            fig_sc.update_layout(
                xaxis_title="Touchdown Vz (m/s)",
                yaxis_title="Peak Axial (MN)",
            )
            style_fig(fig_sc, height=420)
            st.plotly_chart(fig_sc, use_container_width=True)


# ==============================================================
# TAB 3: AI ANALYST
# ==============================================================
with tab_ai:
    st.markdown("")

    if not ai_analyst.is_available():
        st.markdown("")
        st.markdown("## AI Analysis")
        st.markdown("")
        st.markdown(
            '<p style="color:rgba(255,255,255,0.4); font-weight:300;">'
            'Set <code style="color:rgba(255,255,255,0.6);">OPENAI_API_KEY</code> '
            'as an environment variable to enable GPT-4o powered analysis.</p>',
            unsafe_allow_html=True,
        )
    elif "mc_stats" not in st.session_state:
        st.markdown("")
        st.markdown("## AI Analysis")
        st.markdown("")
        st.markdown(
            '<p style="color:rgba(255,255,255,0.4); font-weight:300;">'
            'Run a Monte Carlo campaign first, then return here to query the analyst.</p>',
            unsafe_allow_html=True,
        )
    else:
        stats = st.session_state["mc_stats"]
        mc_results = st.session_state["mc_results"]
        settled_pct = np.mean(mc_results.settled_flags) * 100
        context = ai_analyst.serialize_results(stats, mc_results.n_cases, settled_pct)

        st.markdown("")
        st.markdown("## AI Analysis")
        st.markdown("")

        suggested = ai_analyst.get_default_questions()
        cols = st.columns(3)
        for i, q in enumerate(suggested[:6]):
            if cols[i % 3].button(q, key=f"s_{i}", use_container_width=True):
                st.session_state["ai_question"] = q

        st.markdown("")
        question = st.text_input(
            "Query",
            value=st.session_state.get("ai_question", ""),
            placeholder="Which parameters drive the peak axial force?",
        )

        if question:
            if st.button("Analyze", type="primary"):
                with st.spinner(""):
                    response = ai_analyst.query(question, context)
                st.markdown("---")
                st.markdown(response)
