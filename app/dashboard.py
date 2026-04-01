"""Streamlit dashboard for interactive landing dynamics simulation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml

from landing_sim.simulation import run_simulation, load_config
from landing_sim.monte_carlo import run_monte_carlo
from landing_sim.loads import compute_statistics, compute_load_envelope, format_statistics_table
from landing_sim import ai_analyst


st.set_page_config(page_title="Landing Dynamics Sim", page_icon="🚀", layout="wide")
st.title("Reusable Launch Vehicle Landing Dynamics")
st.caption("3-DOF Propulsive Landing Simulation with Monte Carlo Analysis")

# --- Sidebar: Parameters ---
st.sidebar.header("Vehicle Parameters")
config = load_config()

mass = st.sidebar.number_input("Dry Mass (kg)", value=config["vehicle"]["dry_mass"],
                                step=1000.0, format="%.0f")
prop = st.sidebar.number_input("Landing Propellant (kg)",
                                value=config["vehicle"]["landing_propellant"],
                                step=100.0, format="%.0f")
config["vehicle"]["dry_mass"] = mass
config["vehicle"]["landing_propellant"] = prop

st.sidebar.header("Thrust")
thrust_mode = st.sidebar.selectbox("Thrust Mode", ["simple", "multi_engine"])
config["thrust"]["mode"] = thrust_mode

max_thrust = st.sidebar.number_input("Max Thrust per Engine (MN)",
                                      value=config["thrust"]["simple"]["max_thrust"] / 1e6,
                                      step=0.1, format="%.1f")
config["thrust"]["simple"]["max_thrust"] = max_thrust * 1e6
config["thrust"]["multi_engine"]["thrust_per_engine"] = max_thrust * 1e6

st.sidebar.header("Contact Model")
stiffness_exp = st.sidebar.slider("Ground Stiffness (10^x N/m^n)", 6.0, 10.0, 8.7, 0.1)
config["contact"]["stiffness"] = 10 ** stiffness_exp
config["contact"]["damping"] = st.sidebar.number_input(
    "Damping (N*s/m)", value=config["contact"]["damping"], step=1e5, format="%.0f")

st.sidebar.header("Initial Conditions")
ic_alt = st.sidebar.number_input("Altitude (m)", value=config["initial_conditions"]["altitude"],
                                  step=50.0)
ic_vz = st.sidebar.number_input("Vertical Velocity (m/s)",
                                 value=config["initial_conditions"]["vertical_velocity"],
                                 step=1.0)
config["initial_conditions"]["altitude"] = ic_alt
config["initial_conditions"]["vertical_velocity"] = ic_vz

# --- Main Area: Tabs ---
tab_single, tab_mc, tab_ai = st.tabs(["Single Case", "Monte Carlo", "AI Analyst"])

# ============================================================
# TAB 1: Single Case
# ============================================================
with tab_single:
    col1, col2 = st.columns([1, 3])
    with col1:
        pitch_deg = st.number_input("Pitch (deg)", value=0.0, step=0.5)
        lat_vel = st.number_input("Lateral Vel (m/s)", value=0.0, step=0.1)
        run_single = st.button("Run Single Case", type="primary")

    if run_single:
        ic_over = {
            "pitch_angle": np.radians(pitch_deg),
            "lateral_velocity": lat_vel,
        }
        with st.spinner("Running simulation..."):
            result = run_simulation(config, ic_overrides=ic_over)

        with col2:
            st.success(f"Simulation complete — {'Settled' if result.settled else 'Did not settle'}")

        # Time history plots with Plotly
        fig = make_subplots(rows=3, cols=2,
                            subplot_titles=("Altitude", "Vertical Velocity",
                                          "Lateral Position", "Lateral Velocity",
                                          "Pitch Angle", "Contact Forces"))

        fig.add_trace(go.Scatter(x=result.t, y=result.z, name="Altitude",
                                 line=dict(color="#1a5276")), row=1, col=1)
        fig.add_trace(go.Scatter(x=result.t, y=result.z_dot, name="Vz",
                                 line=dict(color="#1a5276")), row=1, col=2)
        fig.add_trace(go.Scatter(x=result.t, y=result.x, name="x",
                                 line=dict(color="#c0392b")), row=2, col=1)
        fig.add_trace(go.Scatter(x=result.t, y=result.x_dot, name="Vx",
                                 line=dict(color="#c0392b")), row=2, col=2)
        fig.add_trace(go.Scatter(x=result.t, y=np.degrees(result.theta), name="Pitch",
                                 line=dict(color="#27ae60")), row=3, col=1)

        if "axial" in result.total_forces:
            fig.add_trace(go.Scatter(x=result.t, y=result.total_forces["axial"] / 1e6,
                                     name="Axial (MN)", line=dict(color="#1a5276")),
                         row=3, col=2)
            fig.add_trace(go.Scatter(x=result.t,
                                     y=np.abs(result.total_forces["lateral"]) / 1e6,
                                     name="|Lateral| (MN)", line=dict(color="#c0392b")),
                         row=3, col=2)

        fig.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Peak loads summary
        if "axial" in result.total_forces:
            c1, c2, c3 = st.columns(3)
            c1.metric("Peak Axial Force",
                      f"{np.max(result.total_forces['axial']) / 1e6:.2f} MN")
            c2.metric("Peak Lateral Force",
                      f"{np.max(np.abs(result.total_forces['lateral'])) / 1e6:.2f} MN")
            c3.metric("Peak Moment",
                      f"{np.max(np.abs(result.total_forces['moment'])) / 1e6:.2f} MN-m")

# ============================================================
# TAB 2: Monte Carlo
# ============================================================
with tab_mc:
    col1, col2 = st.columns([1, 3])
    with col1:
        n_cases = st.number_input("Number of Cases", value=100, min_value=10,
                                   max_value=10000, step=100)
        n_workers = st.number_input("Workers", value=1, min_value=1, max_value=16)
        run_mc = st.button("Run Monte Carlo", type="primary")

    if run_mc:
        progress = st.progress(0, text="Running Monte Carlo...")

        def update_progress(done, total):
            progress.progress(done / total, text=f"Case {done}/{total}")

        with st.spinner(""):
            mc_results = run_monte_carlo(config, n_cases=n_cases,
                                         n_workers=n_workers,
                                         progress_callback=update_progress)

        progress.empty()
        st.session_state["mc_results"] = mc_results

        stats = compute_statistics(mc_results)
        st.session_state["mc_stats"] = stats

        settled_pct = np.mean(mc_results.settled_flags) * 100
        st.success(f"Completed {n_cases} cases — {settled_pct:.1f}% settled successfully")

        # Statistics table
        st.subheader("Load Statistics")
        st.code(format_statistics_table(stats))

        # Histograms
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=("Peak Axial Force (MN)",
                                          "Peak Lateral Force (MN)",
                                          "Peak Moment (MN-m)"))
        fig.add_trace(go.Histogram(x=mc_results.peak_axial / 1e6, nbinsx=50,
                                    marker_color="#1a5276"), row=1, col=1)
        fig.add_trace(go.Histogram(x=mc_results.peak_lateral / 1e6, nbinsx=50,
                                    marker_color="#c0392b"), row=1, col=2)
        fig.add_trace(go.Histogram(x=mc_results.peak_moment / 1e6, nbinsx=50,
                                    marker_color="#27ae60"), row=1, col=3)
        fig.update_layout(height=400, showlegend=False,
                         title_text="Peak Load Distributions")
        st.plotly_chart(fig, use_container_width=True)

        # Load envelope
        envelope = compute_load_envelope(mc_results.peak_axial, mc_results.peak_lateral,
                                         "Peak Axial Force", "Peak Lateral Force")

        fig_env = go.Figure()
        fig_env.add_trace(go.Scatter(
            x=envelope.x_values / 1e6, y=envelope.y_values / 1e6,
            mode="markers", marker=dict(size=3, color="#1a5276", opacity=0.3),
            name="Cases"))
        if envelope.hull_vertices is not None:
            hull = np.vstack([envelope.hull_vertices, envelope.hull_vertices[0]])
            fig_env.add_trace(go.Scatter(
                x=hull[:, 0] / 1e6, y=hull[:, 1] / 1e6,
                mode="lines", line=dict(color="#c0392b", width=2),
                name="Convex Hull"))
        fig_env.update_layout(
            title="Load Envelope",
            xaxis_title="Peak Axial Force (MN)",
            yaxis_title="Peak Lateral Force (MN)",
            height=500,
        )
        st.plotly_chart(fig_env, use_container_width=True)

        # Scatter: Vz vs Peak Axial colored by pitch
        fig_scatter = px.scatter(
            x=mc_results.touchdown_vz,
            y=mc_results.peak_axial / 1e6,
            color=np.degrees(mc_results.touchdown_theta),
            labels={"x": "Touchdown Vz (m/s)", "y": "Peak Axial (MN)",
                    "color": "Pitch (deg)"},
            title="Touchdown Velocity vs Peak Load",
            color_continuous_scale="RdYlBu_r",
        )
        fig_scatter.update_traces(marker=dict(size=4))
        st.plotly_chart(fig_scatter, use_container_width=True)

# ============================================================
# TAB 3: AI Analyst
# ============================================================
with tab_ai:
    if not ai_analyst.is_available():
        st.warning("Set `OPENAI_API_KEY` environment variable to enable AI analysis.")
    else:
        st.subheader("AI-Powered Results Analysis")
        st.caption("Ask questions about your Monte Carlo results using GPT-4o")

        if "mc_stats" not in st.session_state:
            st.info("Run a Monte Carlo analysis first, then come here to analyze results.")
        else:
            stats = st.session_state["mc_stats"]
            mc_results = st.session_state["mc_results"]

            settled_pct = np.mean(mc_results.settled_flags) * 100
            context = ai_analyst.serialize_results(stats, mc_results.n_cases, settled_pct)

            # Suggested questions
            st.write("**Suggested questions:**")
            suggested = ai_analyst.get_default_questions()
            cols = st.columns(2)
            for i, q in enumerate(suggested):
                if cols[i % 2].button(q, key=f"suggest_{i}"):
                    st.session_state["ai_question"] = q

            # Chat input
            question = st.text_input("Or ask your own question:",
                                      value=st.session_state.get("ai_question", ""))

            if question and st.button("Analyze", type="primary"):
                with st.spinner("Consulting AI analyst..."):
                    response = ai_analyst.query(question, context)
                st.markdown("---")
                st.markdown(response)

            # Chat history
            if "ai_history" not in st.session_state:
                st.session_state["ai_history"] = []
