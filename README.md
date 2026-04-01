# Reusable Launch Vehicle Landing Dynamics Simulation

3-DOF propulsive landing simulation in Python/SciPy for a Starship-class reusable launch vehicle. Models vertical, lateral, and pitch dynamics with nonlinear spring-damper ground contact, thrust vector control, and configurable multi-engine thrust profiles. Runs 5,000 Monte Carlo touchdown dispersions and extracts peak interface loads with statistical analysis.

## Key Results

- **3-DOF dynamics**: Vertical (z), lateral (x), and pitch (theta) with coupled thrust, gravity, contact, and aerodynamic forces
- **Hoverslam guidance**: Constant-deceleration thrust profile with TVC (thrust vector control) for attitude stabilization
- **Nonlinear contact**: Kelvin-Voigt spring-damper model (F = k*delta^n + c*delta_dot) at each landing leg
- **Monte Carlo**: 5,000 cases via Latin Hypercube Sampling across 9 uncertainty parameters
- **Validation**: Peak loads within ~5% of analytical impulse-momentum estimates

## Physics Model

### State Vector
```
[z, z_dot, x, x_dot, theta, theta_dot]
```
- `z` - CG altitude (m)
- `x` - lateral position (m)
- `theta` - pitch angle from vertical (rad)

### Forces & Moments
- **Gravity**: -mg (constant)
- **Thrust**: Along body axis with TVC gimbal (PD controller, max 7 deg)
- **Contact**: Per-leg nonlinear spring-damper with Coulomb friction
- **Wind**: Constant lateral gust force (dispersion parameter)

### Monte Carlo Dispersions (9 parameters)
| Parameter | Nominal | 3-sigma |
|---|---|---|
| Vertical velocity | -25 m/s | +/- 9 m/s |
| Lateral velocity | 0 m/s | +/- 1.5 m/s |
| Pitch angle | 0 rad | +/- 1 deg |
| Pitch rate | 0 rad/s | +/- 0.5 deg/s |
| Thrust scale | 1.0 | +/- 5% |
| Mass scale | 1.0 | +/- 3% |
| CG offset | 0 m | +/- 0.1 m |
| Wind force | 0 N | +/- 10 kN |
| Leg stiffness scale | 1.0 | +/- 10% |

## Project Structure

```
launch-dynamics/
├── config/default.yaml         # Vehicle & simulation parameters
├── src/landing_sim/
│   ├── vehicle.py              # Vehicle model, mass properties, leg geometry
│   ├── thrust.py               # Hoverslam guidance + multi-engine profiles
│   ├── dynamics.py             # 3-DOF equations of motion with TVC
│   ├── contact.py              # Nonlinear spring-damper contact model
│   ├── simulation.py           # ODE integration (scipy.integrate.solve_ivp)
│   ├── monte_carlo.py          # LHS sampling + parallel batch execution
│   ├── loads.py                # Interface load extraction & statistics
│   ├── plotting.py             # Publication-quality matplotlib plots
│   ├── animation.py            # 2D landing animation (FuncAnimation)
│   └── ai_analyst.py           # OpenAI-powered results analysis
├── app/dashboard.py            # Streamlit interactive dashboard
├── tests/                      # pytest unit tests (23 tests)
└── results/                    # Generated outputs (gitignored)
```

## Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Single simulation
python -c "
from landing_sim.simulation import run_simulation, load_config
result = run_simulation(load_config())
print(f'Touchdown Vz: {result.z_dot[result.touchdown_idx]:.2f} m/s')
"

# Monte Carlo (100 cases)
python -c "
from landing_sim.simulation import load_config
from landing_sim.monte_carlo import run_monte_carlo
from landing_sim.loads import compute_statistics, format_statistics_table
mc = run_monte_carlo(load_config(), n_cases=100)
print(format_statistics_table(compute_statistics(mc)))
"

# Launch dashboard
streamlit run app/dashboard.py
```

## Vehicle Parameters (Starship-Approximate)

Based on publicly available data:

| Parameter | Value |
|---|---|
| Dry mass | 100,000 kg |
| Landing propellant | 5,000 kg |
| Height | 50 m |
| Diameter | 9 m |
| Landing legs | 6 (planar projection) |
| Max thrust (landing) | 2.2 MN (1 Raptor) |
| Pitch inertia | 20,000,000 kg-m^2 |

## Dependencies

numpy, scipy, matplotlib, pyyaml, streamlit, plotly, openai, python-dotenv

## Dashboard

The Streamlit dashboard provides:
- Interactive parameter tuning with live simulation
- Single-case time history plots (altitude, velocity, pitch, contact forces)
- Monte Carlo batch execution with progress tracking
- Load envelope and statistical histogram visualization
- AI-powered results analysis via OpenAI API (optional)

Set `OPENAI_API_KEY` in a `.env` file to enable the AI analyst panel.
