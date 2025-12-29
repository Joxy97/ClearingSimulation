# Clearing Simulation

This project is a cleaned-up Python package derived from the original notebook. The main pieces are split into focused modules with small CLI scripts to generate data and run the day-by-day clearing simulation.

## Layout

- `clearing/` core library (market model, quantization, sampler, trading, margin, QUBO, logging, simulation)
- `scripts/` small CLI entry points
- `models/` trained RBM runs
- `data/` generated datasets and quantizers

## Install

```bash
pip install -r requirements.txt
```

Optional: install the Leap hybrid solver if you want `--qubo-solver hybrid`:

```bash
pip install dwave-system
```

## Generate dataset

```bash
python scripts/generate_data.py --T 10000 --out-csv data/data.csv --out-quantizer data/quantizer.pt
```

## Run simulation

```bash
python scripts/run_simulation.py --model-run models/model_mini --quantizer data/quantizer.pt
```

If the quantizer file is missing, the simulation script will fit a new one from a fresh synthetic market run. For best fidelity, keep the quantizer used during model training.

## Streamlit dashboard

```bash
streamlit run streamlit_app.py
```

Use the sidebar to point to `simulations/logs/run_log.pt`.
