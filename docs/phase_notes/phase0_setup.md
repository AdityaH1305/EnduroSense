# Phase 0: Setup and foundations

**Status:** complete (2026-10-01)

## What was done

- **Package.** Created the `src/endurosense` package, installed in editable mode (`pip install -e .`).
  - `config.py`: reads `config.yaml` and provides `data_path()` and `set_seed()` (Python, NumPy and PyTorch).
  - `data/load.py`: reads the raw CSVs (typed loading and a parquet cache come in Phase 1).
  - `data/energy.py`: `energy_wh()`, which integrates V·I·dt.
  - `data/wind.py`: `ambient_wind()`, `ground_speed()` and `height_above_takeoff()`.
  - `data/chains.py`: `assign_battery_chains()`.
  - `data/profile.py`: `summarise_flight()` and `build_flight_summary()`.
- **Refactor.** Moved `summarise_flight()` and `assign_battery_chains()` out of `scripts/01_profile_data.py` into the package. The script is now a thin wrapper around it.
- **Settings.** `config.yaml` holds every tunable setting, including the defaults waiting on guide review (reserve 22.6 V, τ = 0.95). Thresholds that used to be hard-coded (the wind-sampling limits, the chain tolerance and the battery capacity) now live there too.
- **Environment.** `requirements.txt` is pinned to the installed versions. Streamlit 1.64.0 was added; a dry run confirmed it didn't change any existing package.
- **Project files.** Added `pyproject.toml`, `.gitignore` (raw data, features and models are not versioned) and `README.md`.
- **Docs.** `docs/IMPLEMENTATION_PLAN.md` is a copy of the approved phase plan; `docs/literature.md` holds the starter literature notes.

## Verification

- `pytest`: **13 passed**.
  - Unit tests cover config defaults, seeding, energy integration and chain-linking rules.
  - Data-regression tests reproduce the profiling numbers: 209 flights, 196 cruise flights, 60/60 grid cells with 3–5 flights each, 103 chains, 62 multi-flight chains, 25 near-full-to-reserve chains, mean energy 21.23 Wh, and wind from 1.29 to 6.71 m/s.
- `python scripts/01_profile_data.py` writes a `data/processed/flight_summary.csv` that is **identical** to the file produced before the refactor (checked with `pandas.testing.assert_frame_equal`).

> **Correction (2026-10-02 verification pass):** the chain numbers above (103 / 62 / 25) came from flights sorted by start time *as text*, which mis-orders 7 days. With true time order and 4 corrected start times, the Phase 0 profile gives **92 chains, 64 multi-flight and 28 full-to-reserve**; the regression test was updated. Phase 1's motors-off method (90 / 71 / 60 near reserve) is what the project uses.

## Carried into Phase 1

- Typed loading and a parquet cache, and parsing the mixed `altitude` column (e.g. `"25-50-100-25"`).
- The chain containing flight 81 (chain 19) looks suspicious and needs investigating.
- End-of-flight voltage hasn't fully recovered after landing, so use the next flight's start voltage as the rest voltage.
