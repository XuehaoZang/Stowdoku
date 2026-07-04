# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Stowdoku is a CSP (constraint satisfaction) solver for containership stowage planning. Given a vessel's slot geometry and a cargo booking forecast (cbf) per port pair, it assigns container destination ports (POD) and types (GP/reefer) to physical slots across multiple ports of call, backtracking on dead ends and never allowing overstow (a container destined for a nearer port stowed under one destined further away).

Comments and docstrings in this codebase are written in Chinese; match that convention when editing existing files.

## Run

```bash
pip install -e .
python CSP_solver.py          # runs the built-in test scenario in __main__
python debug/test_VesselCass.py   # exercises Vessel class methods directly (uses sys.path hack, run from repo root)
```

There is no formal test runner configured — `debug/test_VesselCass.py` is a standalone script with inline `assert`s, run directly with `python`.

## Architecture

### Coordinate system
Every vessel is represented as a `(n_bay, 2, 2)` array: `bay` (0-indexed, only valid large bays), `lr` (0=left/1=right half of the bay), `hd` (0=hold/1=deck). This is the "田字格" (2x2 grid) unit per bay referenced throughout comments and `print_vessel`.

### `VesselClass.py` — `Vessel`
Holds all solver state, split into two layers:
- **Static geometry (read-only after init)**: `is_valid`, `capacity_total` (40ft slot count per cell, used for GP assignment), `capacity_rf` (reefer-plug slot count, used for RF assignment), `has_reefer` (derived bool mask). `n_bay` derived from `is_valid.shape[0]`.
- **Dynamic search state (mutated during backtracking)**: `vessel_pod`/`vessel_type` (per-cell assignment, -1/None = unassigned), `cbf` (`{POL: {POD: {"GP": n, "RF": n}}}`, mutated in place as containers are assigned/unassigned), `current_pol` (pointer into `cbf`, advanced on port change rather than replacing the dict).

Key invariant enforced by `get_candidates`: within a bay's `(bay, lr)` column, the hold cell's POD must be >= the deck cell's POD (no overstow — you can't load something destined further away on top of something destined nearer).

`assign`/`unassign` mutate `cbf` counts directly (GP debits `capacity_total`, RF debits `capacity_rf`) so unassignment must pass back the exact same `(bay, lr, hd, pod, ctype)` tuple used to assign. `discharge`/`undischarge` only touch `vessel_pod`/`vessel_type`, not `cbf` (arrivals don't change future booking counts). `snapshot`/`restore` deep-copy the full dynamic layer for cross-port backtracking.

### `CSP_solver.py` — `solve()`
Single recursive function that unifies loading and port-advancement in one search tree (see the recursion diagram in `README.md`):
1. If all ports are done, succeed.
2. If the current port's `cbf` is fully assigned (`port_complete()`), snapshot the departure state, advance `current_pol`, discharge arrivals, and recurse — treating the discharge as a special node in the same tree. Failure here backtracks across the port boundary (`undischarge` + `current_pol -= 1`), not just within the current port's loading.
3. Otherwise compute candidates for every unassigned valid cell (`cal_candidates`); an empty candidate set on a cell that still has remaining cbf demand is a dead cell → backtrack immediately.
4. Pick the next cell via MRV (`mrv_select`): prioritize cells with reefer capability that have an RF candidate, then smallest candidate-set size, and try each `(POD, ctype)` candidate in sorted order, recursing and unassigning on failure.

`arxiv/CSP_Planning.py` and `utils/vessel.py` are an earlier single-port-only version of this same algorithm, superseded by `VesselClass.py`/`CSP_solver.py` — don't extend them; only `Vessel`/`solve()` are current.

### Data files
`data/test_data_*.json` are hand-built scenarios (`init` grid + `cbf` demand) used to sanity-check that the solver learns non-obvious placement rules (e.g. loading a far-destined container standing on end so it doesn't force overstow later). See `README.md` for the worked example and expected output for `test_data_1.json`.
