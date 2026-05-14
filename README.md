# Tree Option Pricing

> Academic project — Université Paris-Dauphine  
> Trinomial tree pricer for European and American options, benchmarked against Black-Scholes closed-form solutions.

---

## Overview

This project implements a **trinomial tree** for pricing vanilla options (calls and puts, European and American), with support for:

- **Discrete cash dividends** with exact placement on the tree timeline
- **Probability pruning** to control tree size and computation time
- **Greek computation** via finite differences on the tree (Δ, Γ, Θ, ρ, Vega)
- **Black-Scholes benchmark** for European options, including full Greeks
- **Excel export** of tree structure (stock prices and transition probabilities) via `xlwings`

---

## Repository Structure

```
.
├── arbre_trino.py   # Core: trinomial tree construction and pricing engine
├── bs.py            # Black-Scholes pricer + numerical Greeks via the tree
├── simulation.py    # Thin wrapper: builds a tree, prices an option, logs timing
├── main.py          # Entry point: example calls to price options and compare models
├── analyse.ipynb    # Analysis notebook: convergence study, Greeks, BS vs tree comparison
└── README.md
```

---

## Financial Model

### Trinomial tree (`arbre_trino.py`)

Each node of the tree has three possible successors — up (`fg`), middle (`fm`), down (`fd`) — linked in a doubly-connected mesh to allow re-convergence of paths:

```
         ┌── fg  (× α)
noeud ───┼── fm  (forward)
         └── fd  (÷ α)
```

The branching factor α and the step size δt are calibrated to match the local variance of the underlying:

```
α   = exp(r·δt + σ·√3·√δt)
δt  = T / (n · 365)
```

Transition probabilities are derived by matching the first two moments of the log-return distribution under the risk-neutral measure, accounting for a discrete dividend paid at a specified date.

**Pruning:** nodes whose cumulative probability falls below a configurable threshold are collapsed onto their middle child, saving memory and compute without materially affecting the price.

### Pricing (`Tree.pricing`)

- **European options:** the price is the probability-weighted sum of payoffs at maturity, discounted at the risk-free rate.
- **American options:** backward induction from maturity to root, comparing the continuation value at each node against immediate exercise.

### Black-Scholes (`bs.py`)

The `BS` class computes the closed-form price and Greeks (Δ, Γ, Θ, ρ, Vega) for European options under continuous dividend yield. It also exposes methods to re-derive those Greeks numerically from the trinomial tree via finite differences — enabling a direct comparison between the analytical and tree-based sensitivities.

---

## Data Model

The project uses Python `dataclasses` and `Enum` for clean, typed input:

```python
from arbre_trino import Option, Market, Nature, Type
from datetime import datetime

market = Market(
    r=0.05,                              # Risk-free rate
    vol=0.20,                            # Implied volatility
    div=2.0,                             # Discrete dividend amount
    start_date=datetime(2023, 1, 1),
    div_date=datetime(2023, 6, 1),       # Dividend payment date
    stock_price=100.0
)

option = Option(
    maturity=datetime(2024, 1, 1),
    nature=Nature.CALL,                  # Nature.CALL or Nature.PUT
    type=Type.AMERICAN,                  # Type.EUROPEAN or Type.AMERICAN
    K=105.0                              # Strike
)
```

---

## Quickstart

### Prerequisites

```bash
pip install numpy xlwings
```

> `xlwings` requires a local Excel installation for the `print_xl` export feature. All other features run without it.

### Price an option

```python
from simulation import simulation

price = simulation(n=100, opt=option, mark=market)
```

Output:
```
Temps de création de l'arbre: 0.43s
Pricing:
----------------------------
8.317...
Temps du pricing: 0.02s
----------------------------
```

### Compare tree vs Black-Scholes

```python
from bs import BS
from arbre_trino import Type, Nature

# Black-Scholes closed-form
bs = BS(mark=market, opt=option)
bs.print_bs()

# Greeks from the tree via finite differences
bs.compute_delta_gamma(n=100, opt=option, mark=market, born=5.0, pas=20)
bs.compute_vega(n=100, opt=option, mark=market, born=0.05, pas=20)
bs.compute_theta(n=100, opt=option, mark=market, delta=1, nb=10)
bs.compute_rho(n=100, opt=option, mark=market, born=0.01, pas=20)
```

### Export the tree to Excel

```python
from arbre_trino import Tree

tree = Tree(n=10, mark=market, opt=option, pruning=[True, 1e-9])
tree.print_xl()  # Writes to tree_debug.xlsx in the working directory
```

### Analysis notebook

Open `analyse.ipynb` for a full convergence study (price vs. number of steps), Greek surface plots, and side-by-side comparison between the tree pricer and Black-Scholes.

---

## Implementation Notes

- The tree is built as a **linked node graph** (not a matrix), which naturally handles non-recombining paths around the dividend date and avoids allocating memory for unreachable nodes.
- The `Node_trunk` subclass marks nodes on the central spine of the tree (i.e. the forward path from the root), which is used during backward induction to navigate back to the root.
- Greek computation via finite differences perturbs a single market parameter, rebuilds the full tree, and applies numerical differentiation — straightforward but computationally expensive for large `n`.
