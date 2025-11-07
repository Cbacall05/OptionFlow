# OptionFlow
Physics-Informed Neural Networks (PINNs) for option pricing. Bridging quantitative finance and scientific machine learning through PDE-constrained neural models for Black-Scholes, American, and inverse-volatility problems.

# 🧠 Physics-Informed Neural Networks (PINNs) for Option Pricing

This repository documents an ongoing research project exploring how **Physics-Informed Neural Networks (PINNs)** can solve and extend the **Black–Scholes Partial Differential Equation (PDE)** for option pricing.  
It merges rigorous mathematical modeling with modern deep-learning tools to bridge **quantitative finance** and **scientific machine learning**.

---

## ⚙️ Project Overview

The goal is to build, validate, and extend neural models that learn to price options by directly satisfying the governing PDEs—without requiring large labeled datasets.

**Core Objectives**
1. Develop a finite-difference (FD) solver as the numerical “ground truth.”  
2. Train a PINN that reproduces the same solution purely from the PDE, boundary, and terminal conditions.  
3. Extend the framework to harder problems such as the **American option (free-boundary problem)** or **volatility calibration (inverse problem)**.  
4. Benchmark, visualize, and interpret the model’s ability to generalize and compute analytical Greeks.

---

## 🧩 Repository Structure

pinn-option-pricing/
│
├── notebooks/
│ ├── Notebook_1_Baseline.ipynb # FD solver + analytical comparison
│ ├── Notebook_2_PINN.ipynb # Vanilla PINN implementation
│ ├── Notebook_3_Research.ipynb # American option or inverse-volatility study
│
├── src/
│ ├── fd_solver.py
│ ├── pinn_model.py
│ ├── utils.py
│
├── data/ # Optional simulated or market data
├── results/ # Plots, surfaces, and figures
├── paper/ # Drafts or slides for publication
│
├── README.md
├── requirements.txt
└── LICENSE


---

## 🔬 Methodology Snapshot

### Finite-Difference Benchmark  
A Crank–Nicolson (or other stable) finite-difference scheme solves the Black–Scholes PDE to create the baseline surface \(V(S, t)\).

### Physics-Informed Neural Network  
A neural network \(V_\theta(S, t)\) is trained to minimize a composite loss:
\[
L = w_{\text{pde}} L_{\text{pde}} + w_{\text{bc}} L_{\text{bc}} + w_{\text{data}} L_{\text{data}}
\]
where \(L_{\text{pde}}\) enforces the Black–Scholes PDE residual via autograd,  
\(L_{\text{bc}}\) imposes boundary and terminal conditions, and \(L_{\text{data}}\) optionally anchors known prices.

**Tuning Focus**
- Over-weight boundary terms initially (`w_bc ≈ 100`).
- Train on `log(S)` for numerical stability.
- Optionally normalize gradient norms of each loss term each epoch for stability.

---

## 🚀 Research Directions

### **A. American Option (Free-Boundary)**
Introduce an inequality constraint enforcing early-exercise:
\[
L_{\text{exercise}} = \text{ReLU}(\max(S-K, 0) - V_\theta(S,t))
\]
The network learns both the pricing surface and the **optimal exercise curve** \(S^*(t)\).  
Compare against a binomial-tree or finite-difference benchmark.

### **B. Inverse Problem (Volatility Calibration)**
Make volatility σ a trainable parameter inside the PDE, using real or simulated market data.  
Constrain σ > 0 via `softplus` or `exp`, and recover an **implied-volatility surface** consistent with both data and the PDE.

---

## 📊 Planned Deliverables

- **Notebook 1:** FD solver and analytical validation.  
- **Notebook 2:** PINN reproducing the Black–Scholes solution.  
- **Notebook 3:** Research extension (American or inverse volatility).  
- **Benchmark report:** Accuracy, runtime, and Greeks comparison.  
- **Paper draft:** Suitable for NeurIPS FinML or ICML AI-Finance workshops.  

---

## 🧾 Documentation & Reproducibility

- `requirements.txt` lists dependencies.  
- Each notebook is self-contained and reproducible with fixed random seeds.  
- Figures and animations (GIFs) of surface convergence will be added as experiments progress.

---

## 🧠 Author & Status

**Lead Researcher:** [Calvin Bacall]  
**Affiliation:** University of Notre Dame — Physics / Economics / Theology  
**Current Focus:** Refining PINN loss-term balancing and extending to free-boundary PDEs.  

_This repository is a work-in-progress; expect iterative commits as the research develops._  
