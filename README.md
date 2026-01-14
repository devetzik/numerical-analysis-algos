# Numerical Analysis Algorithms & Applications

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

This repository contains a comprehensive collection of **Numerical Analysis algorithms** implemented in Python. The project was developed as part of the "Numerical Analysis" course at the **Aristotle University of Thessaloniki (AUTH)**.

It combines implementations for solving non-linear equations, linear systems, eigenvalue problems (PageRank), interpolation/approximation, numerical integration, and real-world data prediction (Stock Market).

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
  - [1. Root Finding](#1-root-finding)
  - [2. Modified Root Finding](#2-modified-root-finding)
  - [3. Linear Systems Solvers](#3-linear-systems-solvers)
  - [4. PageRank Simulation](#4-pagerank-simulation)
  - [5. Interpolation](#5-interpolation)
  - [6. Numerical Integration](#6-numerical-integration)
  - [7. Stock Prediction](#7-stock-prediction)
- [Project Structure](#project-structure)
- [Requirements](#requirements)

## Overview

The goal of this project is to implement fundamental numerical methods from scratch—without relying on black-box solvers for the core logic—and apply them to mathematical and real-world problems.

The code explores:
* Comparing convergence rates of different iterative methods.
* Solving sparse linear systems with thousands of variables.
* Simulating the core logic behind Google's search ranking.
* Predicting future stock values using polynomial regression.

## Key Features

### 1. Root Finding
Implementation of standard iterative methods to find roots of non-linear equations (e.g., $f(x) = 0$).
* **Algorithms:** Bisection, Newton-Raphson, Secant.
* **Focus:** Understanding basic convergence behavior.

### 2. Modified Root Finding
Advanced variations of the standard methods designed for optimization.
* **Algorithms:**
    * **Modified Newton-Raphson:** Handles multiple roots using the 2nd derivative.
    * **Modified Bisection:** Splits intervals at $1/3$ for faster narrowing.
    * **Modified Secant:** Uses a 3-point parabolic fit (Muller-like approach).

### 3. Linear Systems Solvers
Direct and iterative methods for solving systems of the form $Ax = b$.
* **PA = LU Decomposition:** With partial pivoting for stability.
* **Cholesky Decomposition:** For symmetric positive-definite matrices.
* **Gauss-Seidel:** Iterative solver for large sparse systems ($5000 \times 5000$).

### 4. PageRank Simulation
A simulation of the Google Search ranking algorithm.
* **Features:** Construction of the Google Matrix, Power Method implementation, handling damping factors ($q$), and experimenting with link analysis/manipulation.

### 5. Interpolation
Approximating functions (e.g., $sin(x)$) using discrete points.
* **Methods:** Lagrange Interpolation, Cubic Splines, and Least Squares (Polynomial Regression).

### 6. Numerical Integration
Computing definite integrals numerically.
* **Methods:** Trapezoidal Rule and Simpson's Rule.
* **Analysis:** Comparing numerical errors against theoretical bounds.

### 7. Stock Prediction
Real-world application using Least Squares.
* **Task:** Predicting stock prices (e.g., AKTOR, TITAN) by fitting polynomials to historical closing data.

## Project Structure

The project is organized into numbered folders corresponding to the course exercises.

```bash
├── 1. root_finding/
│   ├── root_finding_demo.py           # Demonstration script
│   └── root_finding_lib.py            # Implementation of Bisection, Newton, Secant
│
├── 2. modified_root_finding/
│   ├── modified_root_finding_demo.py  # Demo/Comparison script
│   └── modified_root_finding_lib.py   # Implementation of Modified methods
│
├── 3. linear_system_solvers/
│   ├── linear_system_solvers_demo.py  # Demo on sparse matrices
│   └── linear_system_solvers_lib.py   # Implementation of LU, Cholesky, Gauss-Seidel
│
├── 4. pagerank/
│   ├── pagerank_simulation_experiments.py # Scenarios (Damping factor, Link analysis)
│   └── pagerank_simulation_lib.py         # PageRank Solver class
│
├── 5. interpolation/
│   └── interpolation.py               # Lagrange, Splines, Least Squares viz
│
├── 6. numerical_integration/
│   └── numerical_integration.py       # Simpson & Trapezoidal integration
│
├── 7. stock_prediction/
│   └── stock_prediction.py            # Stock market polynomial fitting
│
└── analysis_report.pdf                # Detailed PDF report with theory and results
```

## Requirements

To run the scripts, you need Python 3.x and the following libraries:

NumPy: For matrix operations and linear algebra.

Matplotlib: For plotting functions and error graphs.