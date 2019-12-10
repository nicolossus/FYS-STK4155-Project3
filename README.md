# Project 3: Solving Differential Equations with Feed-Forward Neural Network Models

This repository contains programs, material and report for project 3 in FYS-STK4155 made in collaboration between [Tobias](https://github.com/vxkc), [Nicolai](https://github.com/nicolossus) and [Kristian](https://github.com/KristianWold).

The aim of this project is to design FFNN models suitable for solving differential equations. Traditional numerical methods and closed form solutions, when present, have been used to assess the efficiency and accuracy of the FFNN models.

The first FFNN model is employed to learn the initial-boundary problem given by the heat equation, a PDE with both temporal and spatial dependence. The problem formulation is scaled to the standard unity interval (0,1) in one spatial dimension.

A FFNN model is also employed to learn the solution to the nonlinear, coupled ODE, presented by Yi et. al in the article from [Computers and Mathematics with Applications 47, 1155 (2004)](https://www.sciencedirect.com/science/article/pii/S0898122104901101), describing the state of a CTRNN model. Given a real symmetric matrix A in the source term, the temporal dynamic described by this ODE has convergence properties to the largest eigenvalue. Simply replacing A with -A yield the smallest eigenvalue. The article also states that the network should converge to a different eigenvalue if the initial vector is orthogonal to the eigenvector corresponding to the largest eigenvalue. The aim is to design a FFNN model suitable for solving this ODE, and check if it succeed in computing both the largest and smallest eigenvalue for some benchmark 3x3 and 6x6 real symmetric matrices. We will also check whether the network converges to a different eigenvalue than the largest if the initial vector is chosen to be orthogonal to the eigenvector corresponding to the largest eigenvalue.

### Structure

The __[latex folder](https://github.com/nicolossus/FYS-STK4155-Project2/tree/master/latex)__ contains the LaTeX source for building the report, as well as __[figures](https://github.com/nicolossus/FYS-STK4155-Project2/tree/master/latex/figures)__ and __[tables](https://github.com/nicolossus/FYS-STK4155-Project2/tree/master/tables)__ generated in the analyses.

The __[notebooks folder](https://github.com/nicolossus/FYS-STK4155-Project2/tree/master/notebooks)__ contains Jupyter notebooks used in the analyses. For details, see the __[notebooks readme](https://github.com/nicolossus/FYS-STK4155-Project2/blob/master/notebooks/README.md)__.

The __[report folder](https://github.com/nicolossus/FYS-STK4155-Project2/tree/master/report)__ contains the report rendered to PDF from the LaTeX source.

The __[resources folder](https://github.com/nicolossus/FYS-STK4155-Project2/tree/master/resources)__ contains project resources such as supporting material, raw data to be analysed, etc.

The __[src folder](https://github.com/nicolossus/FYS-STK4155-Project2/tree/master/src)__ contains the source code. For details, see the __[src readme](https://github.com/nicolossus/FYS-STK4155-Project2/blob/master/src/README.md)__.
