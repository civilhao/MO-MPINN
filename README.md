# Introduction
This repository provides the implementation of the Multi-Output Multi-Physics-Informed Neural Network (MO-MPINN) framework for solving the Dimension-Reduced Probability Density Evolution Equation (DR-PDEE). 

The Dimension-Reduced Probability Density Evolution Equation (DR-PDEE), driven by the embedded physical principles, presents a promising alternative for evaluating the probability density evolution in stochastic dynamical systems, eliminating the need for redundant computations at multiple representative points. Physics-Informed Neural Networks (PINNs) provide an exceptionally well-suited scheme for solving DR-PDEE due to their ability encoding physical laws and prior physical knowledge into the learning process. However, the challenge arises from the spatio-temporal-dependence of the unknown intrinsic drift coefficients and diffusion coefficients, which act as the physically driven force in DR-PDEE, along with their derivatives. To address the challenges, a novel framework called Multi-Output Multi-Physics-Informed Neural Network (MO-MPINN) that facilitates capturing the evolution of various outputs including time-varying coefficients and response probability density simultaneously is proposed in this study. The framework exhibits five key characteristics: (i) MO-MPINN utilizes multiple neurons in the output layer, thereby eliminating the necessity for distinct identification of the unknown spatio-temporal-dependent coefficients separately via data from representative structural dynamic analyses. (ii) A network architecture with multiple parallel subnetworks with each one corresponding to a different output is proposed to reduce the training complexity. (iii) Multiple physical laws governing the spatio-temporal-dependent coefficients and probability density conservation are embedded in the loss function to ensure an accurate representation of the underlying principles. (iv) Leveraging the automatic differentiation feature of neural networks, the derivative terms of spatio-temporal-dependent coefficients involved in the DR-PDEE can be accurately and efficiently computed without resorting to numerical differentiation. (v) The proposed MO-MPINN can be applied to high-dimensional stochastic linear or nonlinear systems involving double randomness in structural parameters and excitations.  This study provides a new paradigm for solving partial differential equations involving differentiation of spatio-temporal-dependent coefficients. 

# Requirements
- Tensorflow 2.6.0
- scipy 1.12.0
- pydoe 0.3.8
- numpy 1.22.4

# Usage
To use the MO-MPINN framework, clone the repository and refer to the provided example of the eight-dimensional Ornstein-Uhlenbeck process with random stiffness. More details can be found in the paper.

# Acknowledgements
The neural network code for the PINN part is based on the work of Prateek Bhustali. You can find the original code at the following link [gh repo clone omniscientoctopus/Physics-Informed-Neural-Networks](https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks.git).

# Citation
If you use this framework, please cite the following paper:
Teng-Teng Hao; Wang-Ji Yan; Jian-Bing Chen; Ting-Ting Sun; Ka-Veng Yuen [J] Multi-Output Multi-Physics-Informed Neural Network for Learning Dimension-Reduced Probability Density Evolution Equation with Unknown Spatio-Temporal-Dependent Coefficients
