# MaGeZ contribution to the PETRIC reconstruction challenge

This repository contains algorithms developed by the MaGeZ team submitted to
the [2024 PETRIC reconstruction challenge](https://github.com/SyneRBI/PETRIC).

## Authors

- Matthias Ehrhardt, Univeristy of Bath, United Kingdom
- Georg Schramm, KU Leuven, Belgium
- Zeljko Kereta, Univeristy College London, United Kingdom

## Algorithms

### ALG1

Better-preconditioned SVRG based on the works of:
- R. Twyman et al., "An Investigation of Stochastic Variance Reduction Algorithms for Relative Difference Penalized 3D PET Image Reconstruction," in IEEE TMI, 4, 2023, [link](https://ieeexplore.ieee.org/document/9872020)
- J. Nuyts et al., ""A concave prior penalizing relative differences for maximum-a-posteriori reconstruction in emission tomography," in IEEE TNS, 49, 2022, [link](https://ieeexplore.ieee.org/document/998681)

The update can be summarized as:

$$
x^+ = x + \alpha P \tilde{\nabla} f(x)
$$

where $\tilde{\nabla} f(x)$ is the SVRG stochastic gradient of a subset and
$P$ is a diagonal preconditioner calculated as the harmonic mean of
$x / A^T 1$ and the diagonal of the Hessian of the RDP, as proposed
in work of Nuyts et al.
The scalar step size $\alpha$ is set to 2 in the first epochs and then decreased
to 1 and finally to 0.5 in late epochs.

### ALG2

foo bar

### ALG3

foo bar
