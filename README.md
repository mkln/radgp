## Radial neighbors GP

This package implements RadGP as introduced in 
[*Radial Neighbors for Provably Accurate Scalable Approximations of Gaussian Processes*](https://arxiv.org/abs/2211.14692)
(Yichen Zhu, Michele Peruzzi, Cheng Li and David B. Dunson, 2022).

```
install.packages(c("GPvecchia", "Rcpp", "RcppArmadillo", "RcppEigen"))

devtools::install_github("mkln/radgp")
```
Or download source in zip file, then
```
devtools::install_local("radgp-main.zip")
```


### Abstract:

In geostatistical problems with massive sample size, Gaussian processes (GP) can be approximated using sparse directed acyclic graphs to achieve scalable O(n) computational complexity. In these models, data at each location are typically assumed conditionally dependent on a small set of parents which usually include a subset of the nearest neighbors. These methodologies often exhibit excellent empirical performance, but the lack of theoretical validation leads to unclear guidance in specifying the underlying graphical model and may result in sensitivity to graph choice. We address these issues by introducing radial neighbors Gaussian processes and corresponding theoretical guarantees. We propose to approximate GPs using a sparse directed acyclic graph in which a directed edge connects every location to all of its neighbors within a predetermined radius. Using our novel construction, we show that one can accurately approximate a Gaussian process in Wasserstein-2 distance, with an error rate determined by the approximation radius, the spatial covariance function, and the spatial dispersion of samples. Our method is also insensitive to specific graphical model choice. We offer further empirical validation of our approach via applications on simulated and real world data showing state-of-the-art performance in posterior inference of spatial random effects.
