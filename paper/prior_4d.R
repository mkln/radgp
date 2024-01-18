library(tidyverse)
library(magrittr)
library(Matrix)
library(radgp)
library(ggplot2)
library(gridExtra)
library(MASS)
library(dplyr)
library(cowplot)

Chol_dag <- function(cov, dag){
  n = nrow(cov)
  I = diag(rep(1,n))
  Dvec = diag(cov)
  B = matrix(0,nrow=n,ncol=n)
  adjL = matrix(0,nrow=n,ncol=n)
  for (i in 1:n){
    ni = dag[[i]]
    adjL[i,i] = 1
    if (length(ni)>0){
      cov_ni_inv = solve(cov[ni,ni])
      adjL[ni,i] = 1
      B[i,ni] = cov_ni_inv %*% cov[ni,i]
      Dvec[i] = cov[i,i] - t(cov[ni,i]) %*% cov_ni_inv %*% cov[ni,i]
    }
  }
  return(list(I-B, Dvec))
}

cov_dag <- function(cov, dag){
  n = nrow(cov)
  I = diag(rep(1,n))
  Dvec = diag(cov)
  B = matrix(0,nrow=n,ncol=n)
  adjL = matrix(0,nrow=n,ncol=n)
  for (i in 1:n){
    ni = dag[[i]]
    if (length(ni)>0){
      cov_ni_inv = solve(cov[ni,ni])
      adjL[ni,i] = 1
      B[i,ni] = cov_ni_inv %*% cov[ni,i]
      Dvec[i] = cov[i,i] - t(cov[ni,i]) %*% cov_ni_inv %*% cov[ni,i]
    }
  }
  IB_inv = solve(I-B)
  cov_appro = IB_inv %*% diag(Dvec) %*% t(IB_inv)
  adj = adjL %*% adjL
  return( list(IB_inv %*% diag(Dvec) %*% t(IB_inv), sum(adj!=0)/n) )
}

matrix_half <- function(M){
  eigenobj = eigen(M)
  U = eigenobj$vectors
  Dvec = pmax(eigenobj$values,0)
  Dhalf = diag(sqrt(Dvec))
  return( U %*% Dhalf %*% t(U))
}

tr<- function(M){
  return(sum(diag(M)))  
}

W22 <- function(cov1, cov2){
  cov1_half = matrix_half(cov1)
  cov_mix = matrix_half( cov1_half %*% cov2 %*% cov1_half)
  return(tr(cov1) + tr(cov2) - 2*tr(cov_mix))
}

mean_neighbors <- function(dag){
  S = 0
  n = length(dag)
  for (i in 1:n){
    S = S + length(dag[[i]])
  }
  return(S/n)
}

dichotomy_solver <- function(fun, l, r, tol=1e-5){
  diff = 100 + tol
  while (diff>tol){
    m = (l+r)/2
    fun_value = fun(m)
    diff = abs(fun_value)
    if (fun_value<0){
      l = m
    } else{
      r = m
    }
  }
  return(m)
}

cov_mat <- function(coords, kernel){
  n = nrow(coords)
  C = matrix(0, nrow=n, ncol=n)
  k0 = kernel(0)
  for (i in 1:n){
    C[i,i] = k0
    for (j in 1:(i-1)){
      C[i,j] = kernel(sqrt(sum((coords[i,]-coords[j,])^2)))
      C[j,i] = C[i,j]
    }
  }
  return(C)
}

spin_sort <- function(coords){
  n = nrow(coords)
  center = colMeans(coords)
  ds = rowSums((coords - matrix(1, nrow=n, ncol=1) %*% matrix(center,nrow=1))^2)
  sorted_ind = sort(ds,index.return=TRUE)$ix
}

## trying training data with latin hypercube design
library(lhs)
ntrain = 2500
ntest = 2500
set.seed(77)
coords_train = randomLHS(ntrain, 4)


## Matern covariance function
theta <- c(20, 1, 3/2, 0)
nu_matern = 3/2
dist_thre = 0.05
matern_hint_fun <- function(phi, dist=0.15, sigmasq=1, nu=nu_matern, thre=dist_thre){
  return( dist_thre - spaMM::MaternCorr(dist, phi, smoothness=nu, nu=nu, Nugget = NULL) )
} 
theta[1] = dichotomy_solver(matern_hint_fun, 10, 50)
nugget <- 0.01
CC <- radgp::Correlationc(coords_train, coords_train, theta, 1, T)  ## 0 for power exponential, anything else for matern
CC_train <- CC[1:ntrain,1:ntrain]

rho_lst = c(0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32)
m_lst = c(2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 24, 27, 30, 35)


## nngp
W22_mat = matrix(0,nrow=length(m_lst),ncol=2)
for (j in 1:length(m_lst)){
  m = m_lst[j]
  ixmm <- GPvecchia::order_maxmin_exact(coords_train)
  coords_mm <- coords_train[ixmm,]
  nn_dag_mat <- GPvecchia:::findOrderedNN_kdtree2(coords_mm, m)
  dag_nn = vector('list',length=ntrain)
  for (i in 1:ntrain){
    neighbor_mat = nn_dag_mat[i,]
    dag_nn[[ixmm[i]]] = ixmm[neighbor_mat[!is.na(neighbor_mat)][-1]]
  }
  CC_train_nn_obj = cov_dag(CC_train, dag_nn)
  W22_mat[j,] = c(CC_train_nn_obj[[2]], W22(CC_train, CC_train_nn_obj[[1]]))
  print(paste('nngp: ', j,'th finished'))
}
W22_dat = as.data.frame(W22_mat)
colnames(W22_dat) = c('Ave.Nonzeros', 'W22')
W22_dat$Method = 'NNGP'

## radgp new 
W22_mat = matrix(0,nrow=length(rho_lst),ncol=2)
sorted_ind = spin_sort(coords_train)
sorted_order = vector('numeric',ntrain)
for (i in 1:ntrain){
  sorted_order[sorted_ind[i]] = i
}
for (j in 1:length(rho_lst)){
  rho = rho_lst[j]
  Nsets = radialndag(coords_train, rho)
  for (i in 1:ntrain){
    Nsets[[i]] = Nsets[[i]] + 1
  }
  CC_train_spin_obj = cov_dag(CC_train, Nsets) 
  W22_mat[j,] = c(CC_train_spin_obj[[2]], W22(CC_train, CC_train_spin_obj[[1]]))
  print(paste('radgp_new: ', j,'th finished'))
}
W22_dat_new = as.data.frame(W22_mat)
colnames(W22_dat_new) = c('Ave.Nonzeros', 'W22')
W22_dat_new$Method = 'RadGP'
W22_dat = rbind(W22_dat, W22_dat_new)

## plot results
p1 = ggplot(W22_dat) +
  geom_line(aes(x=Ave.Nonzeros,y=W22,color=Method),size=1.5) +
  theme_minimal(base_size = 25) + theme(legend.position='none') + ylab(bquote(W[2]^2)) 
W22_dat_m = W22_dat
W22_dat_m$logW22 = log(W22_dat_m$W22)
p2 = ggplot(W22_dat_m) +
  geom_line(aes(x=Ave.Nonzeros,y=logW22,color=Method),size=1.5) + 
  theme_minimal(base_size = 25) + ylab(bquote(logW[2]^2)) 
plot_grid(p1, p2, align='h', nrow=1, rel_widths=c(0.45,0.55))







