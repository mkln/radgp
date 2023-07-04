library(tidyverse)
library(magrittr)
library(Matrix)
library(radgp)
library(ggplot2)
library(gridExtra)
library(RcppHungarian)
library(MASS)

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

# Chol_true <- function(cov, ord){
#   
# }


cov_dag <- function(cov, dag){
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

spin_sort <- function(coords, corners){
  n = nrow(coords)
  n_corners = nrow(corners)
  d_2_corners = matrix(0, nrow = n, ncol = n_corners)
  for (i in 1:n_corners){
    d_2_corners[,i] = rowSums((coords - matrix(1, nrow=n, ncol=1) %*% corners[i,])^2)
  }
  ds = vector('numeric',n)
  for (j in 1:n){
    ds[j] = max(d_2_corners[j,])
  }
  coords_init = coords[which.min(ds),]
  d_2_init = rowSums((coords - matrix(1, nrow=n, ncol=1) %*% coords_init)^2)
  sorted_ind = sort(d_2_init,index.return=TRUE)$ix
}


## grid training data
nl = 40
ntrain = nl^2
coords_train = as.matrix(expand.grid(xout<-seq(0,1,length.out=nl),xout))

# ## uniform training data
# ntrain = 1600
# coords_train = matrix(runif(ntrain*2),nrow=(ntrain),ncol=2)

## uniform test data
ntest = 1000
nall = ntrain + ntest
coords_test = matrix(runif(ntest*2),nrow=(ntest),ncol=2)
coords_all = rbind(coords_train, coords_test)

## Gaussian covariance function
# phi = - log(0.05)/0.15^2
# expfun <- function(x,phi=133.14){
#   return(exp(-phi*x^2))
# }
# CC_train = cov_mat(coords_train, expfun)

## Matern covariance function
theta <- c(20, 1, 3/2, 0)
matern_hint_fun <- function(phi, dist=0.15, sigmasq=1, thre=0.05){
  return( thre - sigmasq*(1+phi*dist) * exp(-phi*dist) )
}
theta[1] = dichotomy_solver(matern_hint_fun, 10, 50)
nugget <- 1e-5
CC <- radgp::Correlationc(coords_all, coords_all, theta, 0, T)
CC_train <- CC[1:ntrain,1:ntrain]

# rho_lst = c(0.05, 0.1, 0.15, 0.175, 0.2, 0.225, 0.25, 0.5, 1)
# m_lst = c(3, 5, 10, 20, 40, 80, 130, 200)
rho_lst = c(0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15)
m_lst = c(3, 5, 7, 10, 13, 16, 20, 25, 30, 36, 43, 50)
# rho_lst = c(0.05, 0.07, 0.09, 0.10, 0.11)
# m_lst = c(2, 4, 6, 8, 10, 12, 14, 17)


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
  W22_mat[j,] = c(mean_neighbors(dag_nn), W22(CC_train, CC_train_nn_obj[[1]]))
  print(paste('nngp: ', j,'th finished'))
}
W22_dat = as.data.frame(W22_mat)
colnames(W22_dat) = c('ave.neighbors', 'W22')
W22_dat$method = 'nngp'

## radgp new 
W22_mat = matrix(0,nrow=length(rho_lst),ncol=2)
corners = matrix(c(0,0,1,0,0,1,1,1),nrow=4,ncol=2)
sorted_ind = spin_sort(coords_train, corners)
sorted_order = vector('numeric',ntrain)
for (i in 1:n){
  sorted_order[sorted_ind[i]] = i
}
for (j in 1:length(rho_lst)){
  rho = rho_lst[j]
  Rsets = neighbor_search(coords_train, rho)
  Nsets = vector('list',ntrain)
  for (i in 1:ntrain){
    candidates = Rsets[[i]] + 1   ## C++ starts from 0 while R starts from 1
    Nsets[[i]] = candidates[which(sorted_order[candidates] < sorted_order[i])]
  }
  ## add a single closest location when there is no parents
  for (i in 2:ntrain){
    i_origin = sorted_ind[i]
    if (length(Nsets[[i_origin]]) == 0){
      coords_now = coords_train[i_origin,]
      inds_b = sorted_ind[1:(i-1)]
      ds = rowSums((coords_train[inds_b,] - matrix(1,nrow=length(inds_b),ncol=1) %*% coords_now)^2)
      Nsets[[i_origin]] = inds_b[which.min(ds)]
    }
  }
  CC_train_spin = cov_dag(CC_train, Nsets)[[1]] 
  W22_mat[j,] = c(mean_neighbors(Nsets), W22(CC_train, CC_train_spin))
  print(paste('radgp_new: ', j,'th finished'))
}
W22_dat_new = as.data.frame(W22_mat)
colnames(W22_dat_new) = c('ave.neighbors', 'W22')
W22_dat_new$method = 'radgp_new'
W22_dat = rbind(W22_dat, W22_dat_new)

## radgp old
W22_mat = matrix(0,nrow=length(rho_lst),ncol=2)
for (j in 1:length(rho_lst)){
  rho = rho_lst[j]
  dag_rad <- radial_neighbors_dag(coords_train, rho)
  for (i in 2:ntrain){
    dag_rad$dag[[i]] = dag_rad$dag[[i]] + 1  ## C++ starts from 0 while R starts from 1
  }
  CC_train_rad_obj = cov_dag(CC_train, dag_rad$dag)
  # W22_mat[j,] = c(CC_train_rad_obj[[2]], W22(CC_train, CC_train_rad_obj[[1]]))
  W22_mat[j,] = c(mean_neighbors(dag_rad$dag), W22(CC_train, CC_train_rad_obj[[1]]))
  print(paste('radgp_old: ', j,'th finished'))
}
W22_dat_new = as.data.frame(W22_mat)
colnames(W22_dat_new) = c('ave.neighbors', 'W22')
W22_dat_new$method = 'radgp_old'
W22_dat = rbind(W22_dat, W22_dat_new)






## plot results
ggplot(W22_dat) +
  geom_line(aes(x=ave.neighbors,y=W22,color=method))


# ## plot log results
# W22_dat_m = W22_dat[-which(W22_dat$ave.neighbors<5),]
# W22_dat_m$logW22 = log(W22_dat_m$W22)
# ggplot(W22_dat_m) +
#   geom_line(aes(x=ave.neighbors,y=logW22,color=method))





## codes to investigate same ave.neighbors behaviors
# preci_train = ginv(CC_train)
# 
# rho_com = 0.07
# dag_rad <- radial_neighbors_dag(coords_train, rho_com)
# for (i in 2:ntrain){
#   dag_rad$dag[[i]] = dag_rad$dag[[i]] + 1  ## C++ starts from 0 while R starts from 1
# }
# rad_mats = Chol_dag(CC_train, dag_rad$dag)
# rad_preci = t(rad_mats[[1]]) %*% diag((rad_mats[[2]])^(-1)) %*% rad_mats[[1]]
# rad_preci_F2 = sum((preci_train - rad_preci)^2)
# rad_cov = cov_dag(CC_train, dag_rad$dag)[[1]]
# rad_cov_F2 = sum((CC_train - rad_cov)^2)
# # rad_W22 = W22(CC_train, rad_cov)
# 
# m_com = 6
# ixmm <- GPvecchia::order_maxmin_exact(coords_train)
# coords_mm <- coords_train[ixmm,]
# nn_dag_mat <- GPvecchia:::findOrderedNN_kdtree2(coords_mm, m_com)
# dag_nn = vector('list',length=ntrain)
# for (i in 1:ntrain){
#   neighbor_mat = nn_dag_mat[i,]
#   dag_nn[[ixmm[i]]] = ixmm[neighbor_mat[!is.na(neighbor_mat)][-1]]
# }
# nn_mats = Chol_dag(CC_train, dag_nn)
# nn_preci = t(nn_mats[[1]]) %*% diag((nn_mats[[2]])^(-1)) %*% nn_mats[[1]]
# nn_preci_F2 = sum((preci_train - nn_preci)^2)
# nn_cov = cov_dag(CC_train, dag_nn)[[1]]
# nn_cov_F2 = sum((CC_train - nn_cov)^2)
# # nn_W22 = W22(CC_train, cov_dag(CC_train, nn_cov))









# ## test codes
# library(Matrix)
# library(radgp)
# library(dplyr)
# 
# coords <- cbind(runif(400), runif(400))
# theta <- c(1,1,1.5,0) # phi sigmasq nu nugget
# cov_model <- 0 # 0=power exponential, [anything else]=matern
# CC <- radgp::Correlationc(coords, coords, theta, 1, T)
# rho <- 1
# radgp_out <- radgp::radgp_build(coords, theta, rho, cov_model) # last input 0 for power exp, else matern
# 
# 
# m <- 25
# ixmm <- GPvecchia::order_maxmin_exact(coords)
# coords_mm <- coords[ixmm,]
# nn_dag_mat <- GPvecchia:::findOrderedNN_kdtree2(coords_mm, m)
# nn_dag <- apply(nn_dag_mat, 1, function(x){ x[!is.na(x)][-1]-1 })
# vecchia_out <- radgp::vecchiagp_build(coords_mm, theta, nn_dag, cov_model)








