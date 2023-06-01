library(tidyverse)
library(magrittr)
library(Matrix)
library(radgp)
library(ggplot2)
library(gridExtra)
library(RcppHungarian)

cov_dag <- function(cov, dag){
  n = nrow(cov)
  I = diag(rep(1,n))
  Dvec = diag(cov)
  B = matrix(0,nrow=n,ncol=n)
  for (i in 1:n){
    ni = dag[[i]]
    if (length(ni)>0){
      cov_ni_inv = solve(cov[ni,ni])
      B[i,ni] = cov_ni_inv %*% cov[ni,i]
      Dvec[i] = cov[i,i] - t(cov[ni,i]) %*% cov_ni_inv %*% cov[ni,i]
    }
  }
  IB_inv = solve(I-B)
  return( IB_inv %*% diag(Dvec) %*% t(IB_inv) )
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


## data
nl = 40
ntrain = nl^2
ntest = 1000
nall = ntrain + ntest
coords_train = as.matrix(expand.grid(xout<-seq(0,1,length.out=nl),xout))
coords_test = matrix(runif(ntest*2),nrow=(ntest),ncol=2)
coords_all = rbind(coords_train, coords_test)

phi = - log(0.05)/0.15^2
expfun <- function(x,phi=133.14){
  return(exp(-phi*x^2))
}
CC_train = cov_mat(coords_train, expfun)

# theta <- c(20, 1, 3/2, 0)
# matern_hint_fun <- function(phi, dist=0.15, sigmasq=1, thre=0.05){
#   return( thre - sigmasq*(1+phi*dist) * exp(-phi*dist) )
# }
# theta[1] = dichotomy_solver(matern_hint_fun, 10, 50)
# nugget <- 1e-5
# CC <- radgp::Correlationc(coords_all, coords_all, theta, TRUE) 
# CC_train <- CC[1:ntrain,1:ntrain]

# rho_lst = c(0.05, 0.1, 0.15, 0.175, 0.2, 0.225, 0.25, 0.5, 1)
# m_lst = c(3, 5, 10, 20, 40, 80, 130, 200)
rho_lst = c(0.05, 0.07, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15)
m_lst = c(2, 4, 6, 8, 10, 12, 14, 17, 20, 30, 40, 50)
W22_mat = matrix(0,nrow=length(rho_lst)+length(m_lst),ncol=2)

## radgp 
for (j in 1:length(rho_lst)){
  rho = rho_lst[j]
  dag_rad <- radial_neighbors_dag(coords_train, rho)
  for (i in 2:ntrain){
    dag_rad$dag[[i]] = dag_rad$dag[[i]] + 1  ## C++ starts from 0 while R starts from 1
  }
  n_neighbors = rep(0,ntrain)
  for (i in 1:ntrain){
    n_neighbors[i] = length(dag_rad$dag[[i]])
  }
  CC_train_rad = cov_dag(CC_train, dag_rad$dag) 
  W22_mat[j,] = c(mean_neighbors(dag_rad$dag), W22(CC_train, CC_train_rad))
}

## nngp
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
  CC_train_nn = cov_dag(CC_train, dag_nn)
  W22_mat[j+length(rho_lst),] = c(mean_neighbors(dag_nn), W22(CC_train, CC_train_nn))
}

W22_dat = as.data.frame(W22_mat)
colnames(W22_dat) = c('ave.neighbors', 'W22')
W22_dat$method = c(rep('RadGP',length(rho_lst)), rep('nngp',length(m_lst)))
ggplot(W22_dat) +
  geom_line(aes(x=ave.neighbors,y=W22,color=method))




# ## artificial test codes
# dag_test = vector('list',ntrain)
# for (i in 2:ntrain){
#   dag_test[[i]] = seq(1,i-1,1)
# }
# # H_test = Phi(CC_train, dag_test)
# CC_train_test = cov_dag(CC_train, dag_test)
# W22_test = W22(CC_train, CC_train_test)







