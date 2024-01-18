library(tidyverse)
library(magrittr)
library(Matrix)
library(radgp)
library(ggplot2)
library(gridExtra)
library(MASS)
library(dplyr)
library(cowplot)
library(latex2exp)

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
  return(max(tr(cov1) + tr(cov2) - 2*tr(cov_mix),0))
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
    # print(m)
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

inv_v_r <- function(x, r=1, kmax=1000){
  S = 1
  for (k in 1:kmax){
    temp = 0
    for (j in 1:k){
      temp = temp + log(x/j^r)
    }
    S = S + exp(temp)
  }
  return(1/S)
}


##---------------------------faster than polynomial rate-------------------------------
## Matern covariance function
nu_matern = 3/2
theta <- c(20, 1, nu_matern, 0)
dist_thre = 0.25
matern_hint_fun <- function(phi, dist=0.15, sigmasq=1, nu=nu_matern, thre=dist_thre){
  return( dist_thre - spaMM::MaternCorr(dist, phi, smoothness=nu, nu=nu, Nugget = NULL) )
} 
theta[1] = dichotomy_solver(matern_hint_fun, 1, 100)

matern_fun <- function(x, scaling=theta[1], nu=nu_matern){
  return( spaMM::MaternCorr(x, scaling, smoothness=nu, nu=nu, Nugget = NULL) )
}

## test different rho functions
## for nu=3/2
rho_fun <- function(n, const=200){
  return((log(n))^3/const)
}

## 1d experiment
coord_tot = as.matrix(seq(0,10,1/50), ncol=1)
n_lst = seq(24, 200, 1)
CC_tot <- radgp::Correlationc(coord_tot, coord_tot, theta, 1, T)  ## 0 for power exponential, anything else for matern
nexp = length(n_lst)
W22_mat = matrix(0, nrow=nexp, ncol=2)
rho_lst = rep(0,length(n_lst))

for (iexp in 1:nexp){
  ni = n_lst[iexp]
  coord = matrix(coord_tot[1:ni,1], ncol=1)
  rho = rho_fun(ni)
  rho_lst[iexp] = rho
  CC = CC_tot[1:ni, 1:ni]
  Nsets = radialndag(coord, rho)
  for (i in 1:ni){
    Nsets[[i]] = Nsets[[i]] + 1
  }
  CC_spin_obj = cov_dag(CC, Nsets)
  W22_mat[iexp,] = c(ni, W22(CC, CC_spin_obj[[1]]))
  print(paste('radgp: ', iexp,'th finished'))
}

## fitting theoretical upper bound
# fit scaling constant in v_r
test_x = seq(0, 1, 0.01)
test_n = length(test_x)
s_min = 1
s_max = 1000
tol = 1
while (s_max-s_min>tol){
  s = (s_max+s_min)/2
  test_v = rep(0, test_n)
  test_cov = rep(0, test_n)
  flag = TRUE
  for (j in 1:test_n){
    x = test_x[j]
    test_v[j] = inv_v_r(s*x)
    test_cov[j] = matern_fun(x)
    if (2*test_v[j]/(1+x^2)<test_cov[j]){
      flag = FALSE
      break
    }
  }
  if (flag == TRUE){
    s_min = s
  } else{
    s_max = s
  }
  print(s)
}

upbound <- function(n, in_c=s, out_c=1){
  return(out_c*n * inv_v_r(rho_fun(n)*in_c))
}

bound_lst = rep(0,length(n_lst))
for (i in 1:length(n_lst)){
  bound_lst[i] = upbound(n_lst[i])
}

out_c = max(W22_mat[,2]/bound_lst)
bound_lst = bound_lst * out_c

## plot results
W22_dat = as.data.frame(W22_mat)
colnames(W22_dat) = c('n', 'W22')
W22_dat$Type ='Simulation'
W22_dat_new = data.frame(n=n_lst, W22=bound_lst, Type='Theory')
W22_dat = rbind(W22_dat, W22_dat_new)
p1 =
ggplot(W22_dat) +
  geom_line(aes(x=n,y=W22,color=Type),size=1.5) +
  theme_minimal(base_size = 25) + ylab(bquote(W[2]^2)) + theme(legend.position='none')


##---------------------------polynomial rate-------------------------------
## Matern covariance function
lambda = 5
theta <- c(0, 5)
dist_thre = 0.25
gcauchy_hint_fun <- function(phi, dist=0.15, lam=lambda, thre=dist_thre){
  return( dist_thre - (1+dist*phi)^(-lam) )
} 
theta[1] = dichotomy_solver(gcauchy_hint_fun, 1, 100)

gcauchy_fun <- function(dist, phi=theta[1], lam=lambda){
  return( (1+dist*phi)^(-lam) )
}

## test different rho functions
## for nu=3/2
rho_fun <- function(n, const=20){
  return(n^(1/2)/const)
}

## 1d experiment
coord_tot = as.matrix(seq(0,10,1/50), ncol=1)
n_lst = seq(50, 500, 1)
CC_tot = matrix(0, nrow=length(coord_tot), ncol=length(coord_tot))
for (i in 1:length(coord_tot)){
  for (j in 1:i){
    CC_tot[i,j] = gcauchy_fun(abs(coord_tot[i]-coord_tot[j]))
    CC_tot[j,i] = gcauchy_fun(abs(coord_tot[i]-coord_tot[j]))
  }
}
nexp = length(n_lst)
W22_mat_p = matrix(0, nrow=nexp, ncol=2)
rho_lst = rep(0, length(n_lst))

for (iexp in 1:nexp){
  ni = n_lst[iexp]
  coord = matrix(coord_tot[1:ni,1], ncol=1)
  rho = rho_fun(ni)
  rho_lst[iexp] = rho
  CC = CC_tot[1:ni, 1:ni]
  Nsets = radialndag(coord, rho)
  for (i in 1:ni){
    Nsets[[i]] = Nsets[[i]] + 1
  }
  CC_spin_obj = cov_dag(CC, Nsets)
  W22_mat_p[iexp,] = c(ni, W22(CC, CC_spin_obj[[1]]))
  print(paste('radgp: ', iexp,'th finished'))
}

## fitting theoretical upper bound
upbound <- function(n, const=1){
  return(const * n / (1+n^(1/2))^3)
}

bound_lst = rep(0,length(n_lst))
for (i in 1:length(n_lst)){
  bound_lst[i] = upbound(n_lst[i])
}

out_c = max(W22_mat_p[,2]/bound_lst)
bound_lst = bound_lst * out_c

## plot results
W22_dat_p = as.data.frame(W22_mat_p)
colnames(W22_dat_p) = c('n', 'W22')
W22_dat_p$Type ='Simulation'
W22_dat_new_p = data.frame(n=n_lst, W22=bound_lst, Type='Theory')
W22_dat_p = rbind(W22_dat_p, W22_dat_new_p)
p2 = 
ggplot(W22_dat_p) +
  geom_line(aes(x=n,y=W22,color=Type),size=1.5) +
  theme_minimal(base_size = 25) + ylab(bquote(W[2]^2))



##----------------------------paste two plots-------------------------------
plot_grid(p1, p2, align='h', nrow=1, rel_widths=c(0.45,0.55))












