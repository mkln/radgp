library(radgp)
library(ggplot2)
library(transport)
library(rSPDE)
library(MASS)
library(reshape2)
library(latex2exp)

##--------------------------------------------------------------------
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

ave_neighbors <- function(cov, dag){
  n = nrow(cov)
  adjL = matrix(0,nrow=n,ncol=n)
  for (i in 1:n){
    ni = dag[[i]]
    if (length(ni)>0){
      adjL[ni,i] = 1
    }
  }
  adj = adjL %*% adjL
  return(sum(adj!=0)/n)
}


##-----------------------------------------------------------------------
t0 = Sys.time()
# set.seed(77)
set.seed(66)
mc_true = 1000
mcmc = 2500
n_repeat = 50

## Matern covariance function
nu_matern = 3/2
theta = c(20, 1, nu_matern, 0)
dist_thre = 0.25
matern_hint_fun <- function(phi, dist=0.15, sigmasq=1, nu=nu_matern, thre=dist_thre){
  return( thre - spaMM::MaternCorr(dist, phi, smoothness=nu, nu=nu, Nugget = NULL) )
} 
theta[1] = dichotomy_solver(matern_hint_fun, 10, 100)



## nugget for latent model
nugget <- 1e-2

## training data
nl = 50
ntrain = nl^2
coords_train = as.matrix(expand.grid(xout<-seq(0,1,length.out=nl),xout))
CC_train <- radgp::Correlationc(coords_train, coords_train, theta, 1, T)  ## 0 for power exponential, anything else for matern
CC_train_inv <- ginv(CC_train)
eigenobj_train = eigen(CC_train,symmetric=TRUE)
D_train = pmax(eigenobj_train$values,0)
U_train = eigenobj_train$vectors
LC_train <- U_train %*% diag(sqrt(D_train)) %*% t(U_train)

## calculate information only relavent to training data
rho_lst = c(0.035, 0.045, 0.054, 0.062, 0.070, 0.078, 0.086, 0.094)
m_lst = c(3, 5, 7, 9, 12, 15, 18, 22)
ave_nonzeros = vector('numeric', length=length(rho_lst)+length(m_lst))
for (j in 1:length(rho_lst)){
  rho = rho_lst[j]
  rad_Nsets = radialndag(coords_train, rho)
  for (i in 1:ntrain){
    rad_Nsets[[i]] = rad_Nsets[[i]] + 1
  }
  ave_nonzeros[j] = ave_neighbors(CC_train, rad_Nsets)
}
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
  ave_nonzeros[length(rho_lst)+j] = ave_neighbors(CC_train, dag_nn)
}

t1 = Sys.time()
print('head finished')
print(t1-t0)


##---------------------------repeated experiments------------------------------
results_tensor = array(0, dim=c(length(rho_lst)+2*length(m_lst), 1, n_repeat))

for (i_repeat in 1:n_repeat){
  
  ## uniform test data
  ntest = 1000
  nall = ntrain + ntest
  coords_test = matrix(runif(ntest*2), nrow=ntest, ncol=2)
  coords_all = rbind(coords_train, coords_test)
  
  ## generate joint test samples from the true conditional distribution
  CC <- radgp::Correlationc(coords_all, coords_all, theta, 1, T)
  z_train <- LC_train %*% rnorm(ntrain) 
  y_train = z_train + nugget^.5 * rnorm(ntrain)
  CC_test <- CC[(ntrain+1):nall,(ntrain+1):nall]
  CC_testtrain <- CC[(ntrain+1):nall,1:ntrain]
  CC_cond <- CC_test - CC_testtrain %*% CC_train_inv %*% t(CC_testtrain)
  eigenobj_cond = eigen(CC_cond,symmetric=TRUE)
  D_cond = pmax(eigenobj_cond$values,0)
  U_cond = eigenobj_cond$vectors
  LC_cond <- U_cond %*% diag(sqrt(D_cond)) %*% t(U_cond)
  ztrue_mc = matrix(nrow=mc_true,ncol=ntest)
  ## ztrue_mc is a mc_true*ntest matrices of joint test samples
  for (i in 1:mc_true){
    ztrue_mc[i,] = LC_cond %*% rnorm(ntest) + CC_testtrain %*% CC_train_inv %*% z_train
  }

  ##----------------------------------mcmc----------------------------------------
  theta_start <- c(40, 1, theta[3])
  theta_unif_bounds <- matrix(nrow=3, ncol=2)
  theta_unif_bounds[1,] <- c(1, 100) # phi
  theta_unif_bounds[2,] <- c(.1, 10) # sigmasq
  theta_unif_bounds[3,] <- c(theta[3]-0.001, theta[3]+0.001) # nu
  nugget_start <- c(.01)
  nugget_prior <- c(1, 0.01)

  ## radgp
  t2 = Sys.time()
  
  for (j in 1:length(rho_lst)){
    rho = rho_lst[j]
    radgp_obj <- latent.model(y_train, coords_train, rho,
                                     theta_start=theta_start,
                                     theta_prior=theta_unif_bounds,
                                     nugg_start=nugget,
                                     nugg_prior=nugget_prior,
                                     mcmc=mcmc, n_threads=16, covariance_model = 1, printn=3)
    radgp_predict_obj = predict(radgp_obj, coords_test, mcmc_keep=mc_true, n_threads=16)
    radgp_pred <- t(radgp_predict_obj$wout)
    results_tensor[j,1,i_repeat] = wasserstein(pp(ztrue_mc), pp(radgp_pred), p=2)
    
    print(paste('Inner loop finished: ', j, '/', length(rho_lst)+length(m_lst), sep=''))
    print(Sys.time() - t2)
    t2 = Sys.time()
  }
  
  ## nngp
  for (j in 1:length(m_lst)){
    m = m_lst[j]
    nngp_obj <- latent.model.vecchia(y_train, coords_train, m,
                                           theta_start=theta_start,
                                           theta_prior=theta_unif_bounds,
                                           nugg_start=nugget,
                                           nugg_prior=nugget_prior,
                                           mcmc=mcmc, n_threads=16, covariance_model = 1, printn=3)
    nngp_predict_obj = predict(nngp_obj, coords_test, mcmc_keep=mc_true, n_threads=16, independent=FALSE)
    nngp_pred = t(nngp_predict_obj$wout)
    nngp_predict_i_obj = predict(nngp_obj, coords_test, mcmc_keep=mc_true, n_threads=16, independent=TRUE)
    nngp_pred_i <- t(nngp_predict_i_obj$wout)
    results_tensor[length(rho_lst)+j,1,i_repeat] =  wasserstein(pp(ztrue_mc), pp(nngp_pred), p=2)
    results_tensor[length(rho_lst)+length(m_lst)+j,1,i_repeat] =  wasserstein(pp(ztrue_mc), pp(nngp_pred_i), p=2)
    
    print(paste('Inner loop finished: ', length(rho_lst)+j, '/', length(rho_lst)+length(m_lst), sep=''))
    print(Sys.time() - t2)
    t2 = Sys.time()
  }
  
  print(paste('Out loop finished: ', i_repeat, '/', n_repeat, sep=''))
  print(Sys.time() - t1)
  t1 = Sys.time()
}


print('total time: ')
print(Sys.time() - t0)



##-------------------------------- plot results --------------------------------
df = data.frame(matrix(0,nrow=(length(m_lst)*2+length(rho_lst)), ncol=4))
colnames(df) <- c('Nonzeros', 'W22', 'W22low', 'W22up')
df$Region = 1
df$Method = 'RadGP'
l = 1
alpha = 0.1
left = floor(n_repeat*alpha/2)
right = floor(n_repeat-n_repeat*alpha/2)
for (j in 1:length(rho_lst)){
  samples = results_tensor[j,1,]
  samples = sort(samples)
  df$Nonzeros[l] = ave_nonzeros[j]
  df$Method[l] = 'RadGP'
  df$W22low[l] = mean(samples[left:(left+1)])
  df$W22up[l] = mean(samples[right:(right+1)])
  df$W22[l] = mean(samples)
  l = l + 1
}
for (j in 1:length(m_lst)){
  samples = results_tensor[length(rho_lst)+j,1,]
  samples = sort(samples)
  df$Nonzeros[l] = ave_nonzeros[length(rho_lst)+j]
  df$Method[l] = 'V-Pred'
  df$W22low[l] = mean(samples[left:(left+1)])
  df$W22up[l] = mean(samples[right:(right+1)])
  df$W22[l] = mean(samples)
  l = l + 1
}
for (j in 1:length(m_lst)){
  samples = results_tensor[length(rho_lst)+length(m_lst)+j,1,]
  samples = sort(samples)
  df$Nonzeros[l] = ave_nonzeros[length(rho_lst)+j]
  df$Method[l] = 'NNGP'
  df$W22low[l] = mean(samples[left:(left+1)])
  df$W22up[l] = mean(samples[right:(right+1)])
  df$W22[l] = mean(samples)
  l = l + 1
}


ggplot(df) +
  geom_line(aes(x=Nonzeros,y=W22,color=Method)) +
  geom_ribbon(aes(x=Nonzeros,ymin=W22low,ymax=W22up,fill=Method),alpha=0.2) +
  xlab('Precision Nonzeros') + ylab(TeX('$W_2^2$')) + theme_minimal(base_size = 25)










