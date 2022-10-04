rm(list=ls())
library(tidyverse)
library(magrittr)
library(Matrix)
#library(reticulate)
library(aptdag)
library(ggplot2)
library(gridExtra)
library(RcppHungarian)

##--------------------------------------------------------------------
submatrix_indices <- function(coords, range1, range2){
  sub = which(coords[,1]>=range1[1] & coords[,1]<=range1[2] & coords[,2]>=range2[1] & coords[,2]<=range2[2])
  return(sub)
}
rowMaxs <- function(mat){
  n = nrow(mat)
  colm = vector('numeric',n)
  for (i in 1:n){
    colm[i] = max(mat[i,])
  }
  return(colm)
}
nngp_pre <- function(coords, theta, m){
  ixmm <- GPvecchia::order_maxmin_exact(coords)
  coords_mm <- coords[ixmm,]
  nn_dag_mat <- GPvecchia:::findOrderedNN_kdtree2(coords_mm, m)
  nn_dag <- apply(nn_dag_mat, 1, function(x){ x[!is.na(x)][-1]-1 })
  vecchia_out <- vecchiagp(coords_mm, theta, nn_dag)
  Hord <- vecchia_out$H
  ord <- order(ixmm)
  Hmat_v <- Hord[ord, ord]
  precision <- crossprod(Hmat_v)
  return(as.matrix(precision))
}

nngp_cov <- function(coords, theta, m){
  ixmm <- GPvecchia::order_maxmin_exact(coords)
  coords_mm <- coords[ixmm,]
  nn_dag_mat <- GPvecchia:::findOrderedNN_kdtree2(coords_mm, m)
  nn_dag <- apply(nn_dag_mat, 1, function(x){ x[!is.na(x)][-1]-1 })
  vecchia_out <- vecchiagp(coords_mm, theta, nn_dag)
  Hord <- vecchia_out$H
  ord <- order(ixmm)
  Hmat_v <- Hord[ord, ord]
  precision <- crossprod(Hmat_v)
  n_nonzero <- length(precision@i)
  vecchia_cov <- solve(precision)
  # chol_cov <- solve(Hmat_v)
  # vecchia_cov <- tcrossprod(chol_cov)
  return(list('cov'=as.matrix(vecchia_cov), 'n_nonzero'=n_nonzero))
}

discrete_W <- function(path1, path2, order=2){
  n = nrow(path1)
  mat = matrix(0,nrow=n,ncol=n)
  for (i in 1:n){
    for (j in 1:n){
      mat[i,j] = sum(abs((path1[i,]-path2[j,]))^order)
    }
  }
  W = (HungarianSolver(mat)$cost/n)^(1/order)
  return(W)
}

discrete_W_1d <- function(path1, path2, order=2){
  sort1 = sort(path1)
  sort2 = sort(path2)
  n = length(path1)
  W = (sum((abs(sort1-sort2))^order)/n)^(1/order)
  return(W)
}

fun_max_min <-function(v){
  return(max(v)-min(v))
}

fun_thre_portion <- function(v,thre=1){
  return(sum(v>thre)/length(v))
}

fun_m2 <- function(v){
  return(mean(v^2))
}

fun_relu <- function(v){
  return(mean(pmax(v,0)))
}

fun_exp <- function(v){
  return(mean(exp(v)))
}

##-----------------------------------------------------------------------
## Specify global parameter
model='latent'
# model='response'
nexp = 2  ## total number of iterations
fun_lst = c(max, mean, sd, fun_relu, median, fun_max_min)
fun_label = c('max', 'mean', 'sd', 'relu', 'median', 'fun_max_min')
if (model=='latent'){
  nugget = 0.1
} else{
  nugget = 0
}
cov_thre = 0.05
x_thre = 0.15
theta = c(15, 1, 1, 10^{-5})
theta[1] = - log(cov_thre/theta[2]) / x_thre^theta[3]
plates = matrix(c(0.15,0.25,0.45,0.55,0.75,0.85),nrow=2)
n_region = ncol(plates)^2
W2_fun_tensor = array(rnorm(n_region*3*nexp*length(fun_lst)),dim=c(n_region,3,nexp,length(fun_lst)))
time_ls = matrix(0,nrow=nexp,ncol=3)

for (iexp in 1:nexp){
  region_valid = FALSE
  while (!region_valid){
    region_valid = TRUE
    ##---------------------------------------------------------------------
    ## Generate training from grids and test data uniformly
    nl = 30
    coords_train = as.matrix(expand.grid(xout<-seq(0,1,length.out=nl),xout))
    coords_test = matrix(runif(nl^2*2),nrow=(nl^2),ncol=2)
    ntrain = nrow(coords_train)
    ntest = nrow(coords_test)
    nall = ntrain + ntest
    coords_all = rbind(coords_train, coords_test)

    #-------------------------------------------------------------------------
    ## set local regions
    region_lst = vector(mode='list',length=n_region)
    n_select = 0
    for (i in 1:ncol(plates)){
      if (!region_valid){
        break
      }
      for (j in 1:ncol(plates)){
        ind = ceiling((i-1)*ncol(plates)+j)
        region_lst[[ind]] = submatrix_indices(coords_test,plates[,i],plates[,j])
        if (length(region_lst[[ind]])<2){
          region_valid = FALSE
          break
        }
        n_select = length(region_lst[[ind]]) + n_select
      }
    }
  }
  
  #-------------------------------------------------------------------------
  # gen data from full GP
  # theta : phi, sigmasq, nu, nugg
  CC <- aptdag::Correlationc(coords_all, coords_all, theta, TRUE) 
  CC_train <- CC[1:ntrain,1:ntrain]
  CC_train_inv <- solve(CC_train)
  LC_train <- t(chol(CC_train))
  z_train <- LC_train %*% rnorm(ntrain) 
  y_train = z_train + nugget^.5 * rnorm(ntrain)
  CC_test <- CC[(ntrain+1):nall,(ntrain+1):nall]
  CC_testtrain <- CC[(ntrain+1):nall,1:ntrain]
  CC_cond <- CC_test - CC_testtrain %*% CC_train_inv %*% t(CC_testtrain)
  LC_cond <- t(chol(CC_cond))
  mc_true <- 1000
  ztrue_mc = matrix(nrow=mc_true,ncol=ntest)
  ## ztrue_mc is a ntest*mc_true matrices of joint test samples
  for (i in 1:mc_true){
    ztrue_mc[i,] = LC_cond %*% rnorm(ntest) + CC_testtrain %*% CC_train_inv %*% y_train
  }
  
  ##------------------------------------------------------------------ aptdag model 
  mcmc <- 5000
  unif_bounds <- matrix(nrow=3, ncol=2)
  unif_bounds[1,] <- c(1, 300) # phi
  unif_bounds[2,] <- c(.1, 10) # sigmasq
  unif_bounds[3,] <- c(0.75, 2-.001) # nu
  nugg_bounds <- c(0.5*1e-5, 1.5*1e-5) # nugget for response model
  theta_start <- c(50, 1, 1.5)
  
  rho = 0.050
  rho_test = rho
  
  if (model=='latent'){
    t0 = Sys.time()
    altdag_model <- latent.model(y_train, coords_train, theta_start=theta_start,
                                   theta_prior=unif_bounds, rho=rho, mcmc=mcmc, n_threads=16, printn=0)
    t1 = Sys.time()      
  } else{
    t0 = Sys.time()
    altdag_model <- response.model(y_train, coords_train, theta_start=theta_start,
                                   theta_prior=unif_bounds, nugg_prior=nugg_bounds, rho=rho, mcmc=mcmc, n_threads=16, printn=0)
    t1 = Sys.time()  
  }
  
  altdag_predict <- predict(altdag_model, coords_test, rho=rho_test, mcmc_keep=mc_true, n_threads=16)
  t2 = Sys.time()
  
  if (model=='latent'){
    zalt_mc <- t(altdag_predict$wout)    
  } else{
    zalt_mc <- t(altdag_predict$yout)
  }
  t2-t0
  
  # m_alt = mean(sapply(altdag_model$dag, length))
  # m_alt_test = mean(sapply(altdag_predict$predict_dag, length))
  # c(m_alt,m_alt_test)

  ##-------------------------------------------------------------------------
  ## nngp model
  # vecchia-maxmin estimation MCMC
  m = 6
  if (model=='latent'){
    t3 = Sys.time()
    maxmin_model <- latent.model.vecchia(y_train, coords_train, m=m, theta_start=theta_start,
                                           theta_prior=unif_bounds,
                                           mcmc=mcmc, n_threads=16, printn=0)
    t4 = Sys.time()    
  } else{
    t3 = Sys.time()
    maxmin_model <- reponse.model.vecchia(y_train, coords_train, m=m, theta_start=theta_start,
                                         theta_prior=unif_bounds, nugg_prior=nugg_bounds,
                                         mcmc=mcmc, n_threads=16, printn=0)
    t4 = Sys.time()      
  }

  # NNGP prediction
  nngp_predict <- predict(maxmin_model, coords_test, mcmc_keep=mc_true, n_threads=16, independent=FALSE)
  t5 = Sys.time()
  inngp_predict <- predict(maxmin_model, coords_test, mcmc_keep=mc_true, n_threads=16, independent=TRUE)
  t6 = Sys.time()

  if (model=='latent'){
    znngp_mc <- t(nngp_predict$wout)
    zinngp_mc <- t(inngp_predict$wout)
  } else{
    znngp_mc <- t(nngp_predict$yout)
    zinngp_mc <- t(inngp_predict$yout)    
  }
  
  ##------------------------------------------------------------------------------
  ## print time
  # print(c(m_alt, m_alt_test))
  print(paste('Alt DAG training time: ', t1-t0, ', testing time: ', t2-t1, ', total time: ', t2-t0, sep=''))
  print(paste('NNGP training time: ', t4-t3, ', testing time: ', t5-t4, ', total time: ', t5-t3, sep=''))
  print(paste('i-NNGP total time: ', t4-t3, ', testing time: ', t6-t5, ', total time: ', t6-t5+t4-t3, sep=''))
  print(paste('Apt DAG theta mean: ', apply(altdag_model$theta[,-(1:(mcmc-mc_true))],1,mean)))
  print(paste('Vecchia maxmin theta mean: ', apply(maxmin_model$theta[,-(1:(mcmc-mc_true))],1,mean)))
  time_ls[iexp,] = c(t2-t0,t5-t3,t6-t5+t4-t3)
  
  ##----------------------------------------------------------------------------

  
  ##----------------------------------------------------------------------------
  ## Compare true mc samples with altdag mcmc samples on maxima of local regions
  dat_fun <- data.frame(matrix(0,nrow=mc_true*n_region*4,ncol=3))
  colnames(dat_fun) <- c('region', 'model', 'maximum')
  W2_fun <- matrix(0,nrow=n_region,ncol=3)
  loc = 0
  for (i in 1:ncol(plates)){
    for (j in 1:ncol(plates)){
      for (k in 1:length(fun_lst)){
        fun = fun_lst[[k]]
        ind = ceiling((i-1)*ncol(plates)+j)
        samples_true = apply(ztrue_mc[,region_lst[[ind]]],1,fun)
        samples_alt = apply(zalt_mc[,region_lst[[ind]]],1,fun)
        samples_nngp = apply(znngp_mc[,region_lst[[ind]]],1,fun)
        samples_inngp = apply(zinngp_mc[,region_lst[[ind]]],1,fun)
        W2_fun_tensor[ind,,iexp,k] = c(discrete_W_1d(samples_true, samples_alt), 
                                       discrete_W_1d(samples_true, samples_nngp), 
                                       discrete_W_1d(samples_true, samples_inngp))        
        # dat_fun[((loc+1):(loc+4*mc_true)),1] = ind
        # dat_fun[((loc+1):(loc+mc_true)),2] = 'true'
        # dat_fun[((loc+1+mc_true):(loc+2*mc_true)),2] = 'altdag'
        # dat_fun[((loc+1+2*mc_true):(loc+3*mc_true)),2] = 'nngp'
        # dat_fun[((loc+1+3*mc_true):(loc+4*mc_true)),2] = 'inngp'
        # dat_fun[((loc+1):(loc+mc_true)),3] = apply(ztrue_mc[,region_lst[[ind]]],1,fun)
        # dat_fun[((loc+1+mc_true):(loc+2*mc_true)),3] = apply(zalt_mc[,region_lst[[ind]]],1,fun)
        # dat_fun[((loc+1+2*mc_true):(loc+3*mc_true)),3] = apply(znngp_mc[,region_lst[[ind]]],1,fun)
        # dat_fun[((loc+1+3*mc_true):(loc+4*mc_true)),3] = apply(zinngp_mc[,region_lst[[ind]]],1,fun)
      }
      # loc = loc + mc_true*4
    }
  }
  
  ##----------------------------------------------------------------------------
  # p_region_max_lst = vector('list',n_region)
  # for (i in 1:n_region){
  #   p_region_max_lst[[i]] <- ggplot(dat_fun[which(dat_fun$region==i),]) +
  #     geom_boxplot(aes(y=maximum,x=model))
  # }
  # grid.arrange(p_region_max_lst[[1]],p_region_max_lst[[2]],p_region_max_lst[[3]],p_region_max_lst[[4]],
  #              p_region_max_lst[[5]],p_region_max_lst[[6]],p_region_max_lst[[7]],p_region_max_lst[[8]],
  #              p_region_max_lst[[9]],nrow=3)
  # 
  # p_region_max_W2_lst = vector('list',n_region)
  # for (i in 1:n_region){
  #   p_region_max_W2_lst[[i]] <- ggplot(data.frame(model=c('altdag','nngp','inngp'),W2_fun=W2_fun[i,])) +
  #     geom_col(aes(y=W2_fun,x=model))
  # }
  # grid.arrange(p_region_max_W2_lst[[1]],p_region_max_W2_lst[[2]],p_region_max_W2_lst[[3]],p_region_max_W2_lst[[4]],
  #              p_region_max_W2_lst[[5]],p_region_max_W2_lst[[6]],p_region_max_W2_lst[[7]],p_region_max_W2_lst[[8]],
  #              p_region_max_W2_lst[[9]],nrow=3)
  print(paste(iexp, 'th outer loop finished', sep=''))
}  

fun_summary = array(0, dim=c(3,2,length(fun_lst)))
for (l in 1:3){
  for (k in 1:length(fun_lst)){
    fun_samples = as.vector(W2_fun_tensor[,l,,k])
    fun_summary[l,1,k] = mean(fun_samples)
    fun_summary[l,2,k] = sd(fun_samples)
  }
}


print(paste('time:', mean(time_ls[,1]), mean(time_ls[,2]), mean(time_ls[,3])))
print(paste('theta:',theta[1], theta[2], theta[3], theta[4]))
for (k in 1:length(fun_lst)){
  print(paste('fun:', fun_label[[k]]))
  print(fun_summary[,,k])
}















