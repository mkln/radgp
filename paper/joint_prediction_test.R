rm(list=ls())
library(tidyverse)
library(magrittr)
library(Matrix)
#library(reticulate)
library(radgp)
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

fun_gamma <- function(v){
  return(mean(gamma(v)))
}

SW <- function(mat1, mat2, rseed=10, L=500){
  n = nrow(mat1)
  d = ncol(mat1)
  set.seed(rseed)
  directs = matrix(rnorm(L*d),nrow=d,ncol=L)
  directs = t(t(directs) / sqrt(colSums(directs^2)))
  mat1p = mat1 %*% directs
  mat2p = mat2 %*% directs
  S = 0
  for (l in 1:L){
    S = S + discrete_W_1d(mat1p[,l],mat2p[,l])
  }
  S = S/L
  return(S)
}

coverage <- function(zhat_mcmc, ztrue, alpha=0.95){
  ## zhat_mcmc is a n_mcmc\times ntest matrix, ztrue is a ntest length vector
  s = 0
  n_mcmc = dim(zhat_mcmc)[1]
  indices = c(round(n_mcmc*(1-alpha)/2), n_mcmc-round(n_mcmc*(1-alpha)/2))
  ntest = dim(zhat_mcmc)[2]
  for (i in 1:ntest){
    chain_sort = sort(zhat_mcmc[,i])
    if (ztrue[i]>=chain_sort[indices[1]] & ztrue[i]<=chain_sort[indices[2]]){
      s = s + 1
    }
  }
  return(s/n_mcmc)
}

CI <- function(lst,alpha=0.90){
  out = c(mean(lst),0,0)
  n = length(lst)
  lst_sort = sort(lst)
  l = n*(1-alpha)/2
  if (floor(l)!=l){
    l1 = floor(l)
    u1 = n-l1-1
    out[2:3] = c(mean(lst_sort[l1:(l1+1)]), mean(lst_sort[u1:(u1+1)]))
  } else{
    u = n-l
    out[2:3] = c(lst_sort[l], lst_sort[u])
  }
  return(out)
}

##-----------------------------------------------------------------------
set.seed(10)
## Specify global parameter
model='latent'
# model='response'
nexp = 50  ## total number of iterations
fun_lst = c(max, mean, sd, fun_relu, median, fun_max_min, fun_exp, fun_gamma)
fun_label = c('Max', 'Mean', 'Standard Deviation', 'Mean of Relu', 'Median', 'Max_min', 'exp', 'gamma')
if (model=='latent'){
  nugget = 0.01
} else{
  nugget = 0
}
cov_thre = 0.05
x_thre = 0.15
theta = c(19.97, 1, 1, 10^{-5})
theta[1] = - log(cov_thre/theta[2]) / x_thre^theta[3]
plates = matrix(c(0.15,0.25,0.45,0.55,0.75,0.85),nrow=2)
n_region = ncol(plates)^2
W2_fun_tensor = array(rnorm(n_region*3*nexp*length(fun_lst)),dim=c(n_region,3,nexp,length(fun_lst)))
SW_tensor = array(rnorm(n_region*3*nexp),dim=c(n_region,3,nexp))
n_reports = 6 ## report theta[1], theta[2], theta[3], nugget, MSE, coverage
reports_tensor = array(rnorm(3*nexp*n_reports),dim=c(3,nexp,n_reports))
time_ls = matrix(0,nrow=nexp,ncol=3)

# m = 20
# rho = 0.081
m = 8
rho = 0.055
# m = 5
# rho = 0.048

rho_test = rho
for (iexp in 1:nexp){
  region_valid = FALSE
  while (!region_valid){
    region_valid = TRUE
    ##---------------------------------------------------------------------
    ## Generate training from grids and test data uniformly
    nl = 40
    ntrain = nl^2
    ntest = 1000
    nall = ntrain + ntest
    coords_train = as.matrix(expand.grid(xout<-seq(0,1,length.out=nl),xout))
    coords_test = matrix(runif(ntest*2),nrow=(ntest),ncol=2)
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
  CC <- radgp::Correlationc(coords_all, coords_all, theta, TRUE) 
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
  mcmc <- 2500
  theta_start <- c(25, 1, 1)
  theta_unif_bounds <- matrix(nrow=3, ncol=2)
  theta_unif_bounds[1,] <- c(5, 100) # phi
  theta_unif_bounds[2,] <- c(.2, 10) # sigmasq
  # theta_unif_bounds[3,] <- c(0.9, 2-.001) # nu
  theta_unif_bounds[3,] <- c(0.9999,1.0001)
  if (model=='latent'){
    nugget_start <- nugget
  } else{
    nugget_start <- 1e-5
    nugg_bounds <- c(1e-7,1e-4)
  }
  nugget_prior <- c(2, 0.005)
  
  
  if (model=='latent'){
    t0 = as.numeric(Sys.time())
    altdag_model <- latent.model(y_train, coords_train, rho=rho, 
                                 theta_start=theta_start,
                                 theta_prior=theta_unif_bounds,
                                 nugg_start=nugget_start,
                                 nugg_prior=nugget_prior,
                                 mcmc=mcmc, n_threads=16, printn=1)
    t1 = as.numeric(Sys.time())      
  } else{
    t0 = as.numeric(Sys.time())
    altdag_model <- response.model(y_train, coords_train, rho=rho, 
                                   theta_start=theta_start,
                                   theta_prior=theta_unif_bounds,
                                   nugg_start=nugget_start,
                                   nugg_prior=nugget_prior,
                                   nugg_bounds=nugg_bounds,
                                   mcmc=mcmc, n_threads=16, printn=1)
    t1 = as.numeric(Sys.time())  
  }
  
  altdag_predict <- predict(altdag_model, coords_test, rho=rho_test, mcmc_keep=mc_true, n_threads=16)
  t2 = as.numeric(Sys.time())
  
  if (model=='latent'){
    zalt_mc <- t(altdag_predict$wout)    
  } else{
    zalt_mc <- t(altdag_predict$yout)
  }
  t2-t0
  
  ##-------------------------------------------------------------------------
  # vecchia-maxmin estimation MCMC
  if (model=='latent'){
    t3 = as.numeric(Sys.time())
    maxmin_model <- latent.model.vecchia(y_train, coords_train, m=m,
                                         theta_start=theta_start,
                                         theta_prior=theta_unif_bounds,
                                         nugg_start=nugget_start,
                                         nugg_prior=nugget_prior,
                                         mcmc=mcmc, n_threads=16, printn=1)
    t4 = as.numeric(Sys.time())    
  } else{
    t3 = as.numeric(Sys.time())
    maxmin_model <- response.model.vecchia(y_train, coords_train, m=m, 
                                           theta_start=theta_start,
                                           theta_prior=theta_unif_bounds,
                                           nugg_start=nugget_start,
                                           nugg_prior=nugget_prior,
                                           nugg_bounds=nugg_bounds,
                                           mcmc=mcmc, n_threads=16, printn=1)
    t4 = as.numeric(Sys.time())      
  }
  
  # NNGP prediction
  nngp_predict <- predict(maxmin_model, coords_test, mcmc_keep=mc_true, n_threads=16, independent=FALSE)
  t5 = as.numeric(Sys.time())
  inngp_predict <- predict(maxmin_model, coords_test, mcmc_keep=mc_true, n_threads=16, independent=TRUE)
  t6 = as.numeric(Sys.time())
  
  if (model=='latent'){
    znngp_mc <- t(nngp_predict$wout)
    zinngp_mc <- t(inngp_predict$wout)
  } else{
    znngp_mc <- t(nngp_predict$yout)
    zinngp_mc <- t(inngp_predict$yout)    
  }
  reports_tensor[1,iexp,1:3] = apply(altdag_model$theta[,-(1:(mcmc-mc_true))],1,mean)
  reports_tensor[2,iexp,1:3] = apply(maxmin_model$theta[,-(1:(mcmc-mc_true))],1,mean)
  reports_tensor[3,iexp,1:3] = reports_tensor[2,iexp,1:3]
  reports_tensor[,iexp,4] = c(mean(altdag_model$nugg[-(1:(mcmc-mc_true))]), 
                              mean(maxmin_model$nugg[-(1:(mcmc-mc_true))]), 
                              mean(maxmin_model$nugg[-(1:(mcmc-mc_true))]))
  reports_tensor[,iexp,5] = c(mean((apply(zalt_mc,2,mean)-ztrue_mc[1,])^2), 
                              mean((apply(znngp_mc,2,mean)-ztrue_mc[1,])^2), 
                              mean((apply(zinngp_mc,2,mean)-ztrue_mc[1,])^2))
  reports_tensor[,iexp,6] = c(coverage(zalt_mc, ztrue_mc[1,]), 
                              coverage(znngp_mc, ztrue_mc[1,]), coverage(zinngp_mc, ztrue_mc[1,]))

  ##------------------------------------------------------------------------------
  ## print time
  print(paste('Alt DAG training time: ', t1-t0, ', testing time: ', t2-t1, ', total time: ', t2-t0, sep=''))
  print(paste('NNGP training time: ', t4-t3, ', testing time: ', t5-t4, ', total time: ', t5-t3, sep=''))
  print(paste('i-NNGP total time: ', t4-t3, ', testing time: ', t6-t5, ', total time: ', t6-t5+t4-t3, sep=''))
  print(paste('Apt DAG theta mean: '))
  print(reports_tensor[1,iexp,1:3])
  print(paste('Vecchia maxmin theta mean: '))
  print(reports_tensor[2,iexp,1:3])
  print(paste('Apt nugget mean: ', reports_tensor[1,iexp,4], ' vecchia nugget mean: ', reports_tensor[2,iexp,4], sep=''))
  time_ls[iexp,] = c(t2-t0,t5-t3,t6-t5+t4-t3)
  
  ##----------------------------------------------------------------------------
  ## Compare true mc samples with altdag mcmc samples in local regions
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
      }
    SW_tensor[ind,,iexp] = c(SW(ztrue_mc[,region_lst[[ind]]],zalt_mc[,region_lst[[ind]]]),
                             SW(ztrue_mc[,region_lst[[ind]]],znngp_mc[,region_lst[[ind]]]),
                             SW(ztrue_mc[,region_lst[[ind]]],zinngp_mc[,region_lst[[ind]]]))
    }
  }
  print(paste(iexp, 'th outer loop finished', sep=''))
}  





reports_summary = array(0, dim=c(3,n_reports,3))  ## the last dimension represents mean and lower & upper confidence bound
for (i in 1:3){
  for (j in 1:n_reports){
    reports_summary[i,j,]=CI(reports_tensor[i,,j])
  }
}

fun_summary = array(0, dim=c(3,2,length(fun_lst)+1))
for (l in 1:3){
  for (k in 1:length(fun_lst)){
    fun_samples = as.vector(W2_fun_tensor[,l,,k])
    fun_summary[l,1,k] = mean(fun_samples)
    fun_summary[l,2,k] = sd(fun_samples)
  }
  fun_summary[l,1,length(fun_lst)+1] = mean(SW_tensor[,l,])
  fun_summary[l,2,length(fun_lst)+1] = sd(SW_tensor[,l,])  
}

fun_dat <- data.frame(matrix(0,nrow=n_region*3*nexp*(length(fun_lst)+1),ncol=3))
colnames(fun_dat) <- c('W2_value', 'Test_fun', 'Method')
method_names = c('RadGP','V-Pred','NNGP')
loc = 0
for (i in 1:3){
  gapi = n_region*nexp*(length(fun_lst)+1)
  fun_dat$Method[((i-1)*gapi+1):(i*gapi)] = method_names[i]
  for (j in 1:length(fun_lst)){
    gapj = n_region*nexp
    fun_dat$Test_fun[(loc+1):(loc+gapj)] = fun_label[j]
    fun_dat$W2_value[(loc+1):(loc+gapj)] = as.vector(W2_fun_tensor[,i,,j])
    loc = loc + gapj
  }
  gapj = n_region*nexp
  fun_dat$Test_fun[(loc+1):(loc+gapj)] = 'Sliced W2'
  fun_dat$W2_value[(loc+1):(loc+gapj)] = as.vector(SW_tensor[,i,])
  loc = loc + gapj  
}

print('End of running.')
print(paste('time:', mean(time_ls[,1]), mean(time_ls[,2]), mean(time_ls[,3])))
print(paste('m:', m, ', rho:', rho, ', nugget:', nugget, sep=''))
print('aptdag summary')
print(reports_summary[1,,])
print('vecchia maxmin summary')
print(reports_summary[2,,])
print('vecchia maxmin pred summary')
print(reports_summary[3,,])

fun_label_select = fun_label[2:5]
fun_label_select = c("Mean","Mean of Relu","Median",'Sliced W2')
fun_dat_select = fun_dat[which(fun_dat$Test_fun %in% fun_label_select),]
ggplot(fun_dat_select) +
  geom_boxplot(aes(y=W2_value,x=Method,fill=Method)) + ggtitle(paste(fun_label[i],sep='')) + ylim(0, 0.1) +
  theme_minimal(base_size = 30) + labs(y='W2 distnace',x=NULL,title=NULL) + 
  theme(plot.title = element_text(hjust = 0.5),legend.position = "none") + facet_wrap(~Test_fun, nrow=1) 

for (k in 1:length(fun_lst)){
  print(paste('fun:', fun_label[[k]]))
  print(fun_summary[,,k])
}
print('SW:')
print(fun_summary[,,length(fun_lst)+1])

## plot sliced W-2 values for each local region
SW_dat <- data.frame(matrix(0,nrow=n_region*3*nexp,ncol=3))
colnames(SW_dat) <- c('W2_value', 'Region', 'Method')
loc = 0
for (i in 1:3){
  gapi = n_region*nexp
  SW_dat$Method[((i-1)*gapi+1):(i*gapi)] = method_names[i]
  for (j in 1:n_region){
    gapj = nexp
    SW_dat$Region[(loc+1):(loc+gapj)] = j
    SW_dat$W2_value[(loc+1):(loc+gapj)] = SW_tensor[j,i,]
    loc = loc + gapj
  }
}
# ggplot(SW_dat) +
#   geom_boxplot(aes(y=W2_value,x=Method),fill='lightgoldenrod1') + facet_wrap(~Region, nrow=3) 
#   theme_minimal(base_size = 30) + 
#   theme(plot.title = element_text(hjust = 0.5)) + 

# ggplot(SW_dat) +
#   geom_boxplot(aes(y=W2_value,x=Method),fill='lightgoldenrod1')
  
ggplot(SW_dat[which(SW_dat$Region<4),]) +
  geom_boxplot(aes(y=W2_value,x=Method,fill=Method)) + facet_wrap(~Region, nrow=1) +
  labs(y='Sliced W2',x=NULL,title=NULL) +  theme_minimal(base_size = 30) +
  scale_colour_manual(values=c("magenta", "#0067f7", 'orange')) +
  theme(strip.background = element_blank(),strip.text.x = element_blank(),legend.position="none") 





  

















# plots_lst = vector('list',length(fun_lst))
# for (i in 1:length(fun_lst)){
#   plots_lst[[i]] <- ggplot(fun_dat[which(fun_dat$Test_fun==fun_label[i]),]) +
#     geom_boxplot(aes(y=W2_value,x=Method)) + ggtitle(paste(fun_label[i],sep='')) + 
#     theme_minimal(base_size = 18) + labs(y=NULL,x=NULL) +
#     theme(plot.title = element_text(hjust = 0.5))
# }
# plots_lst[[2]] <- ggplot(fun_dat[which(fun_dat$Test_fun==fun_label[1]),]) +
#   geom_boxplot(aes(y=W2_value,x=Method)) + ggtitle(paste(fun_label[1],sep='')) + 
#   theme_minimal(base_size = 18) + labs(y='W2 distnace',x=NULL) + 
#   theme(plot.title = element_text(hjust = 0.5))
# grid.arrange(plots_lst[[2]], plots_lst[[3]], plots_lst[[4]], plots_lst[[5]], nrow=1)
# grid.arrange(plots_lst[[1]], plots_lst[[2]], plots_lst[[3]], plots_lst[[4]], 
#              plots_lst[[5]], plots_lst[[6]], plots_lst[[7]], plots_lst[[8]], nrow=2)
























##-----------------------------------------------------------------------------
## compute strength of signal

# rad = 0.05
# sigcov = theta[2]*exp(-theta[1]*rad^theta[3])
# sig = 2*theta[2]-2*sigcov


