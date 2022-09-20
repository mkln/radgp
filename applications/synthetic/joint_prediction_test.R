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
nexp = 50  ## total number of iterations
fun_lst = c(max, mean, sd, fun_relu, median, fun_max_min)
fun_label = c('max', 'mean', 'sd', 'relu', 'median', 'fun_max_min')
theta <- c(15, 1, 1, 10^(-3))
cov_thre = 0.05
x_thre = 0.15
theta[1] = - log(cov_thre/theta[2]) / x_thre^theta[3]
# rho_thre = (-log(cov_thre/theta[2]) / theta[1])^(1/theta[3])
## define the local regions
# plates = matrix(c(0.125,0.275,0.425,0.575,0.725,0.875),nrow=2)
plates = matrix(c(0.15,0.25,0.45,0.55,0.75,0.85),nrow=2)
# plates = matrix(c(0.15,0.25),nrow=2)
n_region = ncol(plates)^2
W2_fun_tensor = array(rnorm(n_region*3*nexp*length(fun_lst)),dim=c(n_region,3,nexp,length(fun_lst)))
time_ls = matrix(0,nrow=nexp,ncol=3)

for (iexp in 1:nexp){
  region_valid = FALSE
  while (!region_valid){
    region_valid = TRUE
    ##---------------------------------------------------------------------
    ## Generate training data uniformly and test data from grids
    # nl = 20
    # coords_all0 = as.matrix(expand.grid(xout<-seq(0,1,length.out=2*nl+1),xout))
    # nall = (nl*2+1)^2
    # coords_train = coords_all0[seq(1,nall,2),]
    # coords_test = coords_all0[seq(2,nall,2),]
    # ntrain = nrow(coords_train)
    # ntest = nrow(coords_test)
    # rm(coords_all0)
    # coords_all = rbind(coords_train, coords_test)
    
    nl = 30
    coords_train = as.matrix(expand.grid(xout<-seq(0,1,length.out=nl),xout))
    coords_test = matrix(runif(nl^2*2),nrow=(nl^2),ncol=2)
    ntrain = nrow(coords_train)
    ntest = nrow(coords_test)
    nall = ntrain + ntest
    coords_all = rbind(coords_train, coords_test)
    
    # set.seed(10)
    # ntrain <- 1000
    # coords_train <- runif(ntrain * 2) %>% matrix(ncol=2)
    # coords_test <- expand.grid(xout <- seq(0,1,length.out=25), xout) %>% as.matrix()
    # ntest <- nrow(coords_test)
    # coords_all <- rbind(coords_train, coords_test)
    # nall <- nrow(coords_all)
    
    # nl = 20
    # coords_x = seq(0,1,length.out=2*nl-1)
    # coords_xtrain = coords_x[seq(1,2*nl-1,2)]
    # coords_xtest = coords_x[seq(2,2*nl-2,2)]
    # ntrain = nl^2
    # ntest = (nl-1)^2
    # nall = ntrain + ntest
    # coords_train <- expand.grid(coords_xtrain, coords_xtrain) %>% as.matrix()
    # coords_test <- expand.grid(coords_xtest, coords_xtest) %>% as.matrix()
    # coords_all = rbind(coords_train, coords_test)
    
    # nl = 26
    # ntrain = nl^2
    # nall = (2*nl-1)^2
    # ntest = nall - ntrain
    # coords_train <- expand.grid(xout<-seq(0,1,length.out=nl), xout) %>% as.matrix()
    # coords_x = seq(0,1,length.out=2*nl-1)
    # coords_test <- matrix(0, nrow=ntest, ncol=2)
    # loc = 1
    # for (i in 1:(2*nl-1)){
    #   for (j in 1:(2*nl-1)){
    #     if (i%%2==0 | j%%2==0){
    #       coords_test[loc,] = c(coords_x[i],coords_x[j])
    #       loc = loc + 1
    #     }
    #   }
    # }
    # coords_all = rbind(coords_train, coords_test)
    
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
  y_train <- LC_train %*% rnorm(ntrain) 
  CC_test <- CC[(ntrain+1):nall,(ntrain+1):nall]
  CC_testtrain <- CC[(ntrain+1):nall,1:ntrain]
  CC_cond <- CC_test - CC_testtrain %*% CC_train_inv %*% t(CC_testtrain)
  LC_cond <- t(chol(CC_cond))
  mc_true <- 1000
  ytrue_mc = matrix(nrow=mc_true,ncol=ntest)
  ## ytrue_mc is a ntest*mc_true matrices of joint test samples
  for (i in 1:mc_true){
    ytrue_mc[i,] = LC_cond %*% rnorm(ntest) + CC_testtrain %*% CC_train_inv %*% y_train
  }
  
  ##------------------------------------------------------------------ altdag model 
  mcmc <- 2500
  unif_bounds <- matrix(nrow=4, ncol=2)
  unif_bounds[1,] <- c(1, 300) # phi
  unif_bounds[2,] <- c(.1, 10) # sigmasq
  unif_bounds[3,] <- c(0.75, 2-.001) # nu
  unif_bounds[4,] <- c(1e-5, 1e-1) # nugget
  theta_start <- c(50, 1, 1.5, 1e-4)
  
  rho = 0.050
  rho_test = rho
  
  t0 = Sys.time()
  # using the last rho value used in the preliminary step
  altdag_model <- response.model(y_train, coords_train, theta_start=theta_start,
                                 unif_bounds=unif_bounds, rho=rho, mcmc=mcmc, n_threads=16, printn=0)
  t1 = Sys.time()
  
  # AltDAG prediction
  altdag_predict <- predict(altdag_model, coords_test, rho=rho_test, mcmc_keep=mc_true, n_threads=16)
  t2 = Sys.time()
  # yalt_mc is a 1000*ntest matrix of posterior predictive draws
  yalt_mc <- t(altdag_predict$yout) # retain the last mc_true iterations
  
  m_alt = mean(sapply(altdag_model$dag, length))
  m_alt_test = mean(sapply(altdag_predict$predict_dag, length))
  t2-t0
  c(m_alt,m_alt_test)

  ##-------------------------------------------------------------------------
  ## nngp model
  # vecchia-maxmin estimation MCMC
  m = 6
  t3 = Sys.time()
  maxmin_model <- response.model.vecchia(y_train, coords_train, m=m, theta_start=theta_start,
                                         unif_bounds=unif_bounds,
                                         mcmc=mcmc, n_threads=16, printn=0)
  t4 = Sys.time()
  # NNGP prediction
  nngp_predict <- predict(maxmin_model, coords_test, mcmc_keep=mc_true, n_threads=16, independent=FALSE)
  t5 = Sys.time()
  inngp_predict <- predict(maxmin_model, coords_test, mcmc_keep=mc_true, n_threads=16, independent=TRUE)
  t6 = Sys.time()

  ynngp_mc <- t(nngp_predict$yout)
  yinngp_mc <- t(inngp_predict$yout)
  
  ##------------------------------------------------------------------------------
  ## print time
  print(c(m_alt, m_alt_test))
  print(paste('Alt DAG training time: ', t1-t0, ', testing time: ', t2-t1, ', total time: ', t2-t0, sep=''))
  print(paste('NNGP training time: ', t4-t3, ', testing time: ', t5-t4, ', total time: ', t5-t3, sep=''))
  print(paste('i-NNGP total time: ', t4-t3, ', testing time: ', t6-t5, ', total time: ', t6-t5+t4-t3, sep=''))
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
        samples_true = apply(ytrue_mc[,region_lst[[ind]]],1,fun)
        samples_alt = apply(yalt_mc[,region_lst[[ind]]],1,fun)
        samples_nngp = apply(ynngp_mc[,region_lst[[ind]]],1,fun)
        samples_inngp = apply(yinngp_mc[,region_lst[[ind]]],1,fun)
        W2_fun_tensor[ind,,iexp,k] = c(discrete_W_1d(samples_true, samples_alt), 
                                       discrete_W_1d(samples_true, samples_nngp), 
                                       discrete_W_1d(samples_true, samples_inngp))        
        # dat_fun[((loc+1):(loc+4*mc_true)),1] = ind
        # dat_fun[((loc+1):(loc+mc_true)),2] = 'true'
        # dat_fun[((loc+1+mc_true):(loc+2*mc_true)),2] = 'altdag'
        # dat_fun[((loc+1+2*mc_true):(loc+3*mc_true)),2] = 'nngp'
        # dat_fun[((loc+1+3*mc_true):(loc+4*mc_true)),2] = 'inngp'
        # dat_fun[((loc+1):(loc+mc_true)),3] = apply(ytrue_mc[,region_lst[[ind]]],1,fun)
        # dat_fun[((loc+1+mc_true):(loc+2*mc_true)),3] = apply(yalt_mc[,region_lst[[ind]]],1,fun)
        # dat_fun[((loc+1+2*mc_true):(loc+3*mc_true)),3] = apply(ynngp_mc[,region_lst[[ind]]],1,fun)
        # dat_fun[((loc+1+3*mc_true):(loc+4*mc_true)),3] = apply(yinngp_mc[,region_lst[[ind]]],1,fun)
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











# p_region_max_W2_lst = vector('list',n_region)
# for (i in 1:n_region){
#   dat_r = data.frame(matrix(0,nrow=nexp*3,ncol=2))
#   dat_m = c('altdag','nngp','inngp')
#   loc = 0
#   for (j in 1:3){
#     dat_r[(loc+1):(loc+nexp),1] = W2_fun_tensor[i,j,]
#     dat_r[(loc+1):(loc+nexp),2] = dat_m[j]
#     loc = loc + nexp
#   }
#   colnames(dat_r) <- c('W2', 'model')
#   p_region_max_W2_lst[[i]] <- ggplot(dat_r)+geom_boxplot(aes(y=W2,x=model))
# }
# grid.arrange(p_region_max_W2_lst[[1]],p_region_max_W2_lst[[2]],p_region_max_W2_lst[[3]],p_region_max_W2_lst[[4]],
#              p_region_max_W2_lst[[5]],p_region_max_W2_lst[[6]],p_region_max_W2_lst[[7]],p_region_max_W2_lst[[8]],
#              p_region_max_W2_lst[[9]],nrow=3)
  
  
  
  
  
  
  
  
##----------------------------------------------------------------------------
## Compare true mc samples with altdag mcmc samples on mean of local regions
# dat_mean <- data.frame(matrix(0,nrow=mc_true*n_region*3,ncol=3))
# colnames(dat_mean) <- c('region', 'model', 'mean')
# W2_mean <- matrix(0,nrow=n_region,ncol=2)
# colnames(W2_mean) <- c('altdag', 'nngp')
# loc = 0
# for (i in 1:ncol(plates)){
#   for (j in 1:ncol(plates)){
#     t = Sys.time()
#     ind = ceiling((i-1)*ncol(plates)+j)
#     dat_mean[((loc+1):(loc+3*mc_true)),1] = ind
#     dat_mean[((loc+1):(loc+mc_true)),2] = 'true'
#     dat_mean[((loc+1+mc_true):(loc+2*mc_true)),2] = 'altdag'
#     dat_mean[((loc+1+2*mc_true):(loc+3*mc_true)),2] = 'nngp'
#     dat_mean[((loc+1):(loc+mc_true)),3] = rowMeans(ytrue_mc[,region_lst[[ind]]])
#     dat_mean[((loc+1+mc_true):(loc+2*mc_true)),3] = rowMeans(yalt_mc[,region_lst[[ind]]])
#     dat_mean[((loc+1+2*mc_true):(loc+3*mc_true)),3] = rowMeans(ynngp_mc[,region_lst[[ind]]])
#     loc = loc + mc_true*3
#     W2_mean[ind,1] = discrete_W_1d(dat_mean[((loc+1):(loc+mc_true)),3], dat_mean[((loc+1+mc_true):(loc+2*mc_true)),3])
#     W2_mean[ind,2] = discrete_W_1d(dat_mean[((loc+1):(loc+mc_true)),3], dat_mean[((loc+1+2*mc_true):(loc+3*mc_true)),3])
#   }
# }

# p_region_mean_lst = vector('list',n_region)
# for (i in 1:n_region){
#   p_region_mean_lst[[i]] <- ggplot(dat_mean[which(dat_mean$region==i),]) +
#     geom_boxplot(aes(y=mean,x=model))
# }
# grid.arrange(p_region_mean_lst[[1]],p_region_mean_lst[[2]],p_region_mean_lst[[3]],p_region_mean_lst[[4]],
#              p_region_mean_lst[[5]],p_region_mean_lst[[6]],p_region_mean_lst[[7]],p_region_mean_lst[[8]],
#              p_region_mean_lst[[9]],nrow=3)
# 
# t_end = Sys.time()
# t_end - t0
##-------------------------------------------------------------------------
## compare discrete W2 distances
# costs_alt = cost_mat(yalt_mc, ytrue_mc)
# print(HungarianSolver(costs_alt)$cost)
# 
# costs_nngp = cost_mat(ynngp_mc, ytrue_mc)
# print(HungarianSolver(costs_nngp)$cost)



##----------------------------------------------------------------------------


# out_df <- cbind(coords_test, data.frame(
#   yout_mean = yout_chain %>% apply(1, median),
#   yout_low = yout_chain %>% apply(1, quantile, 0.025),
#   yout_high = yout_chain %>% apply(1, quantile, 0.975)
# )) 
# 
# colnames(out_df) <- c("Var1", "Var2", "y", "y_low", "y_high")
# 
# plot_df <- bind_rows(df %>% mutate(sample = "in"),
#                      out_df %>% mutate(sample = "out"))
# 
# # plot predicttions
# ggplot(plot_df %>% filter(sample=="out"), aes(Var1, Var2, fill=y)) +
#   geom_raster() +
#   scale_fill_viridis_c() +
#   theme_minimal()
# 
# # plot uncertainty
# ggplot(plot_df %>% filter(sample=="out"), aes(Var1, Var2, fill=y_high-y_low)) +
#   geom_raster() +
#   scale_fill_viridis_c() +
#   theme_minimal()
# 
# 
# 
# ################################################################# MLE
# ###############################################################################
# # visualize the likelihood on a grid of phi/sigmasq fixing true tausq
# parvals <- expand.grid(seq(1, 30, length.out=30),
#                        seq(1, 5, length.out=30),
#                        theta[3],
#                        theta[4])
# 
# grid_dens <- parvals %>% apply(1, \(x) daggp_negdens(y_train, coords_train, dag$dag, as.vector(x), 16))
# df <- parvals %>% mutate(dens = grid_dens)
# ggplot(df, aes(Var1, Var2, fill=grid_dens, z=grid_dens)) +
#   geom_raster() +
#   geom_contour(bins=40) +
#   scale_fill_viridis_c() + 
#   theme_minimal()
# 
# # maximum likelihood for covariance parameters using AltDAG
# ldens <- function(x) {
#   return(-daggp_negdens(y_train, coords_train, dag$dag, exp(x), 16)) }
# mle <- optim(c(1,.1,.1), ldens, method="L-BFGS-B", hessian=TRUE)
# (theta_mle <- mle$par %>% exp())
# 
# # maximum likelihood for covariance parameters using Vecchia Maxmin
# ldens <- function(x) {
#   return(-daggp_negdens(y_train[maxmin_model$ord], coords_train[maxmin_model$ord,], maxmin_model$dag, exp(x), 16)) }
# mle_vecchia <- optim(c(1,.1,.1), ldens, method="L-BFGS-B", hessian=TRUE)
# (theta_mle_vecchia <- mle_vecchia$par %>% exp())
# 
# 
# ################################################################# visualization
# ###############################################################################
# 
# # compare parent sets for altdag v vecchia maxmin
# # plot altdag parent sets
# par(mfrow=c(1,2))
# 
# i <- 1
# plot( coords_train[i,,drop=FALSE], pch=19, cex=1.2, xlim = c(0,1), ylim=c(0,1),
#       xlab="x1", ylab="x2", main="ALTDAG")
# points( coords_train[dag$dag[[i]]+1,,drop=FALSE], pch=19, cex=.8, col="red" )
# 
# # plot vecchia maxmin dag parent sets
# vecchia_maxmin <- maxmin_model$dag
# vecchia_ord <- maxmin_model$ord
# iv <- 1400 #which(vecchia_ord == i)
# coords_ord <- coords_train[vecchia_ord,]
# plot( coords_ord[iv,,drop=FALSE], pch=19, cex=2, xlim = c(0,1), ylim=c(0,1),
#       xlab="x1", ylab="x2", main="Vecchia MAXMIN")
# points( coords_ord[vecchia_maxmin[[iv]]+1,,drop=FALSE], pch=19, cex=.8, col="red" )
# 
# par(mfrow=c(1,1))



# # partitioning of AltDAG.
# df_layers <- coords_train %>% cbind(data.frame(layers = dag$layers)) 
# colnames(df_layers) <- c("Var1", "Var2", "layers")
# 
# # partitioning into layers, all colors at once
# ggplot(df_layers, aes(Var1, Var2, color=factor(layers))) +
#   geom_point() +
#   theme_minimal()
# 
# # plot locations belonging to one of the layers only
# ggplot(df_layers %>% filter(layers == 2), aes(Var1, Var2, color=factor(layers))) +
#   geom_point() +
#   theme_minimal()




