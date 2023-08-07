library(radgp)
library(ggplot2)
library(transport)
library(rSPDE)
library(MASS)

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
set.seed(77)
mc_true = 1000
mcmc = 2500
n_repeat = 50

## covariance settings
## Gaussian covariance function
# phi = - log(0.05)/0.15^2
# expfun <- function(x,phi=133.14){
#   return(exp(-phi*x^2))
# }

## Matern covariance function
theta <- c(20, 1, 3/2, 0)
matern_hint_fun <- function(phi, dist=0.15, sigmasq=1, thre=0.05){
  return( thre - sigmasq*(1+phi*dist) * exp(-phi*dist) )
}
theta[1] = dichotomy_solver(matern_hint_fun, 10, 50)

## if desire high smoothness
# theta = c(20, 1, 3/2, 0)
# theta = c(25, 1, 5/2, 0)

## nugget for latent model
nugget <- 1e-2

## training data
nl = 40
ntrain = nl^2
coords_train = as.matrix(expand.grid(xout<-seq(0,1,length.out=nl),xout))
CC_train <- radgp::Correlationc(coords_train, coords_train, theta, 0, T)
CC_train_inv <- ginv(CC_train)
eigenobj_train = eigen(CC_train,symmetric=TRUE)
D_train = pmax(eigenobj_train$values,0)
U_train = eigenobj_train$vectors
LC_train <- U_train %*% diag(sqrt(D_train)) %*% t(U_train)

## calculate information only relavent to training data
rho_lst = c(0.045, 0.055, 0.065, 0.075, 0.085, 0.095, 0.105, 0.114, 0.125)
m_lst = c(3, 4, 5, 7, 9, 12, 15, 18, 22)
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
n_region = 4
results_tensor = array(0, dim=c(length(rho_lst)+2*length(m_lst), n_region+1, n_repeat))

for (i_repeat in 1:n_repeat){
  
  ## uniform test data in small regions and remaining regions, respectively
  IsInRegion <- function(vec){
    if (vec[1]>0.45 & vec[1]<0.55 & vec[2]>0.45 & vec[2]<0.55){
      return(TRUE)
    } else if (vec[1]>0.1 & vec[1]<0.2 & vec[2]>0.1 & vec[2]<0.2){
      return(TRUE)
    } else if (vec[1]>0.7 & vec[1]<0.8 & vec[2]>0.2 & vec[2]<0.3){
      return(TRUE)
    } else if (vec[1]>0.3 & vec[1]<0.4 & vec[2]>0.9 & vec[2]<1){
      return(TRUE)
    } else
    return(FALSE)
  }
  ntest = 1000
  nall = ntrain + ntest
  coords_test = matrix(0, nrow=ntest, ncol=2)
  coords_test[1:40,] = matrix(runif(40*2),nrow=40,ncol=2)*0.1
  coords_test[1:10,1] = coords_test[1:10,1]+0.45
  coords_test[1:10,2] = coords_test[1:10,2]+0.45
  coords_test[11:20,1] = coords_test[1:10,1]+0.1
  coords_test[11:20,2] = coords_test[1:10,2]+0.1  
  coords_test[21:30,1] = coords_test[1:10,1]+0.7
  coords_test[21:30,2] = coords_test[1:10,2]+0.2
  coords_test[31:40,1] = coords_test[1:10,1]+0.3
  coords_test[31:40,2] = coords_test[1:10,2]+0.9  
  flag = FALSE
  while (flag == FALSE){
    coords_raw = matrix(runif(2*ntest*2),nrow=2*ntest,ncol=2)
    nout = 0
    out_ids = vector('numeric',ntest-40)
    for (i in 1:nrow(coords_raw)){
      if (IsInRegion(coords_raw[i,]) == FALSE){
        nout = nout + 1
        out_ids[nout] = i
        if (nout >= ntest - 40){
          coords_test[41:ntest,] = coords_raw[out_ids,]
          flag = TRUE
          break
        }
      }
    }
  }
  coords_all = rbind(coords_train, coords_test)
  
  ## generate joint test samples from the true conditional distribution
  CC <- radgp::Correlationc(coords_all, coords_all, theta, 0, T)
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
  
  # # some extra experimental data
  # ztrue_mc_2 = matrix(nrow=mc_true,ncol=ntest)
  # for (i in 1:mc_true){
  #   ztrue_mc_2[i,] = LC_cond %*% rnorm(ntest) + CC_testtrain %*% CC_train_inv %*% z_train
  # }
  
  
  ##----------------------------------mcmc----------------------------------------
  theta_start <- c(40, 1, theta[3])
  theta_unif_bounds <- matrix(nrow=3, ncol=2)
  theta_unif_bounds[1,] <- c(1, 100) # phi
  theta_unif_bounds[2,] <- c(.1, 10) # sigmasq
  theta_unif_bounds[3,] <- c(theta[3]-0.001, theta[3]+0.001) # nu
  # theta_unif_bounds[3,] <- c(0.1, 10)
  nugget_start <- c(.01)
  nugget_prior <- c(1, 0.01)
  
  # results_df = data.frame(matrix(0,nrow=n_region*(length(rho_lst)+2*length(m_lst)),ncol=4))
  # colnames(results_df) <- c('Ave.Neighbors', 'W22', 'Method', 'Region')
  # row_loc = 1
  
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
    for (r in 1:n_region){
      results_tensor[j,r,i_repeat] =  wasserstein(pp(ztrue_mc[,(1+10*r-10):10*r]), pp(radgp_pred[,(1+10*r-10):10*r]), p=2)
    }
    results_tensor[j,n_region+1,i_repeat] = wasserstein(pp(ztrue_mc), pp(radgp_pred), p=2)
    
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
    for (r in 1:n_region){
      results_tensor[length(rho_lst)+j,r,i_repeat] =  wasserstein(pp(ztrue_mc[,(1+10*r-10):10*r]), pp(nngp_pred[,(1+10*r-10):10*r]), p=2)
    }
    results_tensor[length(rho_lst)+j,n_region+1,i_repeat] =  wasserstein(pp(ztrue_mc), pp(nngp_pred), p=2)
    for (r in 1:n_region){
      results_tensor[length(rho_lst)+length(m_lst)+j,r,i_repeat] =  wasserstein(pp(ztrue_mc[,(1+10*r-10):10*r]), pp(nngp_pred_i[,(1+10*r-10):10*r]), p=2)
    }
    results_tensor[length(rho_lst)+length(m_lst)+j,n_region+1,i_repeat] =  wasserstein(pp(ztrue_mc), pp(nngp_pred_i), p=2)
    
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
df = data.frame(matrix(0,nrow=(length(m_lst)*2+length(rho_lst))*(n_region+1), ncol=4))
colnames(df) <- c('Nonzeros', 'W22', 'W22low', 'W22up')
df$Region = 1
df$Method = 'RadGP'
# df = data.frame(nonzeros=double(), method=character(), W22=double(), W22up=double(), W22low=double(), Region=integer())
l = 1
alpha = 0.1
left = floor(n_repeat*alpha/2)
right = floor(n_repeat-n_repeat*alpha/2)
for (j in 1:length(rho_lst)){
  for (r in 1:(n_region+1)){
    samples = results_tensor[j,r,]
    samples = sort(samples)
    df$Nonzeros[l] = ave_nonzeros[j]
    df$Method[l] = 'RadGP'
    df$W22low[l] = mean(samples[left:(left+1)])
    df$W22up[l] = mean(samples[right:(right+1)])
    df$W22[l] = mean(samples)
    df$Region[l] = r
    l = l + 1
  }
}
for (j in 1:length(m_lst)){
  for (r in 1:(n_region+1)){
    samples = results_tensor[length(rho_lst)+j,r,]
    samples = sort(samples)
    df$Nonzeros[l] = ave_nonzeros[length(rho_lst)+j]
    df$Method[l] = 'V-Pred'
    df$W22low[l] = mean(samples[left:(left+1)])
    df$W22up[l] = mean(samples[right:(right+1)])
    df$W22[l] = mean(samples)
    df$Region[l] = r
    l = l + 1
  }
}
for (j in 1:length(m_lst)){
  for (r in 1:(n_region+1)){
    samples = results_tensor[length(rho_lst)+length(m_lst)+j,r,]
    samples = sort(samples)
    df$Nonzeros[l] = ave_nonzeros[length(rho_lst)+j]
    df$Method[l] = 'NNGP'
    df$W22low[l] = mean(samples[left:(left+1)])
    df$W22up[l] = mean(samples[right:(right+1)])
    df$W22[l] = mean(samples)
    df$Region[l] = r
    l = l + 1
  }
}


ggplot(df[which(df$Region<=4),]) +
  geom_line(aes(x=Nonzeros,y=W22,color=Method)) + facet_wrap(~Region, nrow=2)

ggplot(df[which(df$Region==5),]) +
  geom_line(aes(x=Nonzeros,y=W22,color=Method)) +
  geom_ribbon(aes(x=Nonzeros,ymin=W22low,ymax=W22up,fill=Method),alpha=0.2) +
  xlab('Ave.Nonzeros') + theme_minimal(base_size = 25)




# ggplot(results_df) +
#   geom_line(aes(x=Ave.Neighbors,y=W22,color=Method)) + facet_wrap(~Region, nrow=2)
# 
# for (r in 1:n_region){
#   print(wasserstein(pp(ztrue_mc[,region_id[[r]]]), pp(ztrue_mc_2[,region_id[[r]]]), p=2))
# }









# ## examine small region behavior
# region_id = which(coords_test[,1]>0.45 & coords_test[,1]<0.55 & 
#                     coords_test[,2]>0.45 & coords_test[,2]<0.55)
# wasserstein(pp(ztrue_mc[,region_id]), pp(radgp_predict[,region_id]), p=2)
# wasserstein(pp(ztrue_mc[,region_id]), pp(nngp_predict[,region_id]), p=2)
# wasserstein(pp(ztrue_mc[,region_id]), pp(nngp_predict_i[,region_id]), p=2)





























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


