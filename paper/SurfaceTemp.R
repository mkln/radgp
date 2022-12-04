library(ggplot2)
library(imager)
library(radgp)
library(reshape2)

set.seed(10)

coverage <- function(zhat_mcmc, ztrue, alpha=0.95){
  ## zhat_mcmc is a ntest \times n_mcmc matrix, ztrue is a ntest length vector
  s = 0
  n_mcmc = dim(zhat_mcmc)[2]
  ntest = dim(zhat_mcmc)[1]
  indices = c(round(n_mcmc*(1-alpha)/2), n_mcmc-round(n_mcmc*(1-alpha)/2))
  for (i in 1:ntest){
    chain_sort = sort(zhat_mcmc[i,])
    if (ztrue[i]>=chain_sort[indices[1]] & ztrue[i]<=chain_sort[indices[2]]){
      s = s + 1
    }
  }
  return(s/ntest)
}

CI <- function(lst,alpha=0.95){
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

discrete_W_1d <- function(path1, path2, order=2){
  sort1 = sort(path1)
  sort2 = sort(path2)
  n = length(path1)
  W = (sum((abs(sort1-sort2))^order)/n)^(1/order)
  return(W)
}

SW2_pair <- function(vec, mat){
  L = 10
  arclst = seq(0, 2*pi-1e-3, 2*pi/L)
  coslst = cos(arclst)
  sinlst = sin(arclst)
  n = nrow(mat)
  vecp = vec[1]*coslst + vec[2]*sinlst
  matp = mat %*% rbind(coslst, sinlst)
  S = 0
  for (l in 1:L){
    S = S + sqrt(sum((matp[,l]-vecp[l])^2)/n)
  }
  return(S/L)
}




df0 = as.data.frame(load.image('/home/yichen/data/MYD11A1_LST_Day_1km_2021_091.tif'))
df = df0[which(df0$value>10),]
nall = nrow(df)
id_sample = sample(1:nall,replace=FALSE)
ntrain = round(nall*2/3)
ntest = round(nall*1/3)
coords_train = as.matrix(df[id_sample[1:ntrain],1:2])
coords_test = as.matrix(df[id_sample[(ntrain+1):nall],1:2])
y_train = df[id_sample[1:ntrain],3]
y_test = df[id_sample[(ntrain+1):nall],3]
y_train_m = mean(y_train)
y_train_c = y_train - y_train_m
y_test_c = y_test - y_train_m


##---------------------------model settings------------------------------------
rho = 4.01
m = 15
mcmc = 5000
mcmc_keep = 2500

theta_start <- c(0.1, 1, 1)
nugget_start = 0.1
theta_unif_bounds <- matrix(nrow=3, ncol=2)
theta_unif_bounds[1,] <- c(0.001, 1000) # phi
theta_unif_bounds[2,] <- c(0.001, 1000) # sigmasq
theta_unif_bounds[3,] <- c(0.999999,1.000001)
nugget_prior <- c(1, 0.1)

##---------------------------------radGP---------------------------------------
t0 = as.numeric(Sys.time())
raddag_model <- latent.model(y_train_c, coords_train, rho=rho, 
                             theta_start=theta_start,
                             theta_prior=theta_unif_bounds,
                             nugg_start=nugget_start,
                             nugg_prior=nugget_prior,
                             mcmc=mcmc, n_threads=16, printn=100)
t1 = as.numeric(Sys.time())  
raddag_predict <- predict(raddag_model, coords_test, rho=rho, mcmc_keep=mcmc_keep, n_threads=16)
t2 = as.numeric(Sys.time())  
rad_theta_m = apply(raddag_model$theta[,(mcmc-mcmc_keep+1):mcmc], MARGIN=1, mean)
rad_nugg_m = mean(raddag_model$nugg[(mcmc-mcmc_keep+1):mcmc])
rad_MSE = mean((apply(raddag_predict$yout, MARGIN=1, mean) - y_test_c)^2)
rad_coverage = coverage(raddag_predict$yout, y_test_c, alpha=0.95)



##---------------------------------vecchia GP---------------------------------------
t3 = as.numeric(Sys.time())
vecchia_model <- latent.model.vecchia(y_train_c, coords_train, m=m,
                                     theta_start=theta_start,
                                     theta_prior=theta_unif_bounds,
                                     nugg_start=nugget_start,
                                     nugg_prior=nugget_prior,
                                     mcmc=mcmc, n_threads=16, printn=100)
t4 = as.numeric(Sys.time())  
vecchia_predict <- predict(vecchia_model, coords_test, mcmc_keep=mcmc_keep, n_threads=16)
t5 = as.numeric(Sys.time())  
vecchia_theta_m = apply(vecchia_model$theta[,(mcmc-mcmc_keep+1):mcmc], MARGIN=1, mean)
vecchia_nugg_m = mean(vecchia_model$nugg[(mcmc-mcmc_keep+1):mcmc])
vecchia_MSE = mean((apply(vecchia_predict$yout, MARGIN=1, mean) - y_test_c)^2)
vecchia_coverage = coverage(vecchia_predict$yout, y_test_c, alpha=0.95)




##----------------------------print summary------------------------------------
print('rad GP summaries:')
print(paste('training time: ', t1-t0, ', testing time: ', t2-t1, ', total time: ', t2-t0, sep=''))
print(paste('theta:', paste(rad_theta_m,collapse=" "), 'nugget:', rad_nugg_m, sep=' '))
print(paste('MSE:', rad_MSE, 'coverage:', rad_coverage, sep=' '))
rad_CI = matrix(0,nrow=4,ncol=3)
for (i in 1:3){
  rad_CI[i,] = CI(raddag_model$theta[i,(mcmc-mcmc_keep+1):mcmc])
}
rad_CI[4,] = CI(raddag_model$nugg[(mcmc-mcmc_keep+1):mcmc])


print('vecchia GP summaries:')
print(paste('training time: ', t4-t3, ', testing time: ', t5-t4, ', total time: ', t5-t3, sep=''))
print(paste('theta:', paste(vecchia_theta_m,collapse=" "), 'nugget:', vecchia_nugg_m, sep=' '))
print(paste('MSE:', vecchia_MSE, 'coverage:', vecchia_coverage, sep=' '))
vecchia_CI = matrix(0,nrow=4,ncol=3)
for (i in 1:3){
  vecchia_CI[i,] = CI(vecchia_model$theta[i,(mcmc-mcmc_keep+1):mcmc])
}
vecchia_CI[4,] = CI(vecchia_model$nugg[(mcmc-mcmc_keep+1):mcmc])


##--------------------------training on all observed---------------------------
# df0 = as.data.frame(load.image('/home/yichen/data/MYD11A1_LST_Day_1km_2021_091.tif'))
# coords_all = as.matrix(df0[which(df0$value>10),1:2])
# y_all = df0[which(df0$value>10),3]
# y_all_m = mean(y_all)
# y_all_c = y_all - y_all_m
# coords_cloud = as.matrix(df0[which(df0$value<10),1:2])
# 
# rho = 4.01
# m = 15
# mcmc = 5000
# mcmc_keep = 2500
# 
# theta_start <- c(0.1, 1, 1)
# nugget_start = 0.1
# theta_unif_bounds <- matrix(nrow=3, ncol=2)
# theta_unif_bounds[1,] <- c(0.001, 1000) # phi
# theta_unif_bounds[2,] <- c(0.001, 1000) # sigmasq
# theta_unif_bounds[3,] <- c(0.999999,1.000001)
# nugget_prior <- c(1, 0.1)
# 
# t6 = as.numeric(Sys.time())
# raddag_model_all <- latent.model(y_all_c, coords=coords_all, rho=rho, 
#                              theta_start=theta_start,
#                              theta_prior=theta_unif_bounds,
#                              nugg_start=nugget_start,
#                              nugg_prior=nugget_prior,
#                              mcmc=mcmc, n_threads=16, printn=100)
# t7 = as.numeric(Sys.time())  
# raddag_predict_all <- predict(raddag_model_all, coords_cloud, rho=rho, mcmc_keep=mcmc_keep, n_threads=16)
# t8 = as.numeric(Sys.time())  
# y_cloud_pred = raddag_predict_all$yout + y_all_m
# rad_theta_m_all = apply(raddag_model_all$theta[,(mcmc-mcmc_keep+1):mcmc], MARGIN=1, mean)
# rad_nugg_m_all = mean(raddag_model_all$nugg[(mcmc-mcmc_keep+1):mcmc])
# df_cloud = df0
# df_cloud[which(df0$value<10),3] = y_cloud_pred
# 
# ggplot(df_cloud) + 
#   geom_raster(aes(x=x, y=y, fill=value)) + 
#   scale_fill_viridis_c() + xlab("") + ylab("") +
#   theme_minimal(base_size = 20)


  
##------------------------predict cloud using 2/3 training----------------------
coords_test_all = rbind(coords_test, as.matrix(df0[which(df0$value<10),1:2]))
raddag_predict_all <- predict(raddag_model, coords_test_all, rho=rho, mcmc_keep=mcmc_keep, n_threads=16)
w_test_all = apply(raddag_predict_all$wout, MARGIN=1, mean) + y_train_m
df_all = df0
df_all$Temp = 0
df_all$Temp[which(df0$value>10)[id_sample[1:ntrain]]] = y_train_m + apply(raddag_model$w[,(mcmc-mcmc_keep+1):mcmc], MARGIN=1, mean)
df_all$Temp[c(which(df0$value>10)[id_sample[(ntrain+1):nall]],which(df0$value<10))] = y_train_m + apply(raddag_predict_all$wout, MARGIN=1, mean)
ggplot(df_all) +
  geom_raster(aes(x=x, y=y, fill=Temp)) +
  scale_fill_viridis_c() + xlab(NULL) + ylab(NULL) +
  theme_void(base_size = 20) +
  theme(axis.text.x=element_blank(),axis.text.y=element_blank())
  

##------------------------- plot observed data ---------------------------------

colnames(df0) <- c('x', 'y', 'Temp')
ggplot(df0[which(df0$Temp>10),]) +
  geom_raster(aes(x=x, y=y, fill=Temp)) +
  scale_fill_viridis_c(guide="none") + xlab(NULL) + ylab(NULL) +
  theme_void(base_size = 20) + 
  theme(axis.text.x=element_blank(),axis.text.y=element_blank())



##-------------------------Log likelihood ratio for local pairs------------------------
coords_exp = cbind(cbind(coords_test,1:nrow(coords_test)),0)
coords_exp = coords_exp[order(coords_exp[,1]),]
coords_exp = coords_exp[order(coords_exp[,2]),]
r_gap = 10
region_counts = 0
i = 2
while (i < nrow(coords_exp)){
  if (! coords_exp[i,2]%%r_gap == 0){
    i = i + 1
    next
  }
  if (coords_exp[i+1,2] == coords_exp[i,2] & coords_exp[i+1,1] == coords_exp[i,1]+1
      & coords_exp[i-1,1] != coords_exp[i,1]-1){
    region_counts = region_counts + 1
    coords_exp[i:(i+1),4] = region_counts
    i =i + r_gap
  } else{
    i = i + 1
  }
}

rad_theta_m = rowMeans(raddag_model$theta[,(mcmc-mcmc_keep+1):mcmc])
rad_nugg_m = mean(raddag_model$nugg[(mcmc-mcmc_keep+1):mcmc])
rad_w_m = rowMeans(raddag_model$w[,(mcmc-mcmc_keep+1):mcmc])

vecchia_theta_m = rowMeans(vecchia_model$theta[,(mcmc-mcmc_keep+1):mcmc])
vecchia_nugg_m = mean(vecchia_model$nugg[(mcmc-mcmc_keep+1):mcmc])
vecchia_w_m = vector(mode='numeric',length=length(rad_w_m))
vecchia_w_m[vecchia_model$ord] = rowMeans(vecchia_model$w[,(mcmc-mcmc_keep+1):mcmc])

loglike_tab = matrix(0,nrow=region_counts,ncol=2)
mse_tab = matrix(0,nrow=region_counts,ncol=2)
for (i in 1:region_counts){
  wt = coords_exp[which(coords_exp[,4]==i),1:2]
  d1_lst = (coords_train[,1]-wt[1,1])^2 + (coords_train[,2]-wt[1,2])^2
  d2_lst = (coords_train[,1]-wt[2,1])^2 + (coords_train[,2]-wt[2,2])^2
  yob = y_test_c[coords_exp[which(coords_exp[,4]==i),3]]
  
  ## Compute log likelihood for radGP
  N1_ind = which(d1_lst<rho^2)
  n1 = length(N1_ind)
  N2_ind = which(d2_lst<rho^2)  
  n2 = length(N2_ind)
  coords1_rad = rbind(coords_train[N1_ind,],matrix(wt[1,1:2],nrow=1))
  Sig1_rad = radgp::Correlationc(coords1_rad, coords1_rad, c(rad_theta_m,10^(-6)),TRUE) 
  Sig1painv_rad = solve(Sig1_rad[1:n1,1:n1])
  mu1_rad = matrix(Sig1_rad[n1+1,1:n1],nrow=1) %*% Sig1painv_rad %*% matrix(rad_w_m[N1_ind],ncol=1)
  v1_rad = Sig1_rad[n1+1,n1+1] - matrix(Sig1_rad[n1+1,1:n1],nrow=1) %*% Sig1painv_rad %*% matrix(Sig1_rad[n1+1,1:n1],ncol=1)
  
  coords2_rad = rbind(coords_train[N2_ind,],wt)
  Sig2_rad = radgp::Correlationc(coords2_rad, coords2_rad, c(rad_theta_m,10^(-6)),TRUE) 
  Sig2painv_rad = solve(Sig2_rad[1:(n2+1),1:(n2+1)])
  mu2_rad = matrix(Sig2_rad[n2+2,1:(n2+1)],nrow=1) %*% Sig2painv_rad %*% matrix(c(rad_w_m[N2_ind],mu1_rad),ncol=1)
  v2_rad = Sig2_rad[n2+2,n2+2] - matrix(Sig2_rad[n2+2,1:(n2+1)],nrow=1) %*% Sig2painv_rad %*% matrix(Sig2_rad[n2+2,1:(n2+1)],ncol=1)
  corre_rad = matrix(Sig2_rad[n2+2,1:(n2+1)],nrow=1) %*% matrix(Sig2painv_rad[,n2+1],ncol=1)
  
  mu_rad = c(mu1_rad, mu2_rad)
  cov_rad = solve(matrix(c(1/v1_rad, -corre_rad/v2_rad, -corre_rad/v2_rad, 1/v2_rad), nrow=2)) + diag(c(1,1))*rad_nugg_m
  loglike_tab[i,1] = -log(2*pi) - 0.5*log(det(cov_rad)) - 0.5* matrix(yob-mu_rad,nrow=1) %*% solve(cov_rad) %*% matrix(yob-mu_rad,ncol=1)
  mse_tab[i,1] = sum((yob-mu_rad)^2)/2
  
  ## Compute log likelihood for vecchia GP
  N1v_ind = order(d1_lst)[1:m]
  N2v_ind = order(d2_lst)[1:m]
  
  coords1_vecchia = rbind(coords_train[N1v_ind,],matrix(wt[1,1:2],nrow=1))
  Sig1_vecchia = radgp::Correlationc(coords1_vecchia, coords1_vecchia, c(vecchia_theta_m,10^(-6)),TRUE) 
  Sig1painv_vecchia = solve(Sig1_vecchia[1:m,1:m])
  mu1_vecchia = matrix(Sig1_vecchia[m+1,1:m],nrow=1) %*% Sig1painv_vecchia %*% matrix(vecchia_w_m[N1v_ind],ncol=1)
  v1_vecchia = Sig1_vecchia[m+1,m+1] - matrix(Sig1_vecchia[m+1,1:m],nrow=1) %*% Sig1painv_vecchia %*% matrix(Sig1_vecchia[m+1,1:m],ncol=1)
  
  coords2_vecchia = rbind(coords_train[N2v_ind,],matrix(wt[2,1:2],nrow=1))
  Sig2_vecchia = radgp::Correlationc(coords2_vecchia, coords2_vecchia, c(vecchia_theta_m,10^(-6)),TRUE) 
  Sig2painv_vecchia = solve(Sig2_vecchia[1:m,1:m])
  mu2_vecchia = matrix(Sig2_vecchia[m+1,1:m],nrow=1) %*% Sig2painv_vecchia %*% matrix(vecchia_w_m[N2v_ind],ncol=1)
  v2_vecchia = Sig2_vecchia[m+1,m+1] - matrix(Sig2_vecchia[m+1,1:m],nrow=1) %*% Sig2painv_vecchia %*% matrix(Sig2_vecchia[m+1,1:m],ncol=1)

  loglike_tab[i,2] = -log(2*pi) - 0.5*log((v1_vecchia+vecchia_nugg_m)*(v2_vecchia+vecchia_nugg_m)) - 
    0.5* ((yob[1]-mu1_vecchia)^2/(v1_vecchia+vecchia_nugg_m) + (yob[2]-mu2_vecchia)^2/(v2_vecchia+vecchia_nugg_m))
  mse_tab[i,2] = ((yob[1]-mu1_vecchia)^2 + (yob[2]-mu2_vecchia)^2)/2
}


logratio = loglike_tab[,1]-loglike_tab[,2]
logratio = sort(logratio)
alpha_remove = 0.1
lshre = round(length(logratio)*(alpha_remove/2))
rshre = round(length(logratio)*((1-alpha_remove/2)))
df_loglike = data.frame(logratio[lshre:rshre])
colnames(df_loglike) = c('logratio')
ggplot(df_loglike) +
  geom_histogram(aes(logratio),binwidth=0.01) + geom_vline(xintercept = 0, color='red',size=1.5) +
  geom_vline(xintercept = mean(logratio), color='blue',size=1.5) +
  xlab("Log likelihood ratio") + ylab('Counts') + theme_minimal(base_size = 25)
# stat_ecdf(aes(logratio)) + xlim(-1,1)
# geom_boxplot(aes(y=logratio),binwidth=0.1)









# ## unused experiments
# ##---------------------------sliced W2 on pairs---------------------------------
# coords_exp = cbind(cbind(coords_test,1:nrow(coords_test)),0)
# coords_exp = coords_exp[order(coords_exp[,1]),]
# coords_exp = coords_exp[order(coords_exp[,2]),]
# r_gap = 10
# region_counts = 0
# i = 2
# while (i < nrow(coords_exp)){
#   if (! coords_exp[i,2]%%r_gap == 0){
#     i = i + 1
#     next
#   }
#   if (coords_exp[i+1,2] == coords_exp[i,2] & coords_exp[i+1,1] == coords_exp[i,1]+1
#       & coords_exp[i-1,1] != coords_exp[i,1]-1){
#     region_counts = region_counts + 1
#     coords_exp[i:(i+1),4] = region_counts
#     i =i + r_gap
#   } else{
#     i = i + 1
#   }
# }
# SW2_lst = matrix(0, nrow=region_counts, ncol=2)
# for (i in 1:region_counts){
#   yob = y_test_c[coords_exp[which(coords_exp[,4]==i),3]]
#   ypred_rad = t(raddag_predict$yout[coords_exp[which(coords_exp[,4]==i),3],])
#   ypred_vecchia = t(vecchia_predict$yout[coords_exp[which(coords_exp[,4]==i),3],])
#   SW2_lst[i,] = c(SW2_pair(yob, ypred_rad), SW2_pair(yob, ypred_vecchia))
# }

##-------------------------- display local means --------------------------------
# df_pred = data.frame(matrix(0,nrow=mcmc_keep*3,ncol=2))
# colnames(df_pred) <- c('Value','Region')
# gap = mcmc_keep
# n_loc = 441
# loc = 1
# region_names = c('A','B','C')
# for (k in 1:3){
#   idtrain = c(which(coords_train[,1]>df_rects[k,1] & coords_train[,1]<df_rects[k,2] 
#                     & coords_train[,2]>df_rects[k,3] & coords_train[,2]<df_rects[k,4]))
#   idtest = c(which(coords_test_all[,1]>df_rects[k,1] & coords_test_all[,1]<df_rects[k,2] 
#                    & coords_test_all[,2]>df_rects[k,3] & coords_test_all[,2]<df_rects[k,4]))
#   df_pred$Value[loc:(loc+gap-1)] = y_train_m + (apply(raddag_model$w[idtrain,(mcmc-mcmc_keep+1):mcmc],MARGIN=2,FUN=max)
#                            + apply(raddag_predict_all$wout[idtest,],MARGIN=2,FUN=max)) / n_loc
#   df_pred$Region[loc:(loc+gap-1)] = region_names[k]
#   loc = loc + gap
# }
# 
# ggplot(df_pred) +
#   geom_boxplot(aes(y=Value,x=Region, fill=Region)) +
#   labs(y='Local Maximum',x=NULL,title=NULL) +  theme_minimal(base_size = 30) +
#   theme(legend.position="none") + xlab("") +
#   scale_colour_manual(aesthetics = "fill",values=c("#D81B60", "#1E88E5", '#FFC107'))  
# 
# 
# 
# #D81B60", "#1E88E5", '#FFC107'

# ##---------------------------Extract local regions------------------------------
# coords_exp = cbind(cbind(coords_test,1:nrow(coords_test)),0)
# coords_exp = coords_exp[order(coords_exp[,1]),]
# coords_exp = coords_exp[order(coords_exp[,2]),]
# r_gap = 10
# region_counts = 0
# i = 2
# while (i < nrow(coords_exp)){
#   if (! coords_exp[i,2]%%r_gap == 0){
#     i = i + 1
#     next
#   }
#   if (coords_exp[i+1,2] == coords_exp[i,2] & coords_exp[i+1,1] == coords_exp[i,1]+1
#       & coords_exp[i-1,1] != coords_exp[i,1]-1){
#     region_counts = region_counts + 1
#     coords_exp[i:(i+1),4] = region_counts
#     i =i + r_gap
#   } else{
#     i = i + 1
#   }
# }
# 
# maxdiff <- function(vec){
#   return(max(vec)-min(vec))
# }
# JPfun = c(maxdiff)
# JP_records = array(0,dim=c(region_counts,length(JPfun),3))
# for (i in 1:region_counts){
#   for (j in 1:length(JPfun)){
#     fun = JPfun[[j]]
#     f_true = fun(y_test_c[coords_exp[which(coords_exp[,4]==i),3]])
#     f_rad = apply(raddag_predict$yout[coords_exp[which(coords_exp[,4]==i),3],], MARGIN=2, FUN=fun)
#     f_vecchia = apply(vecchia_predict$yout[coords_exp[which(coords_exp[,4]==i),3],], MARGIN=2, FUN=fun)
#     JP_records[i,j,1] = mean(f_rad)
#     JP_records[i,j,2] = mean(f_vecchia)
#     JP_records[i,j,3] = f_true
#   }
# }
# 
# JP_records_df = melt(JP_records[,1,], varnames = c('region', 'method'))
# # JP_records_df$fun = as.factor(JP_records_df$fun)
# JP_records_df$method = as.factor(JP_records_df$method)
# ggplot(JP_records_df) +
#   geom_boxplot(aes(x=method,y=value)) 

# r_start = 19.5
# r_gap = 50
# r_width = 10
# min_counts = 3
# 
# coords_test_exp = cbind(coords_test,0)
# x = r_start
# y = r_start
# region_counts = 1
# while (x < max(coords_train[,1])){
#   while (y < max(coords_train[,2])){
#     id_exp = which(coords_test[,1]>x & coords_test[,1]<x+r_width & coords_test[,2]>y & coords_test[,2]<y+r_width) 
#     if (length(id_exp) > min_counts){
#       coords_test_exp[id_exp,3] = region_counts
#       region_counts = region_counts + 1
#     }
#     y = y + r_gap
#   }
#   y = r_start
#   x = x + r_gap
# }
# 
# region_counts = region_counts - 1
# JPfun = c(sd, mean, max, median)
# JP_records = array(0,dim=c(region_counts,length(JPfun),2))
# for (i in 1:region_counts){
#   for (j in 1:length(JPfun)){
#     fun = JPfun[[j]]
#     f_true = fun(y_test_c[which(coords_test_exp[,3]==i)])
#     f_rad = mean(apply(raddag_predict$yout[which(coords_test_exp[,3]==i),], MARGIN=1, FUN=fun))
#     f_vecchia = mean(apply(vecchia_predict$yout[which(coords_test_exp[,3]==i),], MARGIN=1, FUN=fun))
#     JP_records[i,j,1] = mean((f_rad - f_true)^2)
#     JP_records[i,j,2] = mean((f_vecchia - f_true)^2)
#   }
# }
# 
# JP_records_df = melt(JP_records, varnames = c('region', 'fun', 'method'))
# JP_records_df$fun = as.factor(JP_records_df$fun)
# JP_records_df$method = as.factor(JP_records_df$method)
# ggplot(JP_records_df) +
#   geom_boxplot(aes(x=method,y=value)) + facet_wrap(~fun, nrow=1)







