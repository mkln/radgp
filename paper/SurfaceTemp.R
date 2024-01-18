library(ggplot2)
library(imager)
library(radgp)
library(reshape2)

set.seed(100)

coverage <- function(zhat_mcmc, ztrue, alpha=0.95){
  ## zhat_mcmc is a ntest \times n_mcmc matrix, ztrue is a ntest length vector
  s = 0
  n_mcmc = dim(zhat_mcmc)[2]
  ntest = dim(zhat_mcmc)[1]
  indices = c(round(n_mcmc*(1-alpha)/2), n_mcmc-round(n_mcmc*(1-alpha)/2))
  for (i in 1:ntest){
    chain_sort = sort(zhat_mcmc[i,])
    if (ztrue[i]<=chain_sort[indices[2]]){
      if (indices[1]==0){
        s = s + 1
      } else if (ztrue[i]>=chain_sort[indices[1]]){
        s = s + 1
      }
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




df0 = as.data.frame(load.image('/home/yichen/prev_server_data/data/MYD11A1_LST_Day_1km_2021_091.tif'))
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

theta_start <- c(0.1, 1, 1, 0)  ## the original paper theta_start setting, which is exponential covariance function
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
                             mcmc=mcmc, n_threads=16, covariance_model="pexp", printn=100)    
t1 = as.numeric(Sys.time())  

raddag_predict <- predict(raddag_model, coords_test, mcmc_keep=mcmc_keep, n_threads=16)
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
                                     mcmc=mcmc, n_threads=16, covariance_model="pexp", printn=100)
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

  
##------------------------predict cloud using 2/3 training----------------------
coords_test_all = rbind(coords_test, as.matrix(df0[which(df0$value<10),1:2]))
raddag_predict_all <- predict(raddag_model, coords_test_all, rho=rho, mcmc_keep=mcmc_keep, n_threads=16)
w_test_all = apply(raddag_predict_all$wout, MARGIN=1, mean) + y_train_m
df_all = df0
df_all$Temp = 0
df_all$Temp[which(df0$value>10)[id_sample[1:ntrain]]] = y_train_m + apply(raddag_model$w[,(mcmc-mcmc_keep+1):mcmc], MARGIN=1, mean)
df_all$Temp[c(which(df0$value>10)[id_sample[(ntrain+1):nall]],which(df0$value<10))] = y_train_m + apply(raddag_predict_all$wout, MARGIN=1, mean)

##------------------------- plot observed data ---------------------------------

df_joint = rbind(df_all[,-3],df0[which(df0$Temp>10),])
df_joint$type = 0
df_joint$type[1:nrow(df_all)] = 1
df_joint$type = as.factor(df_joint$type)
colnames(df0) <- c('x', 'y', 'Temp')
ggplot(df_joint) +
  geom_raster(aes(x=x, y=y, fill=Temp)) +
  scale_fill_viridis_c() + xlab(NULL) + ylab(NULL) +
  theme_void(base_size = 25) + 
  facet_wrap(~type) +
  theme(axis.text.x=element_blank(), axis.text.y=element_blank(), 
        strip.background = element_blank(),  strip.text.x = element_blank()) 


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
  Sig1_rad = radgp::Correlationc(coords1_rad, coords1_rad, c(rad_theta_m,10^(-6)), 1, T) 
  Sig1painv_rad = solve(Sig1_rad[1:n1,1:n1])
  mu1_rad = matrix(Sig1_rad[n1+1,1:n1],nrow=1) %*% Sig1painv_rad %*% matrix(rad_w_m[N1_ind],ncol=1)
  v1_rad = Sig1_rad[n1+1,n1+1] - matrix(Sig1_rad[n1+1,1:n1],nrow=1) %*% Sig1painv_rad %*% matrix(Sig1_rad[n1+1,1:n1],ncol=1)
  
  coords2_rad = rbind(coords_train[N2_ind,],wt)
  Sig2_rad = radgp::Correlationc(coords2_rad, coords2_rad, c(rad_theta_m,10^(-6)), 1, T) 
  Sig2painv_rad = solve(Sig2_rad[1:(n2+1),1:(n2+1)])
  mu2_rad = matrix(Sig2_rad[n2+2,1:(n2+1)],nrow=1) %*% Sig2painv_rad %*% matrix(c(rad_w_m[N2_ind],mu1_rad),ncol=1)
  v2_rad = Sig2_rad[n2+2,n2+2] - matrix(Sig2_rad[n2+2,1:(n2+1)],nrow=1) %*% Sig2painv_rad %*% matrix(Sig2_rad[n2+2,1:(n2+1)],ncol=1)
  coef_rad = (matrix(Sig2_rad[n2+2,1:(n2+1)],nrow=1) %*% Sig2painv_rad)[n2+1]
  
  mu_rad = c(mu1_rad, mu2_rad)
  cov_rad = c(v1_rad) * matrix(c(1,coef_rad,coef_rad,coef_rad^2),nrow=2,ncol=2) + c(v2_rad) * matrix(c(0,0,0,1),nrow=2,ncol=2) + diag(c(1,1))*rad_nugg_m
  loglike_tab[i,1] = -log(2*pi) - 0.5*log(det(cov_rad)) - 0.5* matrix(yob-mu_rad,nrow=1) %*% solve(cov_rad) %*% matrix(yob-mu_rad,ncol=1)
  mse_tab[i,1] = sum((yob-mu_rad)^2)/2
  
  ## Compute log likelihood for vecchia GP
  N1v_ind = order(d1_lst)[1:m]
  N2v_ind = order(d2_lst)[1:m]
  
  coords1_vecchia = rbind(coords_train[N1v_ind,],matrix(wt[1,1:2],nrow=1))
  Sig1_vecchia = radgp::Correlationc(coords1_vecchia, coords1_vecchia, c(vecchia_theta_m,10^(-6)), 1, T) 
  Sig1painv_vecchia = solve(Sig1_vecchia[1:m,1:m])
  mu1_vecchia = matrix(Sig1_vecchia[m+1,1:m],nrow=1) %*% Sig1painv_vecchia %*% matrix(vecchia_w_m[N1v_ind],ncol=1)
  v1_vecchia = Sig1_vecchia[m+1,m+1] - matrix(Sig1_vecchia[m+1,1:m],nrow=1) %*% Sig1painv_vecchia %*% matrix(Sig1_vecchia[m+1,1:m],ncol=1)
  
  coords2_vecchia = rbind(coords_train[N2v_ind,],matrix(wt[2,1:2],nrow=1))
  Sig2_vecchia = radgp::Correlationc(coords2_vecchia, coords2_vecchia, c(vecchia_theta_m,10^(-6)), 1, T) 
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
  geom_histogram(aes(logratio),binwidth=0.1) + geom_vline(xintercept = 0, color='red',size=1.5) +
  geom_vline(xintercept = mean(logratio), color='blue',size=1.5) +
  xlab("Log likelihood ratio") + ylab('Counts') + theme_minimal(base_size = 25)








