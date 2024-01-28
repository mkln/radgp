library(radgp)
library(ggplot2)
library(transport)
library(rSPDE)
library(MASS)
library(reshape2)
library(latex2exp)
library(cowplot)

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
set.seed(50)
mc_true = 1000
mcmc = 2500

## Matern covariance function
theta <- c(20, 1, 5/2, 0)
matern_hint_fun <- function(phi, dist=0.15, sigmasq=1, thre=0.05){
  return( thre - sigmasq*(1+phi*dist) * exp(-phi*dist) )
}
theta[1] = dichotomy_solver(matern_hint_fun, 10, 50)


## nugget for latent model
nugget <- 1e-2

## training data
nl = 40
ntrain = nl^2
coords_train = as.matrix(expand.grid(xout<-seq(0,1,length.out=nl),xout))
CC_train <- radgp::Correlationc(coords_train, coords_train, theta, 1, T)  ## 0 for power exponential, anything else for matern
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

phi_records = matrix(0, nrow=length(rho_lst)+length(m_lst), ncol=mc_true)
tau2_records = matrix(0, nrow=length(rho_lst)+length(m_lst), ncol=mc_true)
nugg_records = matrix(0, nrow=length(rho_lst)+length(m_lst), ncol=mc_true)
cg_records = matrix(0, nrow=length(rho_lst)+length(m_lst), ncol=mcmc)

## uniform test data
ntest = 1000
nall = ntrain + ntest
coords_test = matrix(runif(ntest*2), nrow=ntest, ncol=2)
coords_all = rbind(coords_train, coords_test)
CC <- radgp::Correlationc(coords_all, coords_all, theta, 1, T)
z_train <- LC_train %*% rnorm(ntrain) 
y_train = z_train + nugget^.5 * rnorm(ntrain)



##----------------------------------mcmc----------------------------------------
theta_start <- c(40, 1, theta[3])
theta_unif_bounds <- matrix(nrow=3, ncol=2)
theta_unif_bounds[1,] <- c(1, 100) # phi
theta_unif_bounds[2,] <- c(.1, 10) # sigmasq
theta_unif_bounds[3,] <- c(theta[3]-0.001, theta[3]+0.001) # nu
# theta_unif_bounds[3,] <- c(0.1, 10)
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
  phi_records[j,] = radgp_obj$theta[1,(mcmc-mc_true+1):mcmc]
  tau2_records[j,] = radgp_obj$theta[2,(mcmc-mc_true+1):mcmc]
  nugg_records[j,] = radgp_obj$nugg[(mcmc-mc_true+1):mcmc]
  cg_records[j,] = radgp_obj$cg_iter
  
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
  phi_records[length(rho_lst)+j,] = nngp_obj$theta[1,(mcmc-mc_true+1):mcmc]
  tau2_records[length(rho_lst)+j,] = nngp_obj$theta[2,(mcmc-mc_true+1):mcmc]
  nugg_records[length(rho_lst)+j,] = nngp_obj$nugg[(mcmc-mc_true+1):mcmc]  
  cg_records[length(rho_lst)+j,] = nngp_obj$cg_iter

  print(paste('Inner loop finished: ', length(rho_lst)+j, '/', length(rho_lst)+length(m_lst), sep=''))
  print(Sys.time() - t2)
  t2 = Sys.time()
}
  



##-------------------------------- plot results --------------------------------
conf_df <- function(xall, datall, n1){
  df = data.frame(matrix(0,nrow=length(xall), ncol=4))
  colnames(df) <- c('Nonzeros', 'mean', 'low', 'up')
  df$Method = 'RadGP'
  l = 1
  alpha = 0.1
  nsamples = ncol(datall)
  left = floor(nsamples*alpha/2)
  right = floor(nsamples-nsamples*alpha/2)
  for (j in 1:length(xall)){
    samples = datall[j,]
    samples = sort(samples)
    df$Nonzeros[l] = xall[j]
    if (j>n1){
      df$Method[l] = 'NNGP'
    }
    df$low[l] = mean(samples[left:(left+1)])
    df$up[l] = mean(samples[right:(right+1)])
    df$mean[l] = mean(samples)
    l = l + 1
  }
  return(df)
}


df_phi = conf_df(ave_nonzeros, phi_records, length(rho_lst))
p_phi = ggplot(df_phi) +
  geom_line(aes(x=Nonzeros,y=mean,color=Method)) +
  geom_ribbon(aes(x=Nonzeros,ymin=low,ymax=up,fill=Method),alpha=0.2) +
  xlab('Precision Nonzeros') + ylab(TeX('$\\Phi$')) + theme_minimal(base_size = 20) +
  geom_hline(yintercept = theta[1]) + theme(legend.position='none') +
  scale_colour_manual(values = c("#F8766D", "#619CFF", 'black')) + 
  scale_fill_manual(values = c("#F8766D", "#619CFF", 'black')) 

df_tau2 = conf_df(ave_nonzeros, tau2_records, length(rho_lst))
p_tau2 = ggplot(df_tau2) +
  geom_line(aes(x=Nonzeros,y=mean,color=Method)) +
  geom_ribbon(aes(x=Nonzeros,ymin=low,ymax=up,fill=Method),alpha=0.2) +
  xlab('Precision Nonzeros') + ylab(TeX('$\\tau^2$')) + theme_minimal(base_size = 20) +
  geom_hline(yintercept = theta[2]) + theme(legend.position='none') +
  scale_colour_manual(values = c("#F8766D", "#619CFF", 'black')) + 
  scale_fill_manual(values = c("#F8766D", "#619CFF", 'black')) 
  
df_nugg = conf_df(ave_nonzeros, nugg_records, length(rho_lst))
p_nugg = 
  ggplot(df_nugg) +
  geom_line(aes(x=Nonzeros,y=mean,color=Method)) +
  geom_ribbon(aes(x=Nonzeros,ymin=low,ymax=up,fill=Method),alpha=0.2) +
  geom_hline(aes(yintercept = nugget,color='Truth')) +
  xlab('Precision Nonzeros') + ylab(TeX('Nugget $\\sigma^2$')) + theme_minimal(base_size = 20) +
  scale_colour_manual(values = c("#F8766D", "#619CFF", 'black')) + 
  scale_fill_manual(values = c("#F8766D", "#619CFF", 'black')) + guides(fill="none")
  
plot_grid(p_phi, p_tau2, p_nugg, align='h', nrow=1, rel_widths=c(0.31,0.31,0.38))

df_cg = conf_df(ave_nonzeros, cg_records, length(rho_lst))
p_cg = 
  ggplot(df_cg) +
  geom_line(aes(x=Nonzeros,y=mean,color=Method)) +
  geom_ribbon(aes(x=Nonzeros,ymin=low,ymax=up,fill=Method),alpha=0.2) +
  xlab('Precision Nonzeros') + ylab('CG Iterations') + theme_minimal(base_size = 20)




##-----------------------Exam posterior Mixing---------------------------
j1 = 2
j2 = 2
rho = rho_lst[j1]
m = m_lst[j2]
radgp_obj <- latent.model(y_train, coords_train, rho,
                          theta_start=theta_start,
                          theta_prior=theta_unif_bounds,
                          nugg_start=nugget,
                          nugg_prior=nugget_prior,
                          mcmc=mcmc, n_threads=16, covariance_model = 1, printn=3)

nngp_obj <- latent.model.vecchia(y_train, coords_train, m,
                                 theta_start=theta_start,
                                 theta_prior=theta_unif_bounds,
                                 nugg_start=nugget,
                                 nugg_prior=nugget_prior,
                                 mcmc=mcmc, n_threads=16, covariance_model = 1, printn=3)
df_trace = data.frame(c(1:mcmc,1:mcmc), c(radgp_obj$theta[1,],nngp_obj$theta[1,]), 
                      c(radgp_obj$theta[2,],nngp_obj$theta[2,]),
                      c(radgp_obj$nugg,nngp_obj$nugg))
colnames(df_trace) <- c('mcmc', 'phi', 'tau2', 'nugg')
df_trace$Method = 'RadGP'
df_trace$Method[(mcmc+1):(2*mcmc)] = 'NNGP'



p_t1 = 
  ggplot(df_trace) +
  geom_line(aes(x=mcmc,y=phi,color=Method)) +
  xlab('Precision Nonzeros') + ylab(TeX('$\\Phi$')) + theme_minimal(base_size = 20) +
  geom_hline(yintercept = theta[1]) + theme(legend.position='none') +
  scale_colour_manual(values = c("#F8766D", "#619CFF", 'black'))

p_t2 = 
  ggplot(df_trace) +
  geom_line(aes(x=mcmc,y=tau2,color=Method)) +
  xlab('Precision Nonzeros') + ylab(TeX('$\\tau^2$')) + theme_minimal(base_size = 20) +
  geom_hline(yintercept = theta[2]) + theme(legend.position='none') +
  scale_colour_manual(values = c("#F8766D", "#619CFF", 'black')) 

p_t3 = ggplot(df_trace) +
  geom_line(aes(x=mcmc,y=nugg,color=Method)) +
  xlab('Precision Nonzeros') + ylab(TeX('$Nugget \\sigma^2$')) + theme_minimal(base_size = 20) +
  geom_hline(yintercept = nugget) +
  scale_colour_manual(values = c("#F8766D", "#619CFF", 'black')) 

a1 = acf(df_trace$phi)[['acf']][,1,1]
a2 = acf(df_trace$tau2)[['acf']][,1,1]
a3 = acf(df_trace$nugg)[['acf']][,1,1]
n_ac = length(a1)
df_ac = data.frame(c(1:n_ac,1:n_ac,1:n_ac),c(a1,a2,a3),
                c(rep('phi',n_ac),rep('tau2',n_ac),rep('nugg',n_ac)))
colnames(df_ac) <- c('Lag','ACF','Variable')
p_ac = 
  ggplot(df_ac) +
  geom_line(aes(x=Lag,y=ACF,color=Variable)) +
  ylab('Auto Correlation') + theme_minimal(base_size = 20)

plot_grid(p_t1, p_t2, p_t3, align='h', nrow=1, rel_widths=c(0.31,0.31,0.38))
plot_grid(p_cg, p_ac, align='h', nrow=1, rel_widths=c(0.5, 0.5))



