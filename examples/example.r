rm(list=ls())
library(tidyverse)
library(magrittr)
library(Matrix)
library(aptdag)

################################################################# data
###############################################################################
ntrain <- 1500

set.seed(10)

coords_train <- runif(ntrain * 2) %>% matrix(ncol=2)

# test set of size 25^2 = 625
coords_test <- expand.grid(xout <- seq(0,1,length.out=25), xout) %>% as.matrix()
ntest <- nrow(coords_test)

coords_all <- rbind(coords_train, coords_test)
nall <- nrow(coords_all)

# gen data from full GP
# theta : phi, sigmasq, nu, nugg
theta <- c(30, 1, 1.6, 0)
nugget <- .1

CC <- aptdag::Correlationc(coords_all, coords_all, theta, TRUE)
LC <- t(chol(CC))
wall <- LC %*% rnorm(nall) 
yall <- wall + nugget^.5 * rnorm(nall)
df <- cbind(coords_all, yall) %>% as.data.frame() 
colnames(df) <- c("Var1", "Var2", "y")

y_train <- yall[1:ntrain]
w_train <- wall[1:ntrain]
y_test <- tail(yall, ntest)

# visualize data on the gridded test set
df %>% tail(ntest) %>% 
  ggplot(aes(Var1, Var2, fill=y)) + 
  geom_raster() + 
  scale_fill_viridis_c() + 
  theme_minimal()

# visualize data on the training set
df %>% head(ntrain) %>% 
  ggplot(aes(Var1, Var2, color=y)) +
  geom_point() + 
  scale_color_viridis_c() +
  theme_minimal()

################################################################# MCMC 
###############################################################################
mcmc <- 500

## setup MCMC for theta and nugget
## common to all models

theta_start <- c(10, 1, 1.5)
theta_unif_bounds <- matrix(nrow=3, ncol=2)
theta_unif_bounds[1,] <- c(5, 100) # phi
theta_unif_bounds[2,] <- c(.2, 10) # sigmasq
theta_unif_bounds[3,] <- c(1.001, 2-.001) # nu

nugget_start <- c(.5)



#################################################################### Apt DAG GP
###############################################################################
# Preliminary : check size of conditioning set
system.time({
  dag <- Raptdagbuild(coords_train, rho <- 0.1)
})
dag$dag %>% sapply(length) %>% mean()
# visual
plot(coords_train[10,,drop=F], xlim=c(0,1), ylim=c(0,1), pch=19, cex=.8, col="blue")
points(coords_train[dag$dag[[10]]+1,], pch=19, cex=.8, col="red")

################### Apt DAG GP ~ latent model
system.time({
  aptdag_latent <- latent.model(y_train, coords_train, rho=rho, 
                               theta_start=theta_start,
                               theta_prior=theta_unif_bounds,
                               mcmc=mcmc, n_threads=16, printn=10)
})
# posterior mean for theta
aptdag_latent$theta %>% apply(1, mean)
# predictions
aptdag_predict_latent <- predict(aptdag_latent, coords_test, n_threads=16)


################### Apt DAG GP ~ response model
system.time({
  aptdag_response <- response.model(y_train, coords_train, rho=rho, 
                               theta_start=theta_start,
                               theta_prior=theta_unif_bounds,
                               mcmc=mcmc, n_threads=16, printn=10)
})
# posterior mean for theta
aptdag_response$theta %>% apply(1, mean)
# predictions
aptdag_predict <- predict(aptdag_response, coords_test, n_threads=16)


############################################################# Vecchia MaxMin GP
###############################################################################

############ Vecchia MaxMin GP ~ latent model
system.time({
  vecchia_latent <- latent.model.vecchia(y_train, coords_train, m=25,
                                         theta_start=theta_start,
                                         theta_prior=theta_unif_bounds,
                                         mcmc=mcmc, n_threads=16, printn=10)
})
# posterior mean for theta
vecchia_latent$theta %>% apply(1, mean)
# predictions
vecchiagp_predict_latent <- predict(vecchia_latent, coords_test, n_threads=16, independent=FALSE)
vecchiagp_predict_i_latent <- predict(vecchia_latent, coords_test, n_threads=16, independent=TRUE)

############ Vecchia MaxMin GP ~ response model
system.time({
  vecchia_response <- response.model.vecchia(y_train, coords_train, m=25, 
                                         theta_start=theta_start,
                                         theta_prior=theta_unif_bounds,
                                         mcmc=mcmc, n_threads=16, printn=10)
})
# posterior mean for theta
vecchia_response$theta %>% apply(1, mean)
# predictions
vecchiagp_predict <- predict(vecchia_response, coords_test, n_threads=16, independent=FALSE)
vecchiagp_predict_i <- predict(vecchia_response, coords_test, n_threads=16, independent=TRUE)



######################################################@@@####### Postprocessing
###############################################################################

# choose which model to target for plots
model_fit <- vecchia_latent
model_predicts <- vecchiagp_predict_latent

# recovery of latent process (latent models only)
df <- coords_train %>% data.frame() %>% mutate(
  yt = y_train,
  wt = w_train,
  w = model_fit$w %>% apply(1, mean),
  ess = model_fit$w %>% apply(1, coda::effectiveSize))
ggplot(df, aes(X1, X2, color=wt)) +
  geom_point() +
  scale_color_viridis_c() + 
  theme_minimal()

# Parameter chains
df_params <- model_fit$theta %>% t() %>% cbind(model_fit$nugg) %>% as.data.frame() %>% 
  mutate(m = 1:n()) %>% tail(-100)
colnames(df_params) <- c("phi", "sigmasq", "exponent", "nugget", "m")
df_params %<>% tidyr::gather(variable, chain, -m)
ggplot(df_params, aes(m, chain)) +
  geom_line() +
  theme_minimal() +
  facet_wrap(~ variable, scales="free")


# RMSPE
sqrt(mean((apply(model_predicts$yout,1,mean)-y_test)^2))

# aptdag_predict$yout is a ntest x mcmc matrix of posterior predictive draws
yout_chain <- model_predicts$yout

out_df <- cbind(coords_test, data.frame(
  yout_mean = yout_chain %>% apply(1, median),
  yout_low = yout_chain %>% apply(1, quantile, 0.025),
  yout_high = yout_chain %>% apply(1, quantile, 0.975)
)) 
colnames(out_df) <- c("Var1", "Var2", "y", "y_low", "y_high")

plot_df <- bind_rows(df %>% mutate(sample = "in"),
                     out_df %>% mutate(sample = "out"))

# plot predictions
ggplot(plot_df %>% filter(sample=="out"), aes(Var1, Var2, fill=y)) +
  geom_raster() +
  scale_fill_viridis_c() +
  theme_minimal()



################################################################## Response MLE
###############################################################################
# visualize the likelihood on a grid of phi/sigmasq fixing true nugget
parvals <- expand.grid(seq(5, 50, length.out=30),
                       seq(0.1, 2, length.out=30),
                       theta[3],
                       nugget)

grid_dens <- parvals %>% apply(1, \(x) daggp_negdens(y_train, coords_train, dag$dag, as.vector(x), 16))
df <- parvals %>% mutate(dens = grid_dens)
ggplot(df, aes(Var1, Var2, fill=grid_dens, z=grid_dens)) +
  geom_raster() +
  geom_contour(bins=40) +
  scale_fill_viridis_c() + 
  theme_minimal()

# maximum likelihood for covariance parameters using AltDAG
ldens <- function(x) {
  return(-daggp_negdens(y_train, coords_train, dag$dag, x, 16)) }
aptdag_mle <- optim(c(1,.1,1.5,.1), ldens, method="L-BFGS-B", 
             lower=c(1e-9, 1e-9, 1, 1e-9), upper=c(50, Inf, 2, 1), hessian=TRUE)
(apt_parms_mle <- aptdag_mle$par)

# maximum likelihood for covariance parameters using Vecchia Maxmin
v_ord <- vecchia_response$ord
ldens <- function(x) {
  return(-daggp_negdens(y_train[v_ord], coords_train[v_ord,], 
                        vecchia_response$dag, x, 16)) }
vecchia_mle <- optim(c(1,.1,1.5,.1), ldens, method="L-BFGS-B",
                     lower=c(1e-9, 1e-9, 1, 1e-9), upper=c(50, Inf, 2, 1), hessian=TRUE)
(vecchia_parms_mle <- vecchia_mle$par)


################################################################# visualization
###############################################################################

# compare parent sets for aptdag v vecchia maxmin
# plot aptdag parent sets
par(mfrow=c(1,2))

i <- 10
plot( coords_train[i,,drop=FALSE], pch=19, cex=1.2, xlim = c(0,1), ylim=c(0,1),
      xlab="x1", ylab="x2", main="ALTDAG")
points( coords_train[dag$dag[[i]]+1,,drop=FALSE], pch=19, cex=.8, col="red" )

# plot vecchia maxmin dag parent sets
vecchia_maxmin <- vecchia_response$dag
vecchia_ord <- vecchia_response$ord
iv <- 1400 #which(vecchia_ord == i)
coords_ord <- coords_train[vecchia_ord,]
plot( coords_ord[iv,,drop=FALSE], pch=19, cex=2, xlim = c(0,1), ylim=c(0,1),
      xlab="x1", ylab="x2", main="Vecchia MAXMIN")
points( coords_ord[vecchia_maxmin[[iv]]+1,,drop=FALSE], pch=19, cex=.8, col="red" )

par(mfrow=c(1,1))



# partitioning of AltDAG.
df_layers <- coords_train %>% cbind(data.frame(layers = dag$layers)) 
colnames(df_layers) <- c("Var1", "Var2", "layers")

# partitioning into layers, all colors at once
ggplot(df_layers, aes(Var1, Var2, color=factor(layers))) +
  geom_point() +
  theme_minimal()

# plot locations belonging to one of the layers only
ggplot(df_layers %>% filter(layers == 2), aes(Var1, Var2, color=factor(layers))) +
  geom_point() +
  theme_minimal()





