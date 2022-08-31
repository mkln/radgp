rm(list=ls())
library(tidyverse)
library(magrittr)
library(Matrix)
#library(reticulate)
library(altdag)

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
theta <- c(4, 2, 1.6, 0.02)

CC <- altdag::Correlationc(coords_all, coords_all, theta, TRUE)
LC <- t(chol(CC))
yall <- LC %*% rnorm(nall) 
df <- cbind(coords_all, yall) %>% as.data.frame() 
colnames(df) <- c("Var1", "Var2", "y")

y_train <- yall[1:ntrain]
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


################################################################# Preliminary 
###############################################################################
# check size of conditioning set
system.time({
  dag <- Raltdagbuild(coords_train, rho <- 0.1)
})
dag$dag %>% sapply(length) %>% mean()

################################################################# MCMC 
###############################################################################
mcmc <- 2500

# using the last rho value used in the preliminary step
altdag_model <- response.model(y_train, coords_train, rho=rho, mcmc, 16)
# posterior mean for theta
altdag_model$theta %>% apply(1, mean)

# vecchia-maxmin estimation MCMC
maxmin_model <- response.model.vecchia(y_train, coords_train, m=25, mcmc, 16)
# posterior mean for theta
maxmin_model$theta %>% apply(1, mean)

# Parameter chains
df_theta <- altdag_model$theta %>% t() %>% as.data.frame() %>% 
  mutate(m = 1:n()) %>% tail(-100)
colnames(df_theta) <- c("phi", "sigmasq", "tausq", "m")
df_theta %<>% tidyr::gather(variable, chain, -m)
ggplot(df_theta, aes(m, chain)) +
  geom_line() +
  theme_minimal() +
  facet_wrap(~ variable, scales="free")

# AltDAG prediction
altdag_predict <- predict(altdag_model, coords_test, 16)

# VecchiaGP prediction
vecchiagp_predict <- predict(maxmin_model, coords_test, 16)


# altdag_predict$yout is a ntest x mcmc matrix of posterior predictive draws
yout_chain <- altdag_predict$yout[,-(1:1000)] # remove first 1000 iterations for burn-in

out_df <- cbind(coords_test, data.frame(
  yout_mean = yout_chain %>% apply(1, median),
  yout_low = yout_chain %>% apply(1, quantile, 0.025),
  yout_high = yout_chain %>% apply(1, quantile, 0.975)
)) 

colnames(out_df) <- c("Var1", "Var2", "y", "y_low", "y_high")

plot_df <- bind_rows(df %>% mutate(sample = "in"),
                     out_df %>% mutate(sample = "out"))

# plot predicttions
ggplot(plot_df %>% filter(sample=="out"), aes(Var1, Var2, fill=y)) +
  geom_raster() +
  scale_fill_viridis_c() +
  theme_minimal()

# plot uncertainty
ggplot(plot_df %>% filter(sample=="out"), aes(Var1, Var2, fill=y_high)) +
  geom_raster() +
  scale_fill_viridis_c() +
  theme_minimal()



################################################################# MLE
###############################################################################
# visualize the likelihood on a grid of phi/sigmasq fixing true tausq
parvals <- expand.grid(seq(1, 30, length.out=30),
                       seq(1, 5, length.out=30),
                       theta[3],
                       theta[4])

grid_dens <- parvals %>% apply(1, \(x) daggp_negdens(y_train, coords_train, dag$dag, as.vector(x), 16))
df <- parvals %>% mutate(dens = grid_dens)
ggplot(df, aes(Var1, Var2, fill=grid_dens, z=grid_dens)) +
  geom_raster() +
  geom_contour(bins=40) +
  scale_fill_viridis_c() + 
  theme_minimal()

# maximum likelihood for covariance parameters using AltDAG
ldens <- function(x) {
  return(-daggp_negdens(y_train, coords_train, dag$dag, exp(x), 16)) }
mle <- optim(c(1,.1,.1), ldens, method="L-BFGS-B", hessian=TRUE)
(theta_mle <- mle$par %>% exp())

# maximum likelihood for covariance parameters using Vecchia Maxmin
ldens <- function(x) {
  return(-daggp_negdens(y_train[maxmin_model$ord], coords_train[maxmin_model$ord,], maxmin_model$dag, exp(x), 16)) }
mle_vecchia <- optim(c(1,.1,.1), ldens, method="L-BFGS-B", hessian=TRUE)
(theta_mle_vecchia <- mle_vecchia$par %>% exp())


################################################################# visualization
###############################################################################

# compare parent sets for altdag v vecchia maxmin
# plot altdag parent sets
par(mfrow=c(1,2))

i <- 1
plot( coords_train[i,,drop=FALSE], pch=19, cex=1.2, xlim = c(0,1), ylim=c(0,1),
      xlab="x1", ylab="x2", main="ALTDAG")
points( coords_train[dag$dag[[i]]+1,,drop=FALSE], pch=19, cex=.8, col="red" )

# plot vecchia maxmin dag parent sets
vecchia_maxmin <- maxmin_model$dag
vecchia_ord <- maxmin_model$ord
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





