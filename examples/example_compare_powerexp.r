library(tidyverse)
library(magrittr)
library(Matrix)
library(reticulate)
library(altdag)

nr <- 3000

set.seed(10)
coords <- runif(nr*2) %>% matrix(ncol=2)

# gen data from full GP
# theta : phi, nu, sigmasq, nugg
CC <- altdag::Correlationc(coords, coords, c(20, 1, 1, 0.01), FALSE, TRUE)
LC <- t(chol(CC))
w <- LC %*% rnorm(nr)
df <- cbind(coords, w) %>% as.data.frame() 
colnames(df) <- c("Var1", "Var2", "w")

ggplot(df, aes(Var1, Var2, color=w)) +
  geom_point() + 
  scale_color_viridis_c()


## TESTING
theta_start <- c(10, 1.01, 1, 1)
metrop_sd <- .2
unif_bounds <- matrix(nrow=4, ncol=2)
unif_bounds[,1] <- 1e-2
unif_bounds[,2] <- 40
unif_bounds[2,] <- c(1, 2-1e-3)

mcmc <- 10000

# check size of conditioning set
M <- 0
system.time({
  dag <- altdagbuild(coords, rho <- .03, M)
})
dag %>% sapply(length) %>% max()

# visualize 
plot(coords, pch=19, cex=.2, col="grey70")
points(coords[dag[[i <- 582]]+1,,drop=F], pch=19, cex=.5, col="red")
points(coords[i,,drop=F], pch=19, cex=.5, col="blue")

# run mcmc on altdag
system.time({
  altdag_model <- altdaggp_response(w, coords, rho=0.03, mcmc, 16,
                                      theta_start, metrop_sd, unif_bounds) })

# alternative: vecchia nngp with maxmin ordering, same mcmc
m <- 10
system.time({
  maxmin_model <- maxmin_mcmc(w, coords, m=m, mcmc, 16,
                                      theta_start, metrop_sd, unif_bounds) })


# visualize MCMC

results_adag <- altdag_model$theta %>% t() %>% as.data.frame() %>% 
  mutate(model="altdag", m=1:n())
colnames(results_adag)[1:4] <- c("phi", "nu", "sigmasq", "tausq")

results_mmin <- maxmin_model$theta %>% t() %>% as.data.frame() %>% 
  mutate(model="maxmin", m=1:n())
colnames(results_mmin)[1:4] <- c("phi", "nu", "sigmasq", "tausq")

results <- bind_rows(results_adag, results_mmin) %>%
  gather(key="param", value="value", -model, -m)


ggplot(results %>% filter(m > 200), aes(m, value)) + 
  geom_line() +
  theme_minimal() +
  facet_wrap(param~model, scales="free", ncol=2)





