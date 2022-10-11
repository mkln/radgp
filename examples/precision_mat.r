library(Matrix)
library(aptdag)
library(dplyr)

coords <- cbind(runif(400), runif(400))
  #as.matrix(expand.grid(xx <- seq(0,1,length.out=20), xx))
# phi sigmasq nugget
theta <- c(1,1,1.5,0)
rho <- 1

aptgp_out <- aptdaggp(coords, theta, .2)

Hmat_a <- aptgp_out$H
ord <- order(aptgp_out$layers)

layers_o <- aptgp_out$layers[ord,] %>% as.numeric()
coords_o <- coords[ord,]
Hord_a <- Hmat_a[ord, ord]
Aord_a <- aptgp_out$A[ord, ord]

image(Hord_a)

i <- 399
parents_ofi <- setdiff(which(Hord_a[i,] != 0), i)
length(parents_ofi)

aptdag_n_children <- (Hord_a %>% apply(2, \(x) sum(x!=0)-1))
aptdag_n_parents <- (Hord_a %>% apply(1, \(x) sum(x!=0)-1))


overlap_mat <- matrix(0, nrow(coords), nrow(coords))
add_mat <- matrix(0, nrow(coords), nrow(coords))

for(i in 1:nrow(coords)){
  cat(i,"\n")
  parents_ofi <- setdiff(which(Hord_a[i,] != 0), i)
  for(j in parents_ofi){
    parents_ofj <- setdiff(which(Hord_a[j,] != 0), j)
    isectn <- length(intersect(parents_ofi, parents_ofj))
    isectport <- isectn / length(parents_ofi)
    add_pars <- length(parents_ofi)-isectn
    #cat(j, " --> ", isectn," = ", isectport, ", rem: ", add_pars, "\n")
    
    overlap_mat[i, j] <- isectn
    add_mat[i,j] <- add_pars
  }
}

add_mat_min <- apply(add_mat, 1, \(x) min(x[x!=0]))
cbind(aptdag_n_parents, add_mat_min)

j <- 372
layers_o[j]

layers_o[parents_of]

children_of <- setdiff(which(Hord_a[,i] != 0), i)








plot(coords_o[which(layers_o == 1),], pch=19, cex=1, xlim=c(0,1), ylim=c(0,1))
points(coords_o[i,,drop=F], pch=19, cex=1, col="blue")
points(coords_o[parents_of,,drop=F], pch=19, cex=1, col="red")
points(coords_o[children_of,,drop=F], pch=19, cex=1, col="orange")


Ci_ad <- crossprod(Hord_a)
cholCi <- Matrix::Cholesky(Ci_ad)
perm <- cholCi@perm
image(Ci_ad)













# H matrix is the lower-triangular "cholesky" factor of the precision matrix
# using the "ord" order
image(Hmat_a)
image(Hmat_a[ord,ord])

aptdag_prec <- crossprod(Hmat_a)
image(aptdag_prec)


CC <- aptdag::Correlationc(coords, coords, theta, T)
image(solve(crossprod(Hmat_a)))


# same thing with NNGP/maxmin Vecchia
m <- 25

# make DAG
ixmm <- GPvecchia::order_maxmin_exact(coords)
coords_mm <- coords[ixmm,]
nn_dag_mat <- GPvecchia:::findOrderedNN_kdtree2(coords_mm, m)
nn_dag <- apply(nn_dag_mat, 1, function(x){ x[!is.na(x)][-1]-1 })

# get stuff
vecchia_out <- vecchiagp(coords_mm, theta, nn_dag)
Hord_v <- vecchia_out$H
Aord_v <- vecchia_out$A
# this is already ordered
image(Hord_v)
image(Aord_v)

oin <- 1:200
out <- 201:400

set.seed(1)
u <- rnorm(400)
uoin <- u[oin]
uout <- u[out]

U <- t(Hord_v)
Q <- crossprod(Hord_v) #Qout <- Q[out, out]
Sigma <- solve(Q)
L <- t(chol(Sigma))

w0 <- L %*% u

Koi <- Sigma[out, oin]
Kin <- Sigma[oin, oin]
Lin <- t(chol(Kin))
win <- Lin %*% uoin
Kout <- Sigma[out, out]
H <- Koi %*% solve(Kin)
Rout <- Kout - H %*% t(Koi)
wout <- H %*% win + t(chol(forceSymmetric(Rout))) %*% uout

w1 <- c(win@x, wout@x)

w2_i <- win@x

test <- pred_from_dag(coords_mm, vecchia_out$dag, theta, u)

i <- 150
parents_of <- setdiff(which(Hord_v[i,] != 0), i)
children_of <- setdiff(which(Hord_v[,i] != 0), i)

plot(coords_mm, pch=19, cex=.5)
points(coords_mm[i,,drop=F], pch=19, cex=.8, col="blue")
points(coords_mm[parents_of,,drop=F], pch=19, cex=.8, col="red")
points(coords_mm[children_of,,drop=F], pch=19, cex=.8, col="orange")

vecchia_n_children <- sort(Hord_v %>% apply(2, \(x) sum(x!=0)-1))
vecchia_n_parents <- sort(Hord_v %>% apply(1, \(x) sum(x!=0)-1))




overlap_mat <- matrix(0, nrow(coords), nrow(coords))
add_mat <- matrix(0, nrow(coords), nrow(coords))

for(i in 1:nrow(coords)){
  cat(i,"\n")
  parents_ofi <- setdiff(which(Hord_v[i,] != 0), i)
  for(j in parents_ofi){
    parents_ofj <- setdiff(which(Hord_v[j,] != 0), j)
    isectn <- length(intersect(parents_ofi, parents_ofj))
    isectport <- isectn / length(parents_ofi)
    add_pars <- length(parents_ofi)-isectn
    #cat(j, " --> ", isectn," = ", isectport, ", rem: ", add_pars, "\n")
    
    overlap_mat[i, j] <- isectn
    add_mat[i,j] <- add_pars
  }
}

add_mat_min <- apply(add_mat, 1, \(x) min(x[x!=0]))















plot(aptdag_n_children, type='l', ylim=c(0,5000), col="orange")
lines(aptdag_n_parents^3, col="red")
lines(vecchia_n_children, col="cyan")
lines(vecchia_n_parents^3, col="blue")



# original order
ord <- order(ixmm)
Hmat_v <- Hord[ord, ord]

image(Hmat_v)

vecchia_prec <- crossprod(Hmat_v)
image(vecchia_prec)

