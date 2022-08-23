library(Matrix)
library(altdag)


coords <- as.matrix(expand.grid(xx <- seq(0,1,length.out=10), xx))
# phi sigmasq nugget
theta <- c(1,1,0)
rho <- 0.2

altgp_out <- altdaggp(coords, theta, 0.2)

Hmat_a <- altgp_out$H
ord <- order(altgp_out$layers)


# H matrix is the lower-triangular "cholesky" factor of the precision matrix
# using the "ord" order
image(Hmat_a)
image(Hmat_a[ord,ord])

altdag_prec <- tcrossprod(Hmat_a)
image(altdag_prec)



# same thing with NNGP/maxmin Vecchia
m <- 15

# make DAG
ixmm <- GPvecchia::order_maxmin_exact(coords)
coords_mm <- coords[ixmm,]
nn_dag_mat <- GPvecchia:::findOrderedNN_kdtree2(coords_mm, m)
nn_dag <- apply(nn_dag_mat, 1, function(x){ x[!is.na(x)][-1]-1 })

# get stuff
vecchia_out <- vecchiagp(coords_mm, theta, nn_dag)
Hord <- vecchia_out$H
# this is already ordered
image(Hord)
# original order
ord <- order(ixmm)
Hmat_v <- Hord[ord, ord]

image(Hmat_v)

vecchia_prec <- tcrossprod(Hmat_v)
image(vecchia_prec)

