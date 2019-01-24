#!/usr/bin/Rscript

# Bhishan Poudel
# Jan 8, 2016
# clear; Rscript rmatrix.r; rm *~
# ref: http://www.programiz.com/r-programming/vector
# 5 data types in R : vectors, matrix, list, data frame and a factor
setwd("~/Copy/2016Spring/RProgramming/presentation/dataTypes/")


#########################################################################################################
#   MATRIX CREATION
#########################################################################################################

cat("\nMatrix is a two dimensional data structure in R.") 
cat("\nWe can check if a variable is a matrix or not with the class() function.") 
cat("\nMatrix is very much similar to vectors but additionally contains the") 
cat("\ndimension attribute. All attributes of an object can be checked with the") 
cat("\nattributes() function (dimension can be checked directly ")
cat("\nwith the dim() function\n")

cat("\nCreating a Matrix \n")
matrix(1:9, nrow=3, ncol=3)


# same result is obtained by providing only one dimension
#matrix(1:9, nrow=3)
matrix(1:9, nrow=3, byrow=TRUE)    # fill matrix row-wise
x <- matrix(1:9, nrow=3, dimnames=list(c("X","Y","Z"),c("A","B","C")))
x

cat("\nThese names can be accessed or changed with colnames() and rownames().\n")
colnames(x)
rownames(x)
# It is also possible to change names
colnames(x) <- c("C1","C2","C3")
rownames(x) <- c("R1","R2","R3")
x


cat("\nAnother way is using cbind() and rbind()\n")
cbind(c(1,2,3),c(4,5,6))
rbind(c(1,2,3),c(4,5,6))

cat("\ncreate a matrix from a vector by setting its dimension using dim(). \n")
x <- c(1,2,3,4,5,6); x; class(x)

dim(x) <- c(2,3); x; class(x)

#########################################################################################################
#   MATRIX CREATION
#########################################################################################################
cat("\nAccessing Elements in Matrix ")
cat("\nElements can be accessed as var[row, column]\n")
cat("\nUsing integer vector as index\n")
x <- matrix(1:9, nrow=3); x
x[c(1,2),c(2,3)]    # select rows 1 & 2 and columns 2 & 3
x[c(3,2),]    # leaving column field blank will select entire columns
x[,]    # leaving row as well as column field blank will select entire matrix
x[-1,]    # select all rows except first
x[1,]; class(x[1,])

cat("Using logical vector as index\n")
cat("Two logical vectors can be used to index a matrix.\n")
x <- cbind(c(4,6,1),c(8,0,2),c(3,7,9)); x
x[c(TRUE,FALSE)] # indexes recycled T F T F
x[c(TRUE,FALSE,TRUE),c(TRUE,TRUE,FALSE)]
x[c(TRUE,FALSE),c(2,3)]   
# TF will be recycled, columns 2 and 3 will be read
cat("extracting matrix elements\n")
x[x>5]    # select elements greater than 5
x[x%%2 == 0]    # select even elements

#example 2
mymat = matrix(1:12,4,3)
mymat = matrix(1:12,ncol=3,byrow=F)
# reading from a data file
m <- matrix(scan('matrix.dat'),nrow=3,byrow=TRUE); m
cat("scanning the numbers from a file\n")
nums = scan('matrix.dat'); nums
######################################
a <- matrix(1:9, nrow = 3)
colnames(a) <- c("A", "B", "C"); a
a[1:2, ]
#>      A B C
#> [1,] 1 4 7
#> [2,] 2 5 8
a[c(T, F, T), c("B", "A")]
# TFT will be recycled, first B will be read then A will be read
#>      B A
#> [1,] 4 1
#> [2,] 6 3
a[0, -2]
#>      A C

#########################################################################################################
#   MATRIX MODIFICATION
#########################################################################################################
cat("\nModifying a Matrix \n")
x <- matrix(1:9, nrow = 3); x
x[2,2] <- 10; x    # modify a single element
x[x<5] <- 0; x    # modify elements less than 5

cat("\ntranspose a matrix \n")
t(x)
cat("\nWe can add row or column using rbind() and cbind() function \n")

cbind(x,c(1,2,3))    # add columns
rbind(x,c(1,2,3))    # add row
x <- x[1:2,]; x    # remove last row

cat("\nDimension of matrix can be modified as well, using the dim() function. \n")
x <- matrix(1:6, nrow = 2); x
dim(x) <- c(3,2); x    # change to 3X2 matrix
dim(x) <- c(1,6); x    # change to 1X6 matrix

cat("\nMatrix Mathematics \n")
x <- matrix(1:9, nrow = 3,byrow=T); x
y <- matrix(1:9, nrow = 3,byrow=T); y
z <- x %*% y; z; 
det(x)
rowSums(x)
rowMeans(x)
z1 <- cbind(x,y); z1

# finding xy'  and x'y
cat("\nxy' is outer product and x'y is cross product\n")
a<- x %o% y ; a # xy'
a<- crossprod(x,y); a


cat("\nfinding inverse using solve\n")
A=rbind(c(1, 2), c(3, 4)); A
solve(A)  # inverse of A

# find inverses using solve
A =rbind(c(1:3),c(0,4,5),c(1,0,6))
det(A)
solve(A)* det(A)

# inverse using ginv
cat("\ninverse of A using ginv is \n")
library(	MASS)
ginv(A)*det(A)

#1 2 3     inverse is 1/22 *   24  -12  -2
#0 4 5                         5   3    -5
#1 0 6                         -4  2    4

cat("\n \n")
cat("\n \n")

# THE END
############################################################################################
#Matrix facilites
#In the following examples, A and B are matrices and x and b are a vectors.
#Operator or Function	Description

#A * B	Element-wise multiplication
#A %*% B	Matrix multiplication
#A %o% B	Outer product. AB'

#crossprod(A,B)
#crossprod(A)	A'B and A'A respectively.

#t(A)	Transpose

#diag(x)	Creates diagonal matrix with elements of x in the principal diagonal

#diag(A)	Returns a vector containing the elements of the principal diagonal

#diag(k)	If k is a scalar, this creates a k x k identity matrix. Go figure.

#solve(A, b)	Returns vector x in the equation b = Ax (i.e., A-1b)

#solve(A)	Inverse of A where A is a square matrix.

#ginv(A)	Moore-Penrose Generalized Inverse of A. 

#ginv(A) requires loading the MASS package.

#y<-eigen(A)	y$val are the eigenvalues of A

#y$vec are the eigenvectors of A

#y<-svd(A)	Single value decomposition of A.

#y$d = vector containing the singular values of A

#y$u = matrix with columns contain the left singular vectors of A 

#y$v = matrix with columns contain the right singular vectors of A

#R <- chol(A)	Choleski factorization of A. Returns the upper triangular factor, such that R'R = A.

#y <- qr(A)	QR decomposition of A. 

#y$qr has an upper triangle that contains the decomposition and a lower triangle that contains information on the Q decomposition.

#y$rank is the rank of A. 

#y$qraux a vector which contains additional information on Q. 

#y$pivot contains information on the pivoting strategy used.

#cbind(A,B,...)	Combine matrices(vectors) horizontally. Returns a matrix.

#rbind(A,B,...)	Combine matrices(vectors) vertically. Returns a matrix.

#rowMeans(A)	Returns vector of row means.

#rowSums(A)	Returns vector of row sums.

#colMeans(A)	Returns vector of column means.

#colSums(A)	Returns vector of column sums.






