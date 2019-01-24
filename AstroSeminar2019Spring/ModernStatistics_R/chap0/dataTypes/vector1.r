#!/usr/bin/Rscript

# Bhishan Poudel
# Jan 8, 2016
# 5 data types in R : vectors, matrix, list, data frame and a factor

#########################################################################################################
#   VECTORS CREATION
#########################################################################################################
# Creating a Vector
# cat("\nCreating a Vector")
# cat("\nVectors are generally created using the c() function. \n")
# cat("\nVectors contain element of the same type. \nDifferent data types ")
# cat("available in R are logical, integer, double, character, complex and raw.\n")
# cat("Since, a vector must have elements of the same type\n")
# cat("this function will try and coerce elements to the same type, \nif they are different.")
# cat("Coercion is from lower to higher types from logical to integer to double to character.\n")
setwd("~/Copy/2016Spring/RProgramming/presentation/dataTypes/")

# numeric vectors
x <- c(1, 5, 4, 9, 0)
print(x)

typeof(x)
# [1] "double"


length(x)

x[order(x)]
# character vectors


y <- setNames(x, letters[3:4])
y[c("d", "c", "a")]

y <- setNames(x, 2:6)

y[c("d", "c", "a")]

# x 1, 5, 4, 9, 0 
# y a  b  c  d  e
 
# example 2
x <- c(1, 5.4, TRUE, "hello")
x

typeof(x)
# [1] "character"  # logical to integer to double to character

#########################################################################################################
#   VECTORS CREATION USING SEQUENCE
#########################################################################################################
# cat("\nCreating vectors using seq(arguments, by=.., length.out=..,etc) \n")
# cat("help(seq) \n")
# 
# cat("seq(from = 1, to = 1, by = ((to - from)/(length.out - 1)),")
# cat("length.out = NULL,3 along.with = NULL, ...) \n")
# cat("seq() examples\n")

seq(10)
seq_len(10)
seq(0:10)
seq(5:1)
seq(-5)
seq(1:-5)  # from 1 to 7
seq(0,10,2)
seq(0, 1, length.out = 11)
# [1] 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
seq(stats::rnorm(20)) # r is random varible and norm is normal distrubution
# [1]  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
seq(1, 9, by = 2)     # matches 'end'

seq(1, 9, by = pi)    # stays below 'end'

seq(1, 6, by = 3)

#########################################################################################################
#   VECTORS ELEMENTS ACCESSING (using [], using c(), using logical )
#########################################################################################################

# cat("\nAccessing Elements in Vector")
# cat("\nVector index in R starts from 1, unlike most programming languages")
# cat("where index start from 0. \n")
# # help(":")


x <- 1:7; x
x[c(2, 4,2)]     # access 2nd and 4th element
#x[c(2, -4)]     # cannot mix positive and negative integers
x[c(2.4, 3.54)]  # real numbers are truncated to integers
# [1] 2 3
x[-c(3, 1)]



cat("\nUsing logical vector \n")
# consider the following vector s of length 5. 
s = c("aa", "bb", "cc", "dd", "ee")

# To retrieve the the second and fourth members of s, 
# we define a logical vector L of the same length, 
# and have its second and fourth members set as TRUE.

L = c(FALSE, TRUE, FALSE, TRUE, FALSE) 
s[L] 
# [1] "bb" "dd"

# The code can be abbreviated into a single line.
s[c(FALSE, TRUE, FALSE, TRUE, FALSE)] 
# [1]      "bb"         "dd"  




# x <- 1:7; x
# #                                                                 1 2 3 4 5 6 7
# x[c(TRUE, FALSE)] # [1] 1 3 5 7                                   T F T F T F T
# x[c(TRUE, FALSE, FALSE)] # [1] 1 4 7                              T F F T F F T
# x[c(TRUE, FALSE, FALSE, TRUE)]# [1] 1 4 5                         T F F T T F F
# x[c(TRUE, FALSE, FALSE, TRUE,FALSE)] # [1] 1 4 6                  T F F F F T F
# x[c(TRUE, FALSE, FALSE, TRUE,FALSE,TRUE)]  # [1] 1 4 6 7          T F F T F T T
# x[c(TRUE, FALSE, FALSE, TRUE,FALSE,TRUE,F)]  # [1] 1 4 6          T F F T F T F
# #   1     2      3      4    5     6     7
# 
# x[c(TRUE, TRUE, NA, FALSE)]

cat("\nfiltering vectors based on conditions \n")
x[x<4]  # filtering vectors based on conditions
#[1] -3 -1

x
x[x>=4]
#[1] 3


cat("\nUsing character vector as index \n")   # this is similar to factors
x <- c("first"=3, "second"=0, "third"=9)
names(x)
# [1] "first"  "second" "third" 

x["second"]
#second 
#    0 

x[c("first", "third")]
#first third 
#   3     9

#########################################################################################################
#   VECTORS MODIFICATION
#########################################################################################################
cat("\nModifying a Vector \n")
x <- -3:2 ;x
#[1] -3 -2 -1  0  1  2

x[2] <- 0; x        # modify 2nd element
#[1] -3  0 -1  0  1  2

x[x<0] <- 5; x      # modify elements less than 0
#[1] 5 0 5 0 1 2

x <- x[1:4]; x      # truncate x to first 4 elements
#[1] 5 0 5 0

#########################################################################################################
#   VECTORS DELETION
#########################################################################################################
cat("\nDeleting a Vector")
cat("\nWe can delete a vector by simply assigning a NULL to it.  \n")
x <- -3:2 ;x
# [1] -3 -2 -1  0  1  2
x <- NULL

x
#NULL
x[4]
#NULL



