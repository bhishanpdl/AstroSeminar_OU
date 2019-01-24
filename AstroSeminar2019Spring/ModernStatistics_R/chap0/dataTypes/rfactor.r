#!/usr/bin/Rscript

# Bhishan Poudel
# Jan 10, 2016

# clear; Rscript rfactor.r; rm *~
# ref: http://www.programiz.com/r-programming/factor
# 5 data types in R : vectors, matrix, list, data frame and a factor

########################################################
# Creating a factor
x <- factor(c("single","married","married","single")); x

x <- factor(c("single","married","married","single"), levels=c("single","married","divorced")); x
class(x)
levels(x)
str(x)

########################################################
# Accessing Components in Factor
cat("\n")
x
x[3]           # access 3rd element
x[c(2, 4)]     # access 2nd and 4th element
x[-1]          # access all but 1st element
x[c(TRUE, FALSE, FALSE, TRUE)]  # using logical vectors
########################################################
# Modifying a Factor
cat("\n")
x
cat("\n")
x[2] <- "divorced";x    # modify second element
x[3] <- "widowed" ;x   # cannot assign values outside levels
levels(x) <- c(levels(x), "widowed")    # add new level
x[3] <- "widowed";x


