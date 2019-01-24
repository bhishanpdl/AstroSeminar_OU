#!/usr/bin/Rscript

# Bhishan Poudel
# Jan 8, 2016

# clear; Rscript rlist.r; rm *~
# ref: http://www.programiz.com/r-programming/list

# 5 data types in R : vectors, matrix, list, data frame and a factor

#########################################################################################################
#   LIST CREATION
#########################################################################################################

cat("\nCreating a List")
cat("\nList can be created using the list() function.  \n")


x <- list("a"=2.5, "b"=TRUE, "c"=1:3)
x # $a  $b  $c
typeof(x) # [1] "list"
length(x) # [1] 3
str(x)    # List of 3 $ a: num 2.5 $ b: logi TRUE $ c: int [1:3] 1 2 3
cat("\nIn this example, a, b and c are called tags which makes it easier to reference the components \n")

cat("\nWe can create the same list without the tags as follows \n")
x <- list(2.5,TRUE,1:3)
x # [[1]]  [[2]]  [[3]]


#########################################################################################################
#   LIST ELEMENTS ACCESSING
#########################################################################################################
#cat("\nAccessing Components in List\n")
x <- list("name"="Bhishan", "age"=30, "speaks"="English","Nepalese")
x[c(1:2)]    # index using integer vector
x[-2]        # using negative integer to exclude second components
x[c(T,F,F)]  # index using logical vector

#cat("\nUse of [""] operator to access elements\n")
cat("\n \n")

