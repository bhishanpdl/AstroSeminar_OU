#!/usr/bin/Rscript

# Bhishan Poudel
# Jan 10, 2016
# clear; Rscript rdataframe.r; rm *~
# ref: http://www.programiz.com/r-programming/data-frame
# 5 data types in R : vectors, matrix, list, data frame and a factor

#########################################################################################################
#   DATA-FRAME CREATION
#########################################################################################################
cat("\nWe can create a data frame using the data.frame() function \n")
x <- data.frame("SN"=1:2,"Age"=c(21,15),"Name"=c("John","Dora"))
str(x)
typeof(x[,1])

cat("\nNotice above that the third column, Name is of type factor, instead of a character vector")
cat("\nTo suppress this behavior, we can pass the argument stringsAsFactors=FALSE.\n")
x <- data.frame("SN"=1:2,"Age"=c(21,15),"Name"=c("John","Dora"),stringsAsFactors=FALSE)
str(x)    # now the third column is a character vector

cat("\nMany data input functions of R like, read.table(), read.csv(), read.delim(), read.fwf() also read data into a data frame.")


#########################################################################################################
#   DATA-FRAME COMPONENT ACCESSING (access like a list and access like matrix)
#########################################################################################################
cat("\nAccessing Components in Data Frame \n")
cat("\nAccessing like a list \n")
x["Name"]    # Name 1 John 2 Dora
x$Name       # [1] "John" "Dora"
x[["Name"]]  # [1] "John" "Dora"
x[[3]]       # [1] "John" "Dora"


cat("\nAccessing like a matrix \n")
cat("library(help = 'datasets')\n")
str(trees)
cat("We can see that trees is a data frame with 31 rows and 3 columns\n")
head(trees,n=3)
trees[2:3,]    # select 2nd and 3rd row
y <- trees[trees$Height > 82 || trees$Girth > 13,]    # selects rows with Height greater than 82
print(y)
cat("rows 10,11,12 values of column 2 as a vector\n")
trees[10:12,2]
cat("rows 10,11,12 values of column 2 as a matrix\n")
trees[10:12,2, drop=FALSE]

#########################################################################################################
#   DATA-FRAME MODIFICATION (adding component, deleting component)
#########################################################################################################
cat("\nModifying a Data Frame\n")
x
x[1,"Age"] <- 20; x


cat("\nAdding Components \n")
rbind(x,list(1,16,"Paul"))
cbind(x,State=c("NY","FL"))
x$State <- c("NY","FL"); x



cat("\nDeleting Component \n")
cat("Data frame columns can be deleted by assigning NULL to it. \n")
x$State <- NULL; x
cat("\nSimilarly, rows can be deleted through reassignments. \n")
x <- x[-1,];x
########################################################

