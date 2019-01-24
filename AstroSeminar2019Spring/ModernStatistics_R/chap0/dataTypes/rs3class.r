#!/usr/bin/Rscript

# Bhishan Poudel
# Jan 10, 2016

# clear; Rscript rs3class.r; rm *~
# ref: http://www.programiz.com/r-programming/S3-class
# 5 data types in R : vectors, matrix, list, data frame and a factor
########################################################
# Defining S3 Class and Creating S3 Objects

# create a list with required components
s <- list(name="John", age=21, GPA=3.5)
# name the class appropriately
class(s) <- "student"; s

# a constructor function for the "student" class
student <- function(n,a,g) {
# we can add our own integrity checks
if(g>4 || g<0)  stop("GPA must be between 0 and 4")
value <- list(name=n, age=a, GPA=g)
# class can be set using class() or attr() function
attr(value, "class") <- "student"
value
}

# Here is a sample run where we create objects using this constructor. 
s <- student("Paul", 26, 3.7);s
# we get error for this
#s <- student("Paul", 26, 5) 

# it's up to us to maintain it or not
s$GPA <- 5; s
