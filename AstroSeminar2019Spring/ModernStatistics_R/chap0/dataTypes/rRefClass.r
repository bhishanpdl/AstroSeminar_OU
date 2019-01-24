#!/usr/bin/Rscript

# Bhishan Poudel
# Jan 8, 2016

# clear; Rscript rRefClass.r ; rm *~
# ref: http://programiz.com/r-programming/reference-class

#Reference class in R programming language 
#is similar to the object oriented programming 
#we are used to seeing in common languages 
#like C++, Java, Python etc. Unlike S3 and S4 classes, 
#methods belong to class rather than generic functions. 
#Reference class are internally implemented as 
#S4 classes with an environment added to it.

#Defining Reference Class
#Defining reference class is similar 
#to defining a S4 class. 
#Instead of setClass() we use the setRefClass() function.
library(methods)
setRefClass("student")

#Member variables of a class, if defined, need to be 
#included in the class definition. 
#Member variables of reference class are called fields 
#(analogous to slots in S4 classes). 
#Following is an example to define a class called 
#student with 3 fields, name, age and GPA.

setRefClass("student", fields=list(name="character", age="numeric", GPA="numeric"))

#Creating Reference Objects
#The function setRefClass() returns a generator 
#function which is used to create objects of that class.

student <- setRefClass("student",
fields=list(name="character", age="numeric", GPA="numeric"))

# now student() is our generator function which can be used to create new objects
s <- student(name="John", age=21, GPA=3.5)
s

#Accessing and Modifying Fields
#Fields of the object can be accessed using the $ operator.
s$name
s$name <- "Paul"


#Warning Note:
#In R programming, objects are copied when 
#assigned to new variable or passed to a 
#function (pass by value). 
#For example.

# create list a and assign to b
a <- list("x"=1, "y"=2)
b <- a

# modify b
b$y=3
# a remains unaffected
a

# only b is modified
b

