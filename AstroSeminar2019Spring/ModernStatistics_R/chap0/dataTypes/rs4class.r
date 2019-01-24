#!/usr/bin/Rscript

# Bhishan Poudel
# Jan 8, 2016

# clear; Rscript rs4class.r ; rm *~
# ref: http://www.programiz.com/r-programming/S4-class
# 5 data types in R : vectors, matrix, list, data frame and a factor

#S4 class is defined using the setClass() function. 
#In R terminology, member variables are called slots. 
#While defining a class, we need to set the name and the slots 
#(along with class of the slot) it is going to have. 

# Creating S4 Objects
# create an object using new()
# provide the class name and value for slots

library(methods)
setClass("student", slots=list(name="character", age="numeric", GPA="numeric"))
s <- new("student",name="John", age=21, GPA=3.5); s

# if we do not want to use library methods,
# we have to add namespace methods:: each time.
methods::setClass("student", slots=list(name="character", age="numeric", GPA="numeric"))
s <- methods::new("student",name="John", age=21, GPA=3.5); s

# We can check if an object is an S4 object through the function isS4(). 
isS4(s)

#The function setClass() returns a generator function. 
#This generator function (usually having same name as 
#the class) can be used to create new objects. 
#This function acts as a constructor.
student <- setClass("student", slots=list(name="character", age="numeric", GPA="numeric"))
student

#Now we can use this constructor function 
#to create new objects. Note above that our 
#constructor in turn uses the new() function 
#to create objects. 
#It is just a wrap around. 
student(name="John", age=21, GPA=3.5)


#Accessing and Modifying Slot
s@GPA
s@GPA <- 3.7; s
slot(s,"name")
slot(s,"name") <- "Paul";s

