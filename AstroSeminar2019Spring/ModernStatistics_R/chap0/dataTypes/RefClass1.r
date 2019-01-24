#!/usr/bin/Rscript

# Bhishan Poudel
# Jan 8, 2016

# clear; Rscript rRefClass.r ; rm *~
# ref: http://programiz.com/r-programming/reference-class

#  Reference class in R programming language 
#  is similar to the object oriented programming 
#  we are used to seeing in common languages 
#  like C++, Java, Python etc. Unlike S3 and S4 classes, 
#  methods belong to class rather than generic functions. 
#  Reference class are internally implemented as 
#  S4 classes with an environment added to it.

#  Defining Reference Class
#  Defining reference class is similar 
#  to defining a S4 class. 
#  Instead of setClass() we use the setRefClass() function.

#########################################################################################################
#   Defining Reference Class
#########################################################################################################
library(methods)
setRefClass("student")

#  Member variables of a class, if defined, need to be 
#  included in the class definition. 
#  Member variables of reference class are called fields 
#  (analogous to slots in S4 classes). 
#  Following is an example to define a class called 
#  student with 3 fields, name, age and GPA.

setRefClass("student", fields=list(name="character", age="numeric", GPA="numeric"))

#########################################################################################################
#   Accessing elements
#########################################################################################################
print(student)


#########################################################################################################
#   CREATING CLASSES
#########################################################################################################
# creating a class all at once:
Person <- setRefClass("Person", 
                       methods = list( say_hello = function() message("Hi!") 
                      ))

# creating a class piece-by-piece
Person <- setRefClass("Person")
Person$methods(say_hello = function() message("Hi!"))




