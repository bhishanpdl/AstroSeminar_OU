#!/usr/bin/Rscript

# Bhishan Poudel
# Jan 8, 2016

# clear; Rscript ifElse.r


# if else example1
x <- -5
if(x > 0){
  print("Non-negative number")
  } else {
    print("Negative number")
}

# single line example
x <- 5
if(x > 0) print("Non-negative number") else print("Negative number")

# nested if...else statement
x <- 0
if (x < 0) {
  print("Negative number")
} else if (x > 0) {
  print("Positive number")
} else
  print("Zero")


cat("\nexample 3\n")
# example 3
##=================
x <- 6
#x <- 4
if (x < 10){
  if(x < 5) {
    print("The number is less than 5")
  } else {
    print("The number is in between 5 and 10")
  }
}

###################################################