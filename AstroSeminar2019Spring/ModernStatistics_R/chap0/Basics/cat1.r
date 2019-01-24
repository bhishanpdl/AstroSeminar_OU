#!/usr/bin/Rscript

# Bhishan Poudel
# Jan 5, 2016

# clear; Rscript cat1.r; rm *~


cat("This is a string\n")

cat ("A string won't omit quotes\n")

cat("This string has \"double quotes\"\n")

#number of characters
print (nchar ("Tab\t"))   # This string has 4, not 5, characters
cat ("Tab\thello\n")      # print it formatted to the screen


#Vectors of Strings
str <- letters[1:5]
print(str)
length(str)
nchar(str)

#convert string to numerics, note: alphabets shows error NA
str <- c("1", "2", "Yes", "No", "5")
as.numeric(str)

