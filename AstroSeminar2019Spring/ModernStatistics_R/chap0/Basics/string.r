#!/usr/bin/Rscript

# Bhishan Poudel
# Jan 5, 2016

# clear; Rscript string.r; rm *~

#Extracting Pieces of Strings
st <- dimnames(state.x77)[[1]] # State names, from built-in dataset
print(st[1:5])
 # "Alabama"    "Alaska"     "Arizona"    "Arkansas"   "California"
  
st <- st[1:5]               # Let's just use these five for now
print(substring (st, 1, 3)) # Give me the first three characters from each
 # "Ala" "Ala" "Ari" "Ark" "Cal"
 
print(substring (st, 1:5, 3:7)) # Give me 1:3 from the first, 2-4 from the second...
 # "Ala" "las" "izo" "ans" "for"

print(substring (st, nchar(st) - 2, nchar(st))) # Give me the last three
 # "ama" "ska" "ona" "sas" "nia"

# function to convert _ to .
#####################################################################3
strsplit ("Nospaces", "s")

convert.delimiter <- function (string, old="_", new = ".")
                     { # convert string delimited by "_" into strings delimited by "."
                        paste (unlist (strsplit (string, old)), collapse=new)
}

convert.delimiter ("a_thing_with_delimiters")
# "a.thing.with.delimiters"
