#!/bin/bash

# https://www.reddit.com/r/archlinux/comments/sx33g4/whats_happened_to_librta/
touch empty.c  
gcc -fpic -c empty.c  
ar rcsv libdl.a empty.o
