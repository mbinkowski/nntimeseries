# -*- coding: utf-8 -*-
"""
Configuration file. User may set the working directory variable WDIR below to 
the directory of his choice.
"""

import sys, os

#print('sys.argv[0] =', sys.argv[0])             
#pathname = os.path.dirname(sys.argv[0])        
#print('path =', pathname)
#print('full path =', os.path.abspath(pathname)) 

print(__file__)
print(os.path.abspath(__file__))
SEP = os.path.sep
WDIR = SEP.join(os.path.abspath(__file__).split(SEP)[:-2] + [''])
print(WDIR)
#WDIR = os.path.abspath(os.getcwd()) + '\\'
#WDIR = '\\'.join(path.split('\\')[:-1])
#cwd = os.getcwd()
#if __name__ == '__main__':
#    WDIR = cwd[:-4]
#    with open(cwd + '\\models\\config.py', 'w') as f:
#        f.write("WDIR = " + repr(WDIR))
##    with open(cwd + '\\config.txt', 'w') as f:
##        f.write(WDIR)
#else:
#    print(cwd)
#    WDIR = cwd[:-4]
#    assert 'config.txt' in os.listdir(cwd), "config.txt not found. Run config.sh from the package main directory."
#    with open('config.txt', 'r') as f:
#        WDIR = f.read()

import sys 
sys.setrecursionlimit(2000) # to alleviate some problems with model saving.