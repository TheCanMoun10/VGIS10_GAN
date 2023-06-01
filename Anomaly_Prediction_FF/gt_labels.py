import numpy as np


l = []
l.append([399, 485])
l.append([455, 500])

listofzeros = [0] * 503

for i in range(2):
    fi = l[0][i]
    for j in range(l[0][i],l[1][i]):
        listofzeros[fi] = 1
        fi = fi + 1

print(listofzeros)
