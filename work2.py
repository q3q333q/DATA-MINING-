import numpy as np
import matplotlib.pyplot as plt

def dealMatrix(matrix): #转置
    matrix = matrix.transpose()
    return matrix

def dealKmatrix(matrix): #核矩阵(linear kernel matrix)
    K = np.zeros(shape=(row, row))
    y = 0
    while y < row:
        x = 0
        while x < row:
            if x >= y:
                a = matrix[x]
                b = matrix[y]
                b.shape = (1, col)
                b = b.transpose()
                K[x][y] = a @ b 
                x = x + 1
            else:
                K[x][y] = K[y][x] 
                x = x + 1
        y = y + 1
    return K

def dealKmatrix2(matrix): #核矩阵(homogeneous quadratic kernel)
    K = np.zeros(shape=(row, row))
    y = 0
    while y < row:
        x = 0
        while x < row:
            if x >= y:
                a = matrix[x]
                b = matrix[y]
                b.shape = (1, col)
                b = b.transpose()
                K[x][y] = np.square(a @ b) #φ(x)Tφ(y)
                x = x + 1
            else:
                K[x][y] = K[y][x] 
                x = x + 1
        y = y + 1
    return K

def dealKmatrix3(Kmatrix):  #核矩阵(centered kernel matrix)
    a = np.eye(row) - (np.ones((row, row)) / row)
    K = a @ Kmatrix @ a 
    return K

def normalize(Kmatrix): # Normalizing
    W = np.diag(np.diag(Kmatrix))
    a = np.zeros(shape=(row, row))
    y = 0
    while y < row:
        x = 0
        while x < row:
            if x == y:
                a[x][y] = np.power(W[x][y], -1/2) #W -1/2
                x = x + 1
            else: 
                x = x + 1
        y = y + 1
    Kn = a @ Kmatrix @ a
    return Kn, W

def transform(matrix, W): #转换成φn(xi)
    point = np.zeros(shape=(row, col))
    n = 0
    while n < row:
        point[:][n] = 1 / np.sqrt(W[n][n]) * matrix[:][n] #φ(xi)/||φ(xi)||
        n = n + 1
    return point

def Verify(Kn1, Kn2): #对比
    y = 0
    while y < row:
        x = 0
        while x < row:
            if Kn1[x][y] == Kn2[x][y]:
                x = x + 1
            else: 
                print('不相同')
                return
        y = y + 1
    print('相同')

f = open('iris.txt','r')
first_ele = True
for data in f.readlines():
    data = data.strip(',"Iris-virginica"\n')
    data = data.strip(',"Iris-setosa"\n')
    data = data.strip(',"Iris-versicolor"\n')
    nums = data.split(",")
    if first_ele:
        nums = [float(x) for x in nums]
        matrix = np.array(nums)
        first_ele = False
    else:
        nums = [float(x) for x in nums]
        matrix = np.c_[matrix, nums]
matrix = np.delete(matrix, [2, 3], axis = 0)
matrix = dealMatrix(matrix)
row = matrix.shape[0]
col = matrix.shape[1]
lKmatrix = dealKmatrix(matrix)
hKmatrix = dealKmatrix2(matrix) #homogeneous
cKmatrix = dealKmatrix3(lKmatrix) #center
Kn1, W = normalize(lKmatrix) #normaliz
point = transform(matrix, W)
Kn2 = dealKmatrix(point)
Verify(Kn1, Kn2)
