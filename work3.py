import numpy as np
import matplotlib.pyplot as plt

def dealMatrix(matrix): #转置
    matrix = matrix.transpose()
    return matrix

def distance(x, y): #欧式距离
    return np.sqrt(np.power(x - y, 2).sum()) 

def DBSCAN(D, radius, minpts):
    C =[] #聚类
    core = set()
    for d in D:
        if len([ i for i in D if distance(d, i) <= radius]) >= minpts: 
                core.add(tuple(d))  #添加核心点
    k = 0 #聚类个数
    P = set() #未访问
    for ele in D:
        P.add(tuple(ele))
    while len(core):
        P_old = P
        o = list(core)[np.random.randint(0, len(core))] #随机找一个点
        P = P - set(o)
        Q = [] #访问
        Q.append(o)
        while len(Q):
            q = Q[0]
            Nq = [i for i in D if distance(q, i) <= radius] #距离小于radius
            a = set()
            for ele in Nq:
                a.add(tuple(ele))
            if len(Nq) >= minpts:
                S = P & a
                Q += (list(S))
                P = P - S
            Q.remove(q)
        k += 1
        Ck = list(P_old - P)
        core = core - set(Ck)
        C.append(Ck) #新的聚类
    return C, len(C)

def draw(C):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for i in range(len(C)):
        coo_X = []    #x坐标列表
        coo_Y = []    #y坐标列表
        for j in range(len(C[i])):
            coo_X.append(C[i][j][0])
            coo_Y.append(C[i][j][1])
        plt.scatter(coo_X, coo_Y, color=colValue[i%len(colValue)], label=i)

    plt.legend(loc='upper right')
    plt.show()


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
matrix = np.delete(matrix, 2, axis = 0)
D = dealMatrix(matrix)
C, nums = DBSCAN(D, 0.0001, 3)
draw(C)


def _step(x_l0, D, W=None, h=0.1):
    n = D.shape[0]
    d = D.shape[1]
    superweight = 0. 
    x_l1 = np.zeros((1,d))
    if W is None:
        W = np.ones((n,1))
    else:
        W = W
    for j in range(n):
        kernel = kernelize(x_l0, D[j], h, d)
        kernel = kernel * W[j]/(h**d)
        superweight = superweight + kernel
        x_l1 = x_l1 + (kernel * D[j])
    x_l1 = x_l1/superweight
    density = superweight/np.sum(W)
    return [x_l1, density]