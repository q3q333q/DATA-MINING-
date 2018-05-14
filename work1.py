import numpy as np
import matplotlib.pyplot as plt


def dealMatrix(matrix): #转置
    matrix = matrix.transpose()
    return matrix

def dealmeanvector(matrix): #均值向量
	matrix = np.mean(matrix, axis = 0)
	return matrix

def dealcovmatrix(matrix): #样本协方差矩阵(内积)
	meanvector = np.mean(matrix, axis = 0)
	X = matrix - meanvector #中心矩阵
	XT = X.transpose()
	row = matrix.shape[0]
	covMatrix = 1 / row * (XT @ X)
	return covMatrix

def dealcovmatrix2(matrix): #样本协方差矩阵（外积）
	meanvector = np.mean(matrix, axis = 0)
	X = matrix - meanvector
	row = X.shape[0]
	outer = np.zeros(X[0].shape[0])
	n = 0
	while n < row:
		z = X[n]
		a = z.shape[0]
		z.shape = (a, 1)
		zt = np.transpose(z)
		outer = outer + (z @ zt)
		n = n + 1
	covMatrix = 1 / row * outer
	return covMatrix

def dealCorrelation(matrix): #相关系数
	X1 = matrix[:, 0]
	Z1 = X1 - np.mean(X1, axis = 0)
	X2 = matrix[:, 1]
	Z2 = X2 - np.mean(X2, axis = 0)
	Z1t = Z1.transpose()
	Z2t = Z2.transpose()
	correlation = Z1t @ Z2 /(np.sqrt(Z1t @ Z1) * np.sqrt(Z2t @ Z2)) #cosθ
	return correlation

def scatterplot(matrix): #散点图
	plt.figure(figsize=(8,6))
	plt.scatter(matrix[:,0], matrix[:,1])
	plt.title('scatter plot')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend(loc='upper right')
	plt.show()

def plotPDF(matrix): #画出属性1概率密度函数
	X = matrix[:, 0]
	μ = np.mean(X, axis = 0)
	var = np.var(X)
	pi = np.pi
	def f(x):
		return 1/(np.sqrt(2 * pi) * np.sqrt(var)) * np.exp(- np.square(x - μ) / (2 * var)) #正态分布
	x = np.linspace(μ - 300, μ + 300, 1000)
	y = [f(i) for i in x]
	plt.figure(figsize=(8,6))
	plt.plot(x,y)
	plt.title('scatter plot')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend(loc='upper right')
	plt.show()

def compareVar(matrix): #最大最小方差
	col = matrix.shape[1]
	X = matrix[:, 0]
	maxvar = np.var(X)
	minvar = np.var(X)
	maxn = minn = 0
	n = 1
	while n < col:
		X = matrix[:, n]
		var = np.var(X)
		if var > maxvar:
			maxvar = var
			maxn = n
		if var < minvar:
			minvar = var
			minn = n
		n = n + 1
	print('方差最大的是属性%d,方差是%f' %(maxn, maxvar))
	print('方差最小的是属性%d,方差是%f' %(minn, minvar))

#def compareCov(matrix): #最大最小协方差
			


f = open('magic04.txt','r')
first_ele = True
for data in f.readlines():
	data = data.strip(',g\n')
	data = data.strip(',h\n')# 去掉每行末尾的字母
	nums = data.split(",") # 按照逗号进行分割。
	if first_ele:
		nums = [float(x) for x in nums]
		matrix = np.array(nums)
		first_ele =False
	else:
		nums = [float(x) for x in nums]
		matrix = np.c_[matrix,nums]
matrix = dealMatrix(matrix)
#meanvector = dealmeanvector(matrix)
covMatrix = dealcovmatrix2(matrix)
#correlation = dealCorrelation(matrix)
#scatterplot(matrix)
#plotPDF(matrix)
compareVar(matrix)
#compareCov(matrix)
#print(correlation)
f.close





