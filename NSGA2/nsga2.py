# coding  : utf-8
# nsga2的算法复现
# @Author : Labyrinthine Leo
# @Time   : 2020.06.03

import os
import numpy as np
import random
import math
import matplotlib.pyplot as plt

## 测试函数1
def KUR(x):
	"""
	CostFunction
	input:决策变量
	output:决策变量对应的测试函数值
	"""
	a = 0.8
	b = 3
	z1 = 0
	z2 = 0
	z = []
	for i in range(len(x)-1):
		z1 += -10*math.exp(-0.2*math.sqrt(x[i]**2+x[i+1]**2))
	z.append(round(z1,4))
	for i in range(len(x)):
		z2 += math.fabs(x[i])**a+5*(math.sin(x[i]))**b
	z.append(round(z2,4))

	return z

## 测试函数2
def ZDT(x):
	pass

## 支配关系判断函数
def Dominates(x,y):
	"""
	Dominates Function
	input:x和y两个个体
	output：True or False
	"""
	# 个体的测试函数值非空则进行提取比较
	if type(x) is dict:
		x = x["Cost"]
	if type(y) is dict:
		y = y["Cost"]

	# 初始化比较参数
	less_num = 0
	less_equal_num = 0

	# 比较值的大小来判断解的支配性
	for i in range(len(x)):
		if(x[i]<=y[i]):
			less_equal_num += 1
		if(x[i]<y[i]):
			less_num += 1

	# 判断支配条件
	if less_equal_num == len(x) and less_num>0:
		return True
	else:
		return False

## 快速非支配排序函数
def NonDominatedSorting(pop):
	"""
	NonDominatedSorting Function
	input：种群
	output：新种群和pareto front集合
	"""
	n_Pop = len(pop)     # 获取种群的个体数量

	# 初始化个体的支配解集以及置0支配解数量
	for i in range(n_Pop):
		pop[i]["DominationSet"] = []
		pop[i]["DominationCount"] = 0

	# 初始化不同层的pareto最优解集,字典类型
	F = {}
	F[1] = []            # 第1层pareto解

	# 求解每一个体的被支配解个数Np和支配解集合Sp
	for i in range(n_Pop):
		for j in range(i+1,n_Pop):
			p = pop[i]
			q = pop[j]

			# 表示p支配q
			if Dominates(p,q):
				p["DominationSet"].append(j)  # 存放下标而非个体，减少数据量
				q["DominationCount"] += 1

			# 表示q支配p
			if Dominates(q["Cost"],p["Cost"]):
				q["DominationSet"].append(i)
				p["DominationCount"] += 1

			# 刷新改变后的个体信息
			pop[i] = p
			pop[j] = q

		# 查找非支配解
		if pop[i]["DominationCount"] == 0:
			F[1].append(i)       # 将非支配解下标添加进第1层pareto前沿面
			pop[i]["Rank"] = 1   # pareto最优解等级

	k = 1

	# 将pareto解分层
	while True:
		# 初始化第k+1层pareto前沿面
		Q = []

		# 遍历第k层pareto front中的解
		for i in F[k]:
			p = pop[i]
			# 遍历非支配解支配的解
			for j in p["DominationSet"]:
				q = pop[j]

				# 将此非支配解在种群中忽略
				q["DominationCount"] -= 1

				# 判断新一轮此层的非支配解
				if q["DominationCount"] == 0:
					Q.append(j)
					q["Rank"] = k+1

				# 刷新个体元素的信息
				pop[j] = q

		# 循环结束条件：无pareto最优解
		if len(Q) == 0:
			break

		# 保存此层pareto front
		F[k+1] = Q
		k += 1

	return pop,F

## 计算拥挤距离函数
def CalcCrowdingDistance(pop,F):
	"""
	CalcCrowdingDistance Function
	input:种群和pareto front 集合
	output：更新后的pop
	"""
	# 获取 pareto front的层数
	n_F = len(F)

	# 遍历每一层pareto front
	for k in range(1,n_F+1):
		Costs = {}       # 初始化每一层pareto front的目标函数值
		for indiId in F[k]:
			Costs[indiId] = pop[indiId]["Cost"]

		# 获取目标个数
		n_Object = len(pop[indiId]["Cost"])
		# 获取每一层个体的个数
		n_Indi= len(F[k])

		# 初始化二维0矩阵,表示个体的拥挤距离
		dist = {}
		# dist = [[0. for x in range(n_Indi)] for y in range(n_Object)]

		# 根据目标值对非支配解排序
		for j in range(n_Object):
			# 按照第j维的目标函数值进行排序
			OrdeSeq = sorted(Costs.items(),key=lambda item:item[1][j])

			# 设置极端点拥挤距离为无穷大
			dist[OrdeSeq[0][0]] = float('inf')
			dist[OrdeSeq[n_Indi-1][0]] = float('inf')
			# 获取j维上目标函数的极值
			val_min = OrdeSeq[0][1][j]
			val_max = OrdeSeq[n_Indi-1][1][j]
			# 计算其他点的拥挤距离
			for i in range(1,n_Indi-1):
				# 后一个个体在j维目标上的目标值
				cost_pre = OrdeSeq[i-1][1][j]
				cost_next = OrdeSeq[i+1][1][j]
				# 判断该点是否初始化
				if OrdeSeq[i][0] not in dist.keys(): 
					dist[OrdeSeq[i][0]] = 0.
				elif val_max != val_min:
					dist[OrdeSeq[i][0]] += round((cost_next-cost_pre)/(val_max-val_min),4)

		# 每一层计算完毕，更新distance
		for indi_dist in dist:
			pop[indi_dist]["CrowdingDistance"] = dist[indi_dist]

	# 更新种群
	return pop

## 使用拥挤比较算子对种群排序
def SortPopulation(pop):
	"""
	SortPopulation Function
	input:非支配排序且得到拥挤距离的种群
	output:
	"""
	## 先进行拥挤距离排序、再进行Rank排序，技巧性强
	n_Pop = len(pop)
	cd_dict = {}         # 初始化拥挤距离dict
	rank_dict = {}       # 初始化层级dict
	cd_pop = []          # 按cd排序后的种群
	rank_pop = []        # 按rank排序后的种群

	# 基于拥挤距离排序
	for i in range(n_Pop):
		cd_dict[i] = pop[i]["CrowdingDistance"]
	# 暂存dict
	temp_dict = sorted(cd_dict.items(),key=lambda item:item[1],reverse=True)
	# 对种群的cd排序，区分偏序度/适应度
	for i in range(n_Pop):
		cd_pop.append(pop[temp_dict[i][0]])

	# 基于Front Rank排序
	for i in range(n_Pop):
		rank_dict[i] = cd_pop[i]["Rank"]
	# 对rank排序
	temp_dict = sorted(rank_dict.items(),key=lambda item:item[1])
	# 对种群在cd基础上对rank排序
	for i in range(n_Pop):
		rank_pop.append(cd_pop[temp_dict[i][0]])

	# 更新pop
	pop = rank_pop

	# 设置Fronts
	F = {}
	# 获取front层数
	MaxRank = 0
	for i in range(n_Pop):
		if pop[i]["Rank"] > MaxRank:
			MaxRank = pop[i]["Rank"]
	# 初始化F
	for i in range(1,MaxRank+1):
		F[i] = []
	# 更新F
	for i in range(n_Pop):
		F[pop[i]["Rank"]].append(pop[i])

	return pop,F

## 个体交叉(基因重组)函数
def Crossover(x1,x2):
	"""
	对个体的决策变量进行交叉重组
	input: 两个个体的Position
	output: 两个子个体
	"""
	n_dv = len(x1)
	# 生成与个体的决策变量个数相同的参数
	alpha = [random.random() for i in range(n_dv)]

	# SBX交叉公式
	y1 = []
	y2 = []
	for i in range(n_dv):
		y1.append(round(alpha[i]*x1[i]+(1-alpha[i])*x2[i],4))
		y2.append(round(alpha[i]*x2[i]+(1-alpha[i])*x1[i],4))

	return y1,y2

## 个体变异函数
def Mutate(x,mu,sigma):
	"""
	对个体的决策变量进行变异
	input:
	output:
	"""
	# 获取决策变量个数
	n_Var = len(x)
	# 获取变异的决策变量个数
	n_Mu = math.ceil(mu*n_Var)

	# 随机生成发生变异的决策变量
	j = [random.randint(0,n_Var-1) for i in range(n_Mu)]

	# 变异决策变量值
	y = x
	for i in range(len(j)):
		y[j[i]] = round(x[j[i]] + sigma*random.uniform(-0.1,0.1),4)

	return y


## 绘制种群图像函数
def PlotCosts(pop):
	"""
	绘制pareto前沿面
	input:每一代第1层的pareto最优解集
	"""
	# 获取所有个体的目标函数值
	Costs = []
	for i in range(len(pop)):
		Costs.append(pop[i]["Cost"])

	# 获取第1目标值和第2目标值的列表
	Cost_x = []
	Cost_y = []
	for i in range(len(Costs)):
		Cost_x.append(Costs[i][0])
		Cost_y.append(Costs[i][1])

	# 绘制画板
	plt.figure(figsize=(12,6),dpi=80)
	# 绘制散点
	plt.scatter(Cost_x,Cost_y,marker='*',s=150,c='red')

	# 信息描述
	plt.title(r'Non-dominated Solution($\ F_1$)',fontsize=28)
	plt.xlabel(r'$\ 1^{st}$ Objective',fontsize=15)
	plt.ylabel(r'$\ 2^{nd}$ Objective',fontsize=15)
	plt.grid()



## 主函数
def main():
	"""
	主函数：实现完整算法功能
	"""

	## 信息定义
	CostFunction = KUR   # 设置测试函数
	n_Var = 3            # 决策变量个数
	VarSize = 3          # 决策变量矩阵的大小,1*3
	VarMin = -5          # 变量最小值
	VarMax = 5           # 变量最大值
	n_Object = len(CostFunction([random.uniform(-5,5) for i in range(n_Var)]))

	## NSGA-II的参数
	MaxIt = 100          # 最大迭代次数
	n_Pop = 100          # 种群大小
	p_Crossover = 0.7    # 交叉比率
	n_Crossover = 2*round(p_Crossover*n_Pop/2) # 欲交叉的父代种群个数
	p_Mutation = 0.4     # 变异比率
	n_Mutation = round(p_Mutation*n_Pop)       # 产生变异的个体数量
	mu = 0.02            # 变异速率
	sigma = 0.1*(VarMax-VarMin)                # 变异步长

	## 初始化种群个体数据
	# empty_individual = {}
	# empty_individual["Position"] = []          # 个体位置
	# empty_individual["Cost"] = []              # 个体测试函数值
	# empty_individual["Rank"] = 0               # pareto最优解等级
	# empty_individual["DominationSet"] = []     # 当前解的支配解集合
	# empty_individual["DominationCount"] = []   # 支配当前解的解数量
	# empty_individual["CrowdingDistance"] = 0.  # 拥挤距离初始为0
	# pop = [empty_individual for i in range(n_Pop)]  # 初始化种群(个体均为空)
	
	# 种群随机初始化
	pop = []
	for i in range(n_Pop):
		# 添加个体(字典结构)
		pop.append({})

		# 初始化个体的决策变量值
		pop[i]["Position"] = [round(random.uniform(-5,5),4) for i in range(VarSize)]

		# 获取初始化的决策变量对应的测试函数值
		pop[i]["Cost"] = CostFunction(pop[i]["Position"])
		pop[i]["Rank"] = 0               # pareto最优解等级
		pop[i]["DominationSet"] = []     # 当前解的支配解集合
		pop[i]["DominationCount"] = 0    # 支配当前解的解数量
		pop[i]["CrowdingDistance"] = 0.  # 拥挤距离初始为0

	## 种群进行非支配排序
	# print("-----------------first random pop------------------")
	# print(pop)
	pop, F = NonDominatedSorting(pop)
	# print("-----------------first NDS pop------------------")
	# print(pop)
	# print("-----------------first NDS F------------------")
	# print(F)

	## 计算个体的拥挤距离
	pop = CalcCrowdingDistance(pop,F)
	# print("-----------------having CD pop------------------")
	# print(pop)

	## 排序种群，获取个体的偏序度
	pop, F = SortPopulation(pop)
	# print("-----------------having SP pop------------------")
	# print(pop)

	## Main Loop
	# 循环迭代
	# 绘制画板
	plt.figure(figsize=(12,6),dpi=80)
	plt.ion()
	for it in range(MaxIt):
		# 交叉算子
		# 初始化第二代种群(个体均为空)
		# popc = [[empty_individual for i in range(2)] for j in range(n_Crossover/2)]
		popc = []
		for i in range(int(n_Crossover/2)):    # 注意类型转换
			x = []
			for j in range(2):
				x.append({})
				x[j]["Position"] = []          # 个体位置
				x[j]["Cost"] = []              # 个体测试函数值
				x[j]["Rank"] = 0               # pareto最优解等级
				x[j]["DominationSet"] = []     # 当前解的支配解集合
				x[j]["DominationCount"] = 0    # 支配当前解的解数量
				x[j]["CrowdingDistance"] = 0.  # 拥挤距离初始为0
			popc.append(x)

		for k in range(int(n_Crossover/2)):

			# 二元锦标赛选择
			# 随机获取个体1
			i1 = random.randint(0,n_Pop-1)
			p1 = pop[i1]
			# 随机获取个体2
			i2 = random.randint(0,n_Pop-1)    # 是否允许自交？
			p2 = pop[i2]

			# 个体进行交叉
			popc[k][0]["Position"],popc[k][1]["Position"] = Crossover(p1["Position"],p2["Position"])

			# 更新后代的目标函数值
			popc[k][0]["Cost"] = CostFunction(popc[k][0]["Position"]) 
			popc[k][1]["Cost"] = CostFunction(popc[k][1]["Position"])

		# 将popc进行列化
		popd = []
		for i in range(len(popc)):
			popd.append(popc[i][0])
		for i in range(len(popc)):
			popd.append(popc[i][1])

		# 变异算子
		# popm = [empty_individual for i in range(n_Mutation)]
		popm = []
		for i in range(int(n_Mutation)):
			popm.append({})
			popm[i]["Position"] = []          # 个体位置
			popm[i]["Cost"] = []              # 个体测试函数值
			popm[i]["Rank"] = 0               # pareto最优解等级
			popm[i]["DominationSet"] = []     # 当前解的支配解集合
			popm[i]["DominationCount"] = 0    # 支配当前解的解数量
			popm[i]["CrowdingDistance"] = 0.  # 拥挤距离初始为0

		for k in range(int(n_Mutation)):
			
			# 随机获取个体
			i = random.randint(0,n_Pop-1)
			p = pop[i]

			# 个体进行变异
			popm[k]["Position"] = Mutate(p["Position"],mu,sigma)
			popm[k]["Cost"] = CostFunction(pop[k]["Position"])

		# 父代与子代结合
		for i in range(len(popd)):
			pop.append(popd[i])
		for i in range(len(popm)):
			pop.append(popm[i])

		# 新种群进行非支配排序
		pop, F = NonDominatedSorting(pop)

		# 计算个体的拥挤距离
		pop = CalcCrowdingDistance(pop,F)

		# 排序新种群，获取个体的适应度
		pop_temp, F = SortPopulation(pop)

		# 精英策略,选择新父代
		pop = []       # 一定要置空再存储
		for i in range(n_Pop):
			pop.append(pop_temp[i])

		# -----重复一次------
		# 新种群进行非支配排序
		pop, F = NonDominatedSorting(pop)

		# 计算个体的拥挤距离
		pop = CalcCrowdingDistance(pop,F)

		# 排序新种群，获取个体的适应度
		pop, F = SortPopulation(pop)

		# 存储第1层Front
		F1 = F[1]

		# 打印信息
		print("Iteration {0}: Number of F1 Members = {1}".format(it,len(F1)))

		# 绘图
		plt.cla()  # 清除信息
		# 获取所有个体的目标函数值
		Costs = []
		for i in range(len(pop)):
			Costs.append(pop[i]["Cost"])

		# 获取第1目标值和第2目标值的列表
		Cost_x = []
		Cost_y = []
		for i in range(len(Costs)):
			Cost_x.append(Costs[i][0])
			Cost_y.append(Costs[i][1])

		# 绘制散点
		plt.scatter(Cost_x,Cost_y,marker='o',s=75,c='blue',lw=0.5)

		# 信息描述
		plt.title(r'Non-dominated Solution($\ F_1$)',fontsize=18)
		plt.xlabel(r'$\ 1^{st}$ Objective',fontsize=15)
		plt.ylabel(r'$\ 2^{nd}$ Objective',fontsize=15)
		plt.grid()
		plt.pause(0.4)

	plt.ioff()
	plt.show()
		

if __name__ == '__main__':
	main()