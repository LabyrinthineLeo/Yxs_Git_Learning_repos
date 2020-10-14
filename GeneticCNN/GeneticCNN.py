# coding  : utf-8
# role    : GeneticCNN代码复现
# @Author : Labyrinthine Leo
# @Time   : 2020.10.10

"""
1、构建DAG数据结构：dag.py
"""
from copy import copy,deepcopy
from collections import deque
from collections import OrderedDict

# 构建DAG非法error
class DAGValidationError(Exception):
	pass

class DAG(object):
	"""
	role:有向无环图结构的具体实现
	"""
	def __init__(self):
		"""
		role:构建新的DAG结构
		"""
		self.reset_graph()

	def reset_graph(self):
		"""
		role:将graph重置为空状态
		note:graph的结构为有序字典，{'start_node':set(related_node)}
		"""
		self.graph = OrderedDict() # 设置为有序字典，即会按照元素输入顺序有序排列

	def add_node(self,node_name,graph=None):
		"""
		role:添加结点
		:param node_name:添加的结点名字
		:param graph:graph数据结构
		"""
		if not graph:
			graph = self.graph
		if node_name in graph: # 欲添加结点存在则报错
			raise KeyError('node {} already exists'.format(node_name))
		graph[node_name] = set() # 每一个key结点的value是无序不重复集合，用来存放与key相连的结点

	def add_node_if_not_exists(self,node_name,graph=None):
		"""
		role:如果欲添加结点报错则pass，否则正常添加结点
		"""
		try:
			self.add_node(node_name,graph=graph)
		except KeyError:
			pass

	def delete_node(self,node_name,graph=None):
		"""
		role:删除结点以及与其相连的边
		:param node_name:删除的结点名字
		:param graph:graph数据结构
		"""
		if not graph:
			graph = self.graph
		if node_name not in graph: # 欲删除结点不存在则报错
			raise KeyError('node {} does not exist'.format(node_name))
		graph.pop(node_name) # 将结点进行弹出操作

		for node,edges in graph.items(): # 同时要将相关的结点和边进行删除，items表示key和value列表
			if node_name in edges: 
				edges.remove(node_name) # 将其入度结点同时删除

	def delete_node_if_exists(self,node_name,graph=None):
		"""
		role:如果欲删除结点报错则pass，否则正常删除结点
		"""
		try:
			self.delete_node(node_name,graph=graph)
		except KeyError:
			pass

	def add_edge(self,ind_node,dep_node,graph=None):
		"""
		role:添加边
		:param ind_node:初始结点
		:param dep_node:结束结点(依赖结点)
		"""
		if not graph:
			graph = self.graph
		if ind_node not in graph or dep_node not in graph:
			raise KeyError('one or more nodes do not exist in graph')
		test_graph = deepcopy(graph)
		test_graph[ind_node].add(dep_node) # 将初始结点的结束点集合添加信息
		is_valid,message = self.validate(test_graph)
		if is_valid: # 判断添加以后是否合法
			graph[ind_node].add(dep_node)
		else:
			raise DAGValidationError()

	def delete_edge(self,ind_node,dep_node,graph=None):
		"""
		role:删除边
		:param ind_node:初始结点
		:param dep_node:结束结点(依赖结点)
		Example:
			d=collections.OrderedDict()
			d['a']=set()
			d['b']=set()
			d['c']=set()
			d['d']=set()
			d['a'].add('b')
			d['a'].add('c')
			d['b'].add('c')
			x = d.get('a',[])
			print(x)
		>>>{'c', 'b'}
		"""
		if not graph:
			graph = self.graph
		if dep_node not in graph.get(ind_node,[]): # 查看依赖结点是否存在
			raise KeyError('this edge does not exist in graph')
		graph[ind_node].remove(dep_node)

	def rename_edges(self,old_task_name,new_task_name,graph=None):
		"""
		role:更改对现有边中任务的引用
		:param old_task_name:
		:param new_task_name:
		:param graph:
		"""
		if not graph:
			graph = self.graph
		for node,edges in graph.items():
			if node == old_task_name:
				graph[new_task_name] = copy(edges) # 浅拷贝更改引用
				del graph[old_task_name]
			else:
				if old_task_name in edges:
					edges.remove(old_task_name)
					edges.add(new_task_name) # 更改为新的任务引用

	def predecessors(self,node,graph=None):
		pass

	def downstream(self,node,graph=None):
		pass

	def all_downstream(self,node,graph=None):
		pass

	def all_leaves(self,graph=None):
		pass

	def from_dict(self,graph_dict):
		pass

	def validate(self,graph=None):
		"""
		role:检测是否graph结构是否合法,即DAG
		:param graph:graph
		:return : tuple,(Boolean,message)
		"""
		graph = graph if graph is not None else self.graph # 判断是否为空
		if len(self.ind_nodes(graph)) == 0: # 独立结点个数为0
			return (False,'no independent nodes detected') # 显示未检测到独立结点
		try:
			self.topological_sort(graph)
		except ValueError:
			return (False,'failed topological sort') # 无法拓扑排序
		return (True, 'valide')

	def ind_nodes(self,graph=None):
		"""
		role:获取graph中无关联的结点列表
		:param graph:DAG
		Return:list,无关联结点列表
		Example:
			d=collections.OrderedDict()
			d['a']=set()
			d['b']=set()
			d['c']=set()
			d['d']=set()
			d['a'].add('b')
			d['a'].add('c')
			d['b'].add('c')
			print(d.values())
			x = set(j for i in d.values() for j in i)
			print(x)
		>>>odict_values([{'c', 'b'}, {'c'}, set(), set()])
		>>>{'c', 'b'}
		"""
		if graph is None:
			graph = self.graph
		# 返回graph中的所有有依赖关系的结点(无序不重复)
		dependent_nodes = set( # items()是graph的键值对元组的列表
				node for dependents in graph.values() for node in dependents
		)
		# 将不含依赖关系的结点作为列表返回
		return [node for node in graph.keys() if node not in dependent_nodes]

	def topological_sort(self,graph=None):
		"""
		role:对graph结构进行拓扑排序
		:param graph:graph结构
		"""
		if graph is None:
			graph = self.graph

		in_degree = {} # 入度
		for u in graph.keys():
			in_degree[u] = 0

		for u in graph.keys():
			for v in graph[u]:
				in_degree[v] += 1 # 入度+1

		# 之所以使用deque，为了保证拓扑排序过程
		queue = deque() # deque结构类似list，功能在于可以在结构两端进行操作元素
		for u in in_degree.keys():
			if in_degree[u] == 0:
				queue.appendleft(u) # 从左侧添加入度为0的结点元素

		# 整个循环就是在拓扑排序，剪枝操作
		li = []
		while queue:
			u = queue.pop() # 弹出列表尾部的入度为0的结点
			li.append(u) # 添加到li列表中
			for v in graph[u]: # 查看该结点的后继结点(出度)
				in_degree[v] -= 1 # 则其所有后继结点的入度-1
				if in_degree[v] == 0: # 若入度为0，则该结点加入queue中
					queue.appendleft(v)

		if len(li) == len(graph): # 不相等则证明是有环的
			return li # 拓扑序列
		else:
			raise ValueError('graph is not acyclic')


"""
2、构建DAG数据结构：dag.py
"""