# coding  : utf-8
# role    : 实现EvoCNN算法
# @Author : Labyrinthine Leo
# @Time   : 2020.09.04

"""
1、构建3种含默认参数的layers(即paper中的unit)：layers.py
"""
import numpy as np
import random

class ConvLayer:
    """
    实现卷积层类，strid默认为1，因此无需设置参数
    """
    def __init__(self, filter_size=[2, 2], feature_map_size=8, weight_matrix=[0.0,1.0]):
        self.filter_width = filter_size[0] # 卷积核大小
        self.filter_height = filter_size[1]
        self.feature_map_size = feature_map_size # 结果特征图的数量
        """
        每个结构个体的卷积和池化层的权重参数是随机初始化的，这样在不断进化的过程中，
        好的个体中好的权重也会相应保留下来，这就是论文中的所谓新颖的权重初始化方式(important)
        """
        self.weight_matrix_mean = weight_matrix[0] # 权重矩阵平均值,高斯分布
        self.weight_matrix_std = weight_matrix[1] # 权重矩阵标准差
        self.type = 1 # Unit type为1，表示卷积层
    
    def __str__(self):
        # 打印卷积层信息
        return "Conv Layer: filter:[{0},{1}], feature map number:{2}, weight:[{3},{4}]".format(self.filter_width, self.filter_height, self.feature_map_size, self.weight_matrix_mean, self.weight_matrix_std)
    
class PoolLayer:
    """
    实现池化层类，strid默认为1，因此无需设置参数
    """
    def __init__(self, kernel_size=[2,2], pool_type=0.1):
        self.kernel_width = kernel_size[0] # 采样窗口的大小
        self.kernel_height = kernel_size[1]
        self.kernel_type = pool_type # 池化类型，小于0.5表示最大值池化，否则为平均值池化
        self.type = 2
        
    def __str__(self):
        return "Pool Layer: kernel:[{0},{1}], type:{2}".format(self.kernel_width,self.kernel_height, "max" if self.kernel_type<0.5 else "mean")

class FullLayer:
    """
    实现全连接层类
    """
    def __init__(self, hidden_neurons_num=10,weight_matrix=[0.0,1.0]):
        self.hidden_neurons_num = hidden_neurons_num # 隐藏层神经元个数
        self.weight_matrix_mean = weight_matrix[0] # 权重矩阵平均值,高斯分布
        self.weight_matrix_std = weight_matrix[1] # 权重矩阵标准差
        self.type = 3
        
    def __str__(self):
        return "Full Layer: hidden neuron:{0}, weight:[{1},{2}]".format(self.hidden_neurons_num, self.weight_matrix_mean, self.weight_matrix_std)


"""
2、构建individual类(即paper中的一个个体，一个随机的整体性网络结构)：individual.py
"""
import numpy as np
import random

class Individual:
    """
    实现个体元素类
    """
    def __init__(self, x_prob=0.9, x_eta=0.05, m_prob=0.2, m_eta=0.05):
        self.indi = []
        self.x_prob = x_prob #模拟二进制交叉(SBX)的概率
        self.x_eta = x_eta # SBX系数
        self.m_prob = m_prob #变异的概率
        self.m_eta = m_eta # 变异系数
        # Note:注意这里的mean和std会在每一次进行评估后进行更新
        self.mean = 0 # 权重初始化的平均值(其实也是个体做完评估后的准确率list的平均值)
        self.std = 0 # 权重初始化的标准差(其实也是个体做完评估后的准确率list的标准差)
        self.complexity = 0 # 复杂度即个体的权重数量，每一次评估完进行更新
        
        ##
        self.feature_map_size_range = [3, 50] # 特征图数量范围
        self.filter_size_range = [2, 20] # 卷积核大小范围
        self.pool_kernel_size_range = [1, 2] # 采样窗口的大小范围
        self.hidden_neurons_range = [1000, 2000] # 隐藏层神经元个数范围
        self.mean_range = [-1, 1] # 平均值范围
        self.std_range = [0, 1] # 标准差范围
        
    def clear_state_info(self):
        # 清除状态信息
        self.complexity = 0
        self.mean = 0
        self.std = 0
        
    # initialize a simple CNN network including one convolutional layer,one pooling layer, and one full connection layer
    def initialize(self):
        self.indi = self.init_one_individual()
    
    def init_one_individual(self): # 初始化个体，每层的数量[1,6)
        init_num_conv = np.random.randint(1,6) 
        init_num_pool = np.random.randint(1,6)
        init_num_full = np.random.randint(1,6)
        _list = []
        for _ in range(init_num_conv+init_num_full):
            if np.random.random() < 0.5:
                _list.append(self.add_a_random_conv_layer())
            else:
                _list.append(self.add_a_random_pool_layer())
        for _ in range(init_num_full-1):
            _list.append(self.add_a_random_full_layer())
        _list.append(self.add_a_common_full_layer())
        return _list
    
    def get_layer_at(self,i): # 获取个体的第i层结构
        return self.indi[i]
    
    def get_layer_size(self): # 获取个体的Unit层数
        return len(self.indi)
    
    def init_mean(self): # 随机产生范围内的均值(random()函数生成(0,1)之间的随机小数)
        return np.random.random()*(self.mean_range[1] - self.mean_range[0]) + self.mean_range[0]
        
    def init_std(self): # 随机产生范围内的标准差
        return np.random.random()*(self.std_range[1] - self.std_range[0]) + self.std_range[0]
    
    def init_filter_size(self): # 随机产生范围内的卷积核尺寸
        return np.random.randint(self.filter_size_range[0], self.filter_size_range[1])
    
    def init_feature_map_size(self): # 随机产生范围内的特征数量
        return np.random.randint(self.feature_map_size_range[0], self.filter_size_range[1])
    
    def init_pool_kernel_size(self): # 随机产生池化窗口尺寸
        # pool_kernel_size_num = len(self.pool_kernel_size_range)
        # n = np.random.randint(pool_kernel_size_num)
        # return np.power(2, self.pool_kernel_size_range[n])
        return self.pool_kernel_size_range[np.random.randint(2)]
    
    def init_hidden_neurons_size(self): # 随机产生范围内隐藏层神经元个数
        return np.random.randint(self.hidden_neurons_range[0], self.hidden_neurons_range[1])
    
    def add_a_common_full_layer(self): # 添加通用的全连接层
        """
        通用层即最后一层的全连接层，神经元个数为2，接在整个结构最后一层
        """
        mean = self.init_mean()
        std = self.init_std()
        full_layer = FullLayer(hidden_neurons_num=2,weight_matrix=[mean,std])
        return full_layer

    def add_a_random_full_layer(self): # 添加随机的全连接层
        mean = self.init_mean()
        std = self.init_std()
        hidden_neurons_num = self.init_hidden_neurons_size()
        full_layer = FullLayer(hidden_neurons_num=hidden_neurons_num,weight_matrix=[mean,std])
        return full_layer
    
    def add_a_random_conv_layer(self): # 添加随机卷积层
        s1 = self.init_filter_size()
        filter_size = s1,s1 # 初始化卷积核尺寸(元组)
        feature_map_size = self.init_feature_map_size()
        mean = self.init_mean()
        std = self.init_std()
        conv_layer = ConvLayer(filter_size=filter_size,feature_map_size=feature_map_size,weight_matrix=[mean,std])
        return conv_layer
    
    def add_a_random_pool_layer(self): # 添加随机池化层
        s1 = self.init_pool_kernel_size()
        kernel_size = s1,s1 # 随机初始化采样窗口尺寸
        pool_type = np.random.random(size=1) # 随机产生一个池化类型(0.5以下最大池化，否则平均池化)
        pool_layer = PoolLayer(kernel_size=kernel_size,pool_type=pool_type[0]) # 这里其实可以直接random就好了？
        return pool_layer
    
    def generate_a_new_layer(self, current_unit_type, unit_length):
        """
        role：按一定要求产生新的layer
        """
        if current_unit_type == 3: # 全连接层
            # 当个体只有一层且为全连接层时，添加卷积或池化
            if unit_length == 1:
                if random.random()<0.5:
                    return self.add_a_random_conv_layer()
                else:
                    return self.add_a_random_pool_layer()
            else:
                return self.add_a_random_full_layer()
        else:
            if random.random()<0.5:
                return self.add_a_random_conv_layer()
            else:
                return self.add_a_random_pool_layer()

    def mutation(self): # 个体变异，即层次整体结构变异
        """
        role:个体结构变异，用于交叉后的个体(多项式变异)
        """
        if flip(self.m_prob): # 变异概率
            # for the units
            unit_list = []
            for i in range(self.get_layer_size()-1): # 多留一层
                cur_unit = self.get_layer_at(i) # 获取当前层结构
                if flip(0.5):
                    # mutation，每个unit变异概率0.5
                    p_op = self.mutation_ope(np.random.random()) # 随机变异:0添加、1修改、2删除
                    max_length = 6
                    current_length = (len(unit_list) + self.get_layer_size()-i-1)
                    if p_op == 0: # add a new
                        if current_length < max_length: # 小于最大长度，只变异不添加(在当前层之前进行添加)
                            unit_list.append(self.generate_a_new_layer(cur_unit.type,self.get_layer_size()))
                            unit_list.append(cur_unit)
                        else: # 超过最大层数则只变异
                            updated_unit = self.mutation_a_unit(cur_unit,self.m_eta)
                            unit_list.append(updated_unit)
                    if p_op == 1: # modify
                        updated_unit = self.mutation_a_unit(cur_unit,self.m_eta)
                        unit_list.append(updated_unit)
                else: # 否则还是原层
                    unit_list.append(cur_unit)
                    
                # 避免特殊概率下list中无unit，添加卷积池化和全连接层
                if len(unit_list) == 0:
                    unit_list.append(self.add_a_random_conv_layer())
                    unit_list.append(self.add_a_random_pool_layer())
                unit_list.append(self.get_layer_at(-1))
                # 判断前两个unit层的类型
                if unit_list[0].type != 1: # 如果第一层非卷积层则添加卷积层
                    unit_list.insert(0,self.add_a_random_conv_layer())
                self.indi = unit_list # 返回变异后的unit结构
    
    def mutation_a_unit(self, unit, eta): # 单层变异，即某一unit结构变异
        if unit.type == 1: # 卷积层
            return self.mutate_conv_unit(unit, eta)
        elif unit.type == 2: # 池化层
            return self.mutate_pool_unit(unit, eta)
        else: # 全连接层
            return self.mutate_full_layer(unit, eta)
        
    def mutate_conv_unit(self, unit, eta): # 卷积层变异
        # feature map size, feature map number, mean std
        fms = unit.filter_width
        fmn = unit.feature_map_size
        mean = unit.weight_matrix_mean
        std = unit.weight_matrix_std
        
        new_fms = int(self.pm(self.filter_size_range[0],self.filter_size_range[-1],fms,eta))
        new_fmn = int(self.pm(self.feature_map_size_range[0],self.feature_map_size_range[1],fmn,eta))
        new_mean = self.pm(self.mean_range[0],self.mean_range[1],mean,eta)
        new_std = self.pm(self.std_range[0],self.std_range[1],std,eta)
        conv_layer = ConvLayer(filter_size=[new_fms,new_fms],feature_map_size=new_fmn,weight_matrix=[new_mean,new_std])
        
        return conv_layer
    
    def mutate_pool_unit(self, unit, eta): # 池化层变异
        #kernel size, pool_type
        ksize = np.log2(unit.kernel_width)
        pool_type = unit.kernel_type
        
        new_ksize = self.pm(self.pool_kernel_size_range[0],self.pool_kernel_size_range[-1],ksize,eta)
        new_ksize = int(np.power(2,new_ksize))
        new_pool_type = self.pm(0,1,pool_type,eta)
        pool_layer = PoolLayer(kernel_size=[new_ksize,new_ksize],pool_type=new_pool_type)
        
        return pool_layer
    
    def mutate_full_layer(self, unit, eta): # 全连接层变异
        #num of hidden neurons, mean, std
        n_hidden = unit.hidden_neurons_num
        mean = unit.weight_matrix_mean
        std = unit.weight_matrix_std
        
        new_n_hidden = int(self.pm(self.hidden_neurons_range[0],self.hidden_neurons_range[-1],n_hidden,eta))
        new_mean = self.pm(self.mean_range[0],self.mean_range[1],mean,eta)
        new_std = self.pm(self.std_range[0],self.std_range[1],std,eta)
        full_layer = FullLayer(hidden_neurons_num=new_n_hidden,weight_matrix=[new_mean,new_std])
        
        return full_layer

    # 变异操作：0表示添加、1表示修改、2表示删除
    def mutation_ope(self, r):
        if r<0.33:
            return 1
        elif r>0.66:
            return 2
        else:
            return 0

    def pm(self, xl, xu, x, eta):
        """
        role : 多项式变异算子
        :param xl : 取值下限
        :param xu : 取值上限
        :param x  : 欲变异参数的变量值
        :param eat: η
        """
        delta_1 = (x-xl)/(xu-xl) # Δ/δ
        delta_2 = (xu-x)/(xu-xl)
        rand = np.random.random()
        mut_pow = 1.0/(eta+1.) # 变异指数
        if rand<0.5:
            xy = 1.0-delta_1
            val = 2.0*rand+(1.0-2.0*rand)*xy**(eta+1)
            delta_q = val**mut_pow-1.0
        else:
            xy = 1.0-delta_2
            val = 2.0*(1.0-rand)+2.0(rand-0.5)*xy**(eta+1)
            delta_q = 1.0-val**mut_pow
        x = x+delta_q*(xu-xl)
        x = min(max(x,xl),xu)
        return x
    
    def __str__(self):
        str_ = []
        str_.append('Length:{}, Num:{}'.format(self.get_layer_size(),self.complexity))
        str_.append('Mean:{:.2f}'.format(self.mean))
        str_.append('Std:{:.2f}'.format(self.std))
        
        for i in range(self.get_layer_size()):
            unit = self.get_layer_at(i)
            if unit.type == 1:
                str_.append("conv[{},{},{},{:.2f},{:.2f}]".format(unit.filter_width,unit.filter_height,unit.feature_map_size,unit.weight_matrix_mean,unit.weight_matrix_std))
            elif unit.type == 2:
                str_.append("pool[{},{},{:.2f}]".format(unit.kernel_width,unit.kernel_height,unit.kernel_type))
            elif unit.type == 3:
                str_.append("full[{},{},{}]".format(unit.hidden_neurons_num,unit.weight_matrix_mean,unit.weight_matrix_std))
            else:
                raise Exception("Incorrect unit flag")
        
        return ','.join(str_) # 用','将各元素信息连接为字符串

# test code:
# print("----------Individual----------")
# ind = Individual()
# ind.initialize()
# s = []
# 如果将类直接存储则展示类的对象信息，而进行str()函数操作则会将其__str__魔法函数中的内容进行存储
# s.append(str(ind))
# print(s)
# print(ind)
# print("------------------------------")

"""
3、构建population类，即初始化种群：population.py
"""

class Population:  # 种群类
    def __init__(self, num_pops): # 种群初始化
        self.num_pops = num_pops
        self.pops = []
        for i in range(num_pops):
            indi = Individual()
            indi.initialize()
            self.pops.append(indi)

    def get_individual_at(self, i): # 获取第i个个体
        return self.pops[i]

    def get_pop_size(self): # 获取当前种群个体数量
        return len(self.pops)

    def set_populations(self, new_pops): # 种群更新
        self.pops = new_pops

    def __str__(self): # 打印种群个体信息
        _str = []
        for i in range(self.get_pop_size()):
            _str.append(str(self.get_individual_at(i)))
        return '\n'.join(_str)

# test code:
# print("----------Population----------")
# pop = Population(10)
# print(pop)
# print("------------------------------")


"""
4、进行读取数据：get_data.py
"""
import tensorflow as tf
import scipy.io as io
import numpy as np
import sklearn.preprocessing as pre

def load_data(path='../MNIST_Data/mnist.npz'):
    """
    Load the MNIST datasets
    此数据集包含60000张28*28的10分类手写字体灰度图训练集、10000张测试集
    此处将60000张训练集分为50000张训练集和10000张验证集
    """
    with np.load(path,allow_pickle=True) as f:
        x_train,y_train = np.reshape(f['x_train'][:50000],(50000,28,28,1)),np.reshape(f['y_train'][:50000],(50000,1))
        x_validate,y_validate = np.reshape(f['x_train'][50000:],(10000,28,28,1)),np.reshape(f['y_train'][50000:],(10000,1))
        x_test,y_test = np.reshape(f['x_test'],(10000,28,28,1)),np.reshape(f['y_test'],(10000,1))

    return [(x_train,y_train),(x_validate,y_validate),(x_test,y_test)]

def get_train_data(batch_size):
    """
    role : 按批次获取训练集数据
    """
    t_image,t_label = load_data()[0]
    train_image = tf.cast(t_image,tf.float32) # 将图片张量进行数据类型的转换
    train_label = tf.cast(t_label,tf.int32)
    # 将关于图片信息的tensor列表随机抽取放进文件名队列(文件系统->文件名队列->内存队列，shuffle参数表示打乱)
    single_image, single_label = tf.train.slice_input_producer([train_image,train_label],shuffle=True)
    # 将图片信息进行标准化，减小像素数值，加速运算
    single_image = tf.image.per_image_standardization(single_image)
    # 将数据按批次送进文件队列
    image_batch, label_batch = tf.train_batch([single_image,single_label],batch_size=batch_size)
    return image_batch,label_batch

def get_validate_data(batch_size):
    """
    role : 按批次获取验证集数据
    """
    t_image,t_label = load_data()[1]
    validate_image = tf.cast(t_image,tf.float32)
    validate_label = tf.cast(t_label,tf.int32)
    single_image, single_label = tf.train.slice_input_producer([validate_image,validate_label],shuffle=True)
    single_image = tf.image.per_image_standardization(single_image)
    image_batch, label_batch = tf.train_batch([single_image,single_label],batch_size=batch_size)
    return image_batch,label_batch

def get_test_data(batch_size):
    """
    role : 按批次获取测试集数据
    """
    t_image,t_label = load_data()[2]
    test_image = tf.cast(t_image,tf.float32)
    test_label = tf.cast(t_label,tf.int32)
    single_image, single_label = tf.train.slice_input_producer([test_image,test_label],shuffle=True)
    single_image = tf.image.per_image_standardization(single_image)
    image_batch, label_batch = tf.train_batch([single_image,single_label],batch_size=batch_size)
    return image_batch,label_batch

# test code:
# print("----------LoadData----------")
# (train_image,train_label),(vali_image,vali_label),(test_image,test_label) = load_data()
# print(train_image.shape,train_label.shape)
# print(train_image[0])
# print(vali_label[0])
# x,y = load_data()[0]
# print(x.shape,y.shape)


"""
5、工具函数：utils.py
"""
import numpy as np
import os
import pickle
from time import gmtime, strftime

def get_data_path():
    return os.getcwd() + '/pops.dat'

def save_populations(gen_no,pops):
    """
    role:保存种群数据
    param gen_no:种群代数
    param pops:种群
    """
    data = {'gen_no':gen_no,'pops':pops,'create_time':strftime("%Y-%m-%d %H:%M:%S",gmtime())}
    path = get_data_path()
    with open(path,'wb') as file_handler:
        pickle.dump(data,file_handler)

def load_population():
    """
    role:加载种群信息
    """
    path = get_data_path()
    with open(path,'rb') as file_handler:
        data = pickle.load(file_handler)
    return data['gen_no'],data['pops'],data['create_time']

def save_offspring(gen_no,pops):
    """
    role:保存子代
    :param gen_no:种群代数
    :param pops：种群
    """
    data = {'gen_no':gen_no,'pops':pops,'create_time':strftime("%Y-%m-%d %H:%M:%S",gmtime())}
    path = os.getcwd() + '/offspring_data/gen_{}.dat'.format(gen_no)
    with open(path,'wb') as file_handler:
        pickle.dump(data,file_handler)

def load_save_log_data():
    """
    role:打印种群个体信息
    """
    file_name = get_data_path()
    with open(file_name,'br') as file_h:
        data = pick.load(file_h)
        print(data)
        pops = data['pops'].pops
        for i in range(len(pops)):
            print(pops[i])

def save_append_individual(indi,file_path):
    """
    role:存储
    """
    with open(file_path,'a') as myfile:
        myfile.write(indi)
        myfile.write("\n")

def flip(f):
        """
        根据概率f随机获取真假值
        """
        if np.random.random() <= f:
            return True
        else:
            return False


"""
6、进行种群个体的适应度评估，即初始化种群：evaluate.py
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim # tf_1.* 版本
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
import numpy as np
import collections
import timeit
import os
import pickle
from datetime import datetime

class Evaluate:
    """
    适应度评估类
    """
    def __init_(self,pops,train_data,train_label,validate_data,validate_label,number_of_channel,epochs,batch_size,train_data_length,validate_data_length):
        """
        role:构造函数
        :param pops : 种群
        :param train_data : 训练集数据
        :param train_label: 训练集标签
        :param validate_data : 验证集数据
        :param validate_label: 验证集标签
        :param number_of_channel : 通道数量
        :param epochs : 迭代次数
        :param batch_size : 每批次数据数量
        :param train_data_length : 训练集数据长度
        :param validate_data_length : 验证集数据长度
        """
        self.pops = pops
        self.train_data = train_data
        self.train_label = train_label
        self.validate_data = validate_data
        self.validate_label = validate_label
        self.number_of_channel = number_of_channel
        self.epochs = epochs # 迭代次数
        self.batch_size = batch_size # 每次批数据数量
        self.train_data_length = train_data_length
        self.validate_data_length = validate_data_length

    def parse_population(self,gen_no):
        """
        role:将种群中的基因组解析成tf能够直接使用的信息
        param gen_no:种群代数
        """
        save_dir = os.getcwd() + '/save_data/gen_{:03d}'.format(gen_no) # 保存每代数据
        tf.gfile.MakeDirs(save_dir)
        history_best_score = 0 # 记录历史最好分数
        for i in range(self.pops.get_pop_size()):
            inid = self.pops.get_individual_at(i) # 获取每个个体
            rs_mean,rs_std,num_connections,new_best = self.parse_individual(indi,self.number_of_channel,i,save_dir,history_best_score)
            # 更新个体适应度参数(准确率平均值/准确率标准差/权重数量/历史最好分数)
            indi.mean = rs_mean
            indi.std = rs_std
            indi.complexity = num_connections
            history_best_score = new_best
            list_save_path = os.getcwd() + '/save_data/gen_{:03d}/pop.txt'.format(gen_no)
            save_append_individual(str(indi),list_save_path) # 保存个体信息

        pop_list = self.pops
        list_save_path = os.getcwd() + '/save_data/gen_{:03d}/pop.dat'.format(gen_no)
        with open(list_save_path,'wb') as file_handler:
            pickle.dump(pop_list,file_handler)

    def parse_individual(self,indi,num_of_input_channel,indi_index,save_path,history_best_score):
        """
        role : 将个体进行解析
        """
        tf.resnet_default_graph() # 清除默认图形堆栈
        # 读取训练集和验证集
        train_data, train_label = get_train_data(self.batch_size)
        validate_data,validate_label = get_validate_data(self.batch_size)
        is_training,train_op,accuracy,cross_entropy,num_connections,merge_summary = self.build_graph(indi_index,num_of_input_channel,indi,train_data,train_label,validate_data,validate_label)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # 每一次训练集整体训练的训练步数
            steps_in_each_epoch = (self.train_data_length//self.batch_size)
            total_steps = int(self.epochs*steps_in_each_epoch) # 总训练步数
            coord = tf.train.Coordinator() # 训练协调器
            # thread = tf.train.start_queue_runners(sess,coord)
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess,coord=coord,daemon=True,start=True))
                for i in range(total_steps):
                    if coord.should_stop():
                        break
                    # is_training设为true，进行训练集训练
                    _,accuracy_str,loss_str,_ = sess.run([train_op,accuracy,cross_entropy,merge_summary],{is_training:True})
                    if i % (2*steps_in_each_epoch) == 0: # 每在训练集上进行两个epoch，就在验证集上训练一次
                        test_total_step = self.validate_data_length//self.batch_size # 一次验证集训练步数
                        test_accuracy_list = []
                        for _ in range(test_total_step):
                            test_accuracy_str,test_loss_str = sess.run([accuracy,cross_entropy],{is_training:False})
                            test_accuracy_list.append(test_accuracy_str)
                            test_loss_list.append(test_loss_str)
                        # 验证集的平均准确率和损失值
                        mean_test_accu = np.mean(test_accuracy_list)
                        mean_test_loss = np.mean(test_loss_list)
                        print('{}, {}, indi:{}, Step:{}/{}, train_loss:{}, acc:{}, test_loss:{}, acc:{}'.format(datetime.now(),i//steps_in_each_epoch,indi_index,i,total_steps,loss_str,accuracy_str,mean_test_loss,mean_test_accu))
                # 全部训练完以后，最后进行一个epoch训练验证集
                test_total_step = self.validate_data_length//self.batch_size
                test_accuracy_list = []
                test_loss_list = []
                for _ in range(test_total_step):
                    test_accuracy_str,test_loss_str = sess.run([accuracy,cross_entropy],{is_training:False})
                    test_accuracy_list.append(test_accuracy_str)
                    test_loss_list.append(test_loss_str)
                mean_test_accu = np.mean(test_accuracy_list)
                mean_test_loss = np.mean(test_loss_list)
                print('{}, test_loss:{}, acc:{}'.format(datetime.now(),mean_test_loss,mean_test_accu))
                mean_acc = mean_test_accu
                if mean_acc > history_best_score:
                    save_mean_acc = tf.Variable(-1, dtype=tf.float32, name='save_mean')
                    save_mean_acc_op = save_mean_acc.assign(mean_acc)
                    sess.run(save_mean_acc_op)
                    saver0 = tf.train.Saver()
                    saver0.save(sess,save_path + '/model')
                    saver0.export_meta_graph(save_path + '/model.meta')
                    history_best_score = mean_acc # 记录最好分数
            except Exception as e:
                print(e)
                coord.request_stop(e)
            finally:
                print('finally')
                coord.request_stop()
                coord.join(threads)

            # 每个个体的适应度为验证集准确度平均值和标准差以及权重个数和历史最好分数
            return mean_test_accu,np.std(test_accuracy_list),num_connections,history_best_score

    def build_graph(self,indi_index,num_of_input_channel,indi,train_data,train_label,validate_data,validate_label):
        """
        role : 创建计算流图
        """
        is_training = tf.placeholder(tf.bool,[]) # 类似形参，占位符，在具体会话中才赋予具体值
        # tf.cond()即if-else条件判断，这里判断是否进行训练，是则获取训练集；否则获取验证集。
        X = tf.cond(is_training,lambda:train_data,lambda:validate_data)
        y_ = tf.cond(is_training,lambda:train_label,lambda:vali_label)
        true_Y = tf.cast(y_,tf.int64) # 用于进行损失函数的比较

        name_preffix = 'I_{}'.format(indi_index) # 个体索引
        num_of_units = indi.get_layer_size() # 个体层数

        #################variabel for convolution operation###############
        last_output_feature_map_size = num_of_input_channel # 下一次操作的卷积核个数
        #################state the connection numbers#####################
        num_connections = 0 # 用来累计连接权重的总数
        # 
        output_list = [] # 用于添加每一次操作的图像结果
        output_list.append(X)
        # arg_scope用于为list_ops设置默认值，这里其实就是为unit设置相关参数
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.crelu, # 激活函数是CReLU函数，对ReLU函数的改进
                            normalizer_fn=slim.batch_norm, # 正则化函数
                            # weights_regularizer=slim.l2_regularizer(0.005),
                            # slim.batch_norm中的参数，字典形式
                            normalizer_params={'is_training':is_training,'decay':0.99}):
            for i in range(num_of_units): # 遍历每一个unit
                current_unit = indi.get_layer_at(i) # 获取当前unit
                if current_unit.type == 1: # 卷积层
                    name_scope = '{}_conv_{}'.format(name_preffix,i)
                    with tf.variable_scope(name_scope): # tf的变量范围
                        # 获取当前层的信息
                        filter_size = [current_unit.filter_width,current_unit.filter_height]
                        mean = current_unit.weight_matrix_mean
                        stddev = current_unit.weight_matrix_std
                        # 对输入图像矩阵进行卷积操作，并作为输出(从截断的正态分布中输出随机值，生成的值服从具有指定平均值和标准偏差的正态分布；bias初始化为0.1常数初始化)
                        conv_H = slim.conv2d(output_list[-1],current_unit.feature_map_size,filter_size,weights_initializer=tf.truncated_normal_initializer(mean=mean,stddev=stddev),biases_initializer=init_ops.constant_initializer(0.1,dtype=tf.float32))
                        output_list.append(conv_H)
                        # 
                        last_output_feature_map_size = current_unit.feature_map_size
                        num_connections += current_unit.feature_map_size*current_unit.filter_width*current_unit.filter_height+current_unit.feature_map_size # weights+biases的个数
                elif current_unit.type == 2: # 池化层
                    name_scope = '{}_pool_{}'.format(name_preffix,i)
                    with tf.variable_scope(name_scope):
                        kernel_size = [current_unit.kernel_width,current_unit.kernel_height]
                        if current_unit.kernel_type < 0.5: # 表示最大值池化，SAME操作
                            pool_H = slim.max_pool2d(output_list[-1],kernel_size=kernel_size,stride=kernel_size,padding='SAME')
                        else: # 平均池化
                            pool_H = slim.avg_pool2d(output_list[-1],kernel_size=kernel_size,stride=kernel_size,padding='SAME')
                        output_list.append(pool_H) # 添加输出结果
                        # 池化操作不改变通道数量，但是改变输出尺寸
                        last_output_feature_map_size = last_output_feature_map_size
                        num_connections += last_output_feature_map_size # 其实可以不需要加，池化层没有参数(或者认为加偏差bias)
                elif current_unit.type == 3: # 全连接层
                    name_scope = '{}_full_{}'.format(name_preffix,i)
                    with tf.variable_scope(name_scope):
                        last_unit = indi.get_layer_at(i-1) # 获取上一层，判断是否是全连接层，进行不同操作
                        if last_unit.type != 3: # 上一层不是全连接层，则使用其信息计算维度
                            input_data = slim.flatten(output_list[-1])  # 对上一层全连接展开
                            input_dim = input_data.get_shape()[1].value # 展开以后的维度即这一层全连接层的输入维度
                        else: # 上一层是全连接层，该层输入维度为神经元个数
                            input_data = output_list[-1]
                            input_dim = last_unit.hidden_neurons_num
                        mean = current_unit.weight_matrix_mean
                        stddev = current_unit.weight_matrix_std
                        # 做全连接之前需要将卷积层或者池化层数据进行展平操作
                        if i < num_of_units - 1: # 判断是否为最后一层，激活函数默认ReLU函数
                            full_H = slim.fully_connected(input_data,num_outputs=current_unit.hidden_neurons_num,weights_initializer=tf.truncated_normal_initializer(mean=mean,stddev=stddev),biases_initializer=init_ops.constant_initializer(0.1,dtype=tf.float32))
                        else: # 最后一层，None显性设置为线性激活
                            full_H = slim.fully_connected(input_data,num_outputs=current_unit.hidden_neurons_num,activation_fn=None,weights_initializer=tf.truncated_normal_initializer(mean=mean,stddev=stddev),biases_initializer=init_ops.constant_initializer(0.1,dtype=tf.float32))
                        output_list.append(full_H)
                        num_connections += input_dim*current_unit.hidden_neurons_num + current_unit.hidden_neurons_num # 全连接操作的权重+偏差bias
                else:
                    raise NameError('No unit with type value {}'.format(current_unit.type))

            #--------------------三个步骤相应API需要细看--------------------
            with tf.name_scope('{}_loss'.format(name_preffix)):
                logits = output_list[-1]
                # regularization_loss = tf.add_n(tf.losses.get_regularization_losses())
                # 交叉熵损失函数
                cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(Labels=true_Y,logits=logits))
            with tf.name_scope('{}_train'.format(name_preffix)):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                if update_ops:
                    updates = tf.group(*update_ops)
                    cross_entropy = control_flow_ops.with_dependencies([updates],cross_entropy)
                optimizer = tf.train.AdamOptimizer() # 优化器
                # 计算梯度和损失
                train_op = slim.learning.create_train_op(cross_entropy,optimizer)
            with tf.name_scope('{}_test'.format(name_preffix)):
                # 获取准确度
                accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),true_Y),tf.float32))

            tf.summary.scalar('loss',cross_entropy)
            tf.summary.scalar('accuracy',accuracy)
            merge_summary = tf.summary.merge_all()


            return is_training,train_op,accuracy,cross_entropy,num_connections,merge_summary


"""
7、进行种群进化：evolve.py
"""
import numpy as np
import tensorflow.examples.tutorials.mnist as input_data
import tensorflow as tf
import collections
import copy

class Evolve_CNN:
    """
    role:进化类
    """
    def __init__(self,m_prob,m_eta,x_prob,x_eta,pupulation_size,train_data,train_label,validate_data,validate_label,number_of_channel,epochs,batch_size,train_data_length,validate_data_length,eta):
        self.m_prob = m_prob # 个体变异概率
        self.m_eta = m_eta # 个体变异系数
        self.x_prob = x_prob # unit之间发生交叉的概率
        self.x_eta = x_eta # SBX中的系数
        self.population_size = population_size # 种群个体数量，尽量为偶数
        self.train_data = train_data # 训练集数据
        self.train_label = train_label # 训练集标签
        self.validate_data = validate_data # 验证集数据
        self.validate_label = validate_label # 验证集标签
        self.epochs = epochs # 批次
        self.eta = eta
        self.number_of_channel = number_of_channel # 通道数量
        self.batch_size = batch_size
        self.train_data_length = train_data_length
        self.validate_data_length = validate_data_length

    def initialize_population(self):
        print("initializing population with number {}...".format(self.population_size))
        self.pops = Population(self.population_size) # 初始化种群类
        # 所有的初始种群均需被保存
        save_populations(gen_no=-1,pops=self.pops)

    def evaluate_fitness(self,gen_no):
        """
        role:进行适应度评估
        """
        print("evaluate_fitness")
        # 创建Evaluate类对种群进行适应度评估
        evaluate = Evaluate(self.pops,self.train_data,self.train_label,self.validate_data,self.validate_label,self.number_of_channel,self.epochs,self.batch_size,self.train_data_length,self.validate_data_length)
        evaluate.parse_population(gen_no)
        save_populations(gen_no=gen_no,pops=self.pops)
        # 打印更新后的种群信息
        print(self.pops)

    def recombinate(self,gen_no):
        """
        role:后代繁衍(交叉变异)
        :param gen_no:种群代数
        """
        print("crossover and mutation...")
        # 添加后代list
        offspring_list = []
        for _ in range(int(self.pops.get_pop_size()/2)): # 数量折半
            p1 = self.tournament_selection() # 找到winner p1
            p2 = self.tournament_selection() # 找到winner p2
            # crossover 操作
            offset1,offset2 = self.crossover(p1,p2)
            # 后代进行mutation操作
            offset1.mutation()
            offset2.mutation()
            offspring_list.append(offset1)
            offspring_list.append(offset2)
        offspring_pops = Population(0) # 初始化空的后代种群
        offspring_pops.set_populations(offspring_list) # 更新后代种群
        save_offspring(gen_no,offspring_pops) # 保存第gen_no代的子代
        # 评估新的种群个体
        evaluate = Evaluate(self.pops,self.train_data,self.train_label,self.validate_data,self.validate_label,self.number_of_channel,self.epochs,self.batch_size,self.train_data_length,self.validate_data_length)
        evaluate.parse_population(gen_no) # 更新个体的适应度值
        # 保存新代种群
        self.pops.pops.extend(offspring_pops.pops) # 将父代个体和子代个体混合
        save_populations(gen_no=gen_no,pops=self.pops) # 保存此代的混合种群

    def environmental_selection(self,gen_no):
        """
        role:环境选择
        :param gen_no:种群代数
        """
        assert(self.pops.get_pop_size() == 2*self.population_size) # 双倍数量关系
        elistsm = 0.2 # 精英率
        e_count = int(np.floor(self.population_size*elistsm/2)*2) # 选择数量每代保持一致
        indi_list = self.pops.pops # 获取个体list
        indi_list.sort(key=lambda x:x.mean,reverse=True) # 按准确率平均值高低降序
        elistsm_list = indi_list[0:e_count]

        # 将剩下的个体打乱顺序
        left_list = indi_list[e_count:]
        np.random.shuffle(left_list) 
        np.random.shuffle(left_list)

        for _ in range(self.population_size-e_count): # 要选择个体保证新一代种群个体数量保持N
            i1 = randint(0,len(left_list))
            i2 = randint(0,len(left_list))
            winner = self.selection(left_list[i1],left_list[i2])
            elistsm_list.append(winner)

        self.pops.set_populations(elistsm_list) # 更新此代的新种群
        save_populations(gen_no=gen_no, pops=self.pops)
        np.random.shuffle(self.pops.pops)

    def crossover(self,p1,p2):
        """
        role:父代之间交叉操作
        :param p1:父代个体1
        :param p2:父代个体2
        """
        p1 = copy.deepcopy(p1) # 深拷贝
        p2 = copy.deepcopy(p2) 
        p1.clear_state_info() # 清除状态信息
        p2.clear_state_info()

        # Unit Collection
        p1_conv_index_list = []
        p1_conv_layer_list = []
        p1_pool_index_list = []
        p1_pool_layer_list = []
        p1_full_index_list = []
        p1_full_layer_list = []

        p2_conv_index_list = []
        p2_conv_layer_list = []
        p2_pool_index_list = []
        p2_pool_layer_list = []
        p2_full_index_list = []
        p2_full_layer_list = []

        # 论文中的将同种unit进行list化
        for i in range(p1.get_layer_size()):
            unit = p1.get_layer_at(i)
            if unit.type == 1:
                p1_conv_index_list.append(i)
                p1_conv_layer_list.append(unit)
            elif unit.type == 2:
                p1_pool_index_list.append(i)
                p1_pool_layer_list.append(unit)
            else:
                p1_full_index_list.append(i)
                p1_full_layer_list.append(unit)

        for i in range(p2.get_layer_size()):
            unit = p2.get_layer_at(i)
            if unit.type == 1:
                p2_conv_index_list.append(i)
                p2_conv_layer_list.append(unit)
            elif unit.type == 2:
                p2_pool_index_list.append(i)
                p2_pool_layer_list.append(unit)
            else:
                p2_full_index_list.append(i)
                p2_full_layer_list.append(unit)

        # Unit Aligh and Crossover
        # 卷积层进行交叉
        l = min(len(p1_conv_layer_list),len(p2_conv_layer_list))
        for i in range(l):
            unit_p1 = p1_conv_layer_list[i]
            unit_p2 = p2_conv_layer_list[i]
            if flip(self.x_prob):
                # filter size
                this_range_fs = p1.filter_size_range # 获取卷积核尺寸范围
                w1 = unit_p1.filter_width # 获取p1该层卷积核大小
                w2 = unit_p2.filter_width # 获取p2该层卷积核大小
                # 使用sbx算子产生新的子代卷积核大小参数
                n_w1,n_w2 = self.sbx(w1,w2,this_range_fs[0],this_range_fs[-1],self.x_eta)
                unit_p1.filter_width = int(n_w1)
                unit_p1.filter_height = int(n_w1)
                unit_p2.filter_width = int(n_w2)
                unit_p2.filter_height = int(n_w2)
                # feature map size
                this_range_fms = p1.feature_map_size_range
                s1 = unit_p1.feature_map_size # 获取p1该层卷积核个数
                s2 = unit_p2.feature_map_size # 获取p2该层卷积核个数
                # 使用sbx算子产生新的子代卷积核个数参数
                n_s1,n_s2 = self.sbx(s1,s2,this_range_fms[0],this_range_fms[-1],self.x_eta)
                unit_p1.feature_map_size = int(n_s1)
                unit_p2.feature_map_size = int(n_s2)
                # mean
                this_range_mean = p1.mean_range 
                m1 = unit_p1.weight_matrix_mean # 获取p1该层初始化权重平均值
                m2 = unit_p2.weight_matrix_mean # 获取p2该层初始化权重平均值
                n_m1,n_m2 = self.sbx(m1,m2,this_range_mean[0],this_range_mean[-1],self.x_eta)
                unit_p1.weight_matrix_mean = n_m1
                unit_p2.weight_matrix_mean = n_m2
                # std
                this_range_std = p1.std_range
                std1 = unit_p1.weight_matrix_std
                std2 = unit_p2.weight_matrix_std
                n_std1,n_std2 = self.sbx(std1,std2,this_range_std[0],this_range_std[-1],self.x_eta)
                unit_p1.weight_matrix_std = n_std1
                unit_p2.weight_matrix_std = n_std2

            # 更新交叉后的子代层
            p1_conv_layer_list[i] = unit_p1
            p2_conv_layer_list[i] = unit_p2

        # 池化层进行交叉
        l = min(len(p1_pool_layer_list),len(p2_pool_layer_list))
        for i in range(l):
            unit_p1 = p1_pool_layer_list[i]
            unit_p2 = p2_pool_layer_list[i]
            if flip(self.x_prob):
                # filter size
                this_range_fs = p1.pool_kernel_size_range # 获取池化核尺寸范围
                k1 = np.log2(unit_p1.kernel_width) # 获取p1该层池化核大小
                k2 = np.log2(unit_p2.kernel_width) # 获取p2该层池化核大小
                # 使用sbx算子产生新的子代池化核大小参数
                n_k1,n_k2 = self.sbx(k1,k2,this_range_fs[0],this_range_fs[-1],self.x_eta)
                n_k1 = int(np.power(2,n_k1))
                n_k2 = int(np.power(2,n_k2))
                unit_p1.kernel_width = n_k1
                unit_p1.kernel_height = n_k1
                unit_p2.kernel_width = n_k2
                unit_p2.kernel_height = n_k2
                # pool type
                t1 = unit_p1.kernel_type
                t2 = unit_p2.kernel_type
                n_t1,n_t2 = self.sbx(t1,t2,0,1,self.x_eta)
                unit_p1.kernel_type = n_t1
                unit_p2.kernel_type = n_t2

            # 更新交叉后的子代层
            p1_pool_layer_list[i] = unit_p1
            p2_pool_layer_list[i] = unit_p2

        # 全连接层进行交叉
        l = min(len(p1_full_layer_list),len(p2_full_layer_list))
        for i in range(l-1):
            unit_p1 = p1_full_layer_list[i]
            unit_p2 = p2_full_layer_list[i]
            if filp(self.x_prob):
                this_range_neurons = p1.hidden_neurons_range
                n1 = unit_p1.hidden_neurons_num # 获取p1该层神经元个数
                n2 = unit_p2.hidden_neurons_num # 获取p2该层神经元个数
                n_n1,n_n2 = self.sbx(n1,n2,this_range_neurons[0],this_range_neurons[1],self.x_eta)
                unit_p1.hidden_neurons_num = int(n_n1)
                unit_p2.hidden_neurons_num = int(n_n2)
                # std and mean
                this_range_mean = p1.mean_range
                m1 = unit_p1.weight_matrix_mean
                m2 = unit_p2.weight_matrix_mean
                n_m1,n_m2 = self.sbx(m1,m2,this_range_mean[0],this_range_mean[-1],self.x_eta)
                unit_p1.weight_matrix_mean = n_m1
                unit_p2.weight_matrix_mean = n_m2

                this_range_std = p1.std.range
                std1 = unit_p1.weight_matrix_std
                std2 = unit_p2.weight_matrix_std
                n_std1,n_std2 = self.sbx(std1,std2,this_range_std[0],this_range_std[-1],self.x_eta)
                unit_p1.weight_matrix_std = n_std1
                unit_p2.weight_matrix_std = n_std2

            p1_full_layer_list[i] = unit_p1
            p2_full_layer_list[i] = unit_p2

        # 最后一层全连接层，神经元个数不变化
        unit_p1 = p1_full_index_list[-1]
        unit_p2 = p2_full_layer_list[-1]
        if filp(self.x_prob):
            # std and mean
            this_range_mean = p1.mean_range
            m1 = unit_p1.weight_matrix_mean
            m2 = unit_p2.weight_matrix_mean
            n_m1,n_m2 = self.sbx(m1,m2,this_range_mean[0],this_range_mean[-1],self.x_eta)
            unit_p1.weight_matrix_mean = n_m1
            unit_p2.weight_matrix_mean = n_m2

            this_range_std = p1.std_range
            std1 = unit_p1.weight_matrix_std
            std2 = unit_p2.weight_matrix_std
            n_std1,n_std2 = self.sbx(std1,std2,this_range_std[0],this_range_std[-1],self.x_eta)
            unit_p1.weight_matrix_std = n_std1
            unit_p2.weight_matrix_std = n_std2

        p1_full_layer_list[-1] = unit_p1
        p2_full_layer_list[-1] = unit_p2

        p1_units = p1.indi # 获取父代p1的units list,此处为浅拷贝因此共用内存直接改值
        # 将交叉后代层重构成新的个体
        # 实现过程利用index list和layer list，思路巧妙
        for i in range(len(p1_conv_index_list)): # 将交叉后的卷积层放回原位
            p1_units[p1_conv_index_list[i]] = p1_conv_layer_list[i]
        for i in range(len(p1_pool_index_list)): # 将交叉后的池化层放回原位
            p1_units[p1_pool_index_list[i]] = p1_pool_layer_list[i]
        for i in range(len(p1_full_index_list)): # 将交叉后的全连接层放回原位
            p1_units[p1_full_index_list[i]] = p1_full_layer_list[i]
        p1.indi = p1_units

        p2_units = p2.indi # 获取父代p2的units list,此处为浅拷贝因此共用内存直接改值
        # 将交叉后代层重构成新的个体
        # 实现过程利用index list和layer list，思路巧妙
        for i in range(len(p2_conv_index_list)): # 将交叉后的卷积层放回原位
            p1_units[p2_conv_index_list[i]] = p2_conv_layer_list[i]
        for i in range(len(p1_pool_index_list)): # 将交叉后的池化层放回原位
            p1_units[p2_pool_index_list[i]] = p2_pool_layer_list[i]
        for i in range(len(p1_full_index_list)): # 将交叉后的全连接层放回原位
            p1_units[p2_full_index_list[i]] = p2_full_layer_list[i]
        p2.indi = p2_units

    def sbx(self,v1,v2,xl,xu,eta):
        """
        role:进行模拟二进制交叉算子操作
        :param v1:变异参数1
        :param v2:变异参数2
        :param xl:参数范围下限
        :param xu:参数范围上限
        :param eta:η系数
        """
        if flip(0.5): # 0.5的随机概率
            if abs(v1-v2) > 1e-14:
                x1 = min(v1,v2)
                x2 = max(v1,v2)
                r = np.random.random()
                beta = 1.0 + (2.0 * (x1-x2) / (x2-x1))
                alpha = 2.0 - beta**-(eta+1)
                if r <= 1.0 / alpha:
                    beta_q = (r*alpha)**(1.0 / (eta+1))
                else:
                    beta_q = (1.0 / (2.0-r*alpha))**(1.0/(eta+1))
                c1 = 0.5*(x1+x2-beta_q*(x2-x1))
                beta = 1.0+(2.0*(xu-x2)/(x2-x1))
                alpha = 2.0-beta**-(eta+1)
                if r <= 1.0/alpha:
                    beta_q = (r*alpha)**(1.0/(eta+1))
                else:
                    beta_q = (1.0/(2.0-r*alpha))**(1.0/(eta+1))
                c2 = 0.5*(x1+x2+beta_q*(x2-x1))
                c1 = min(max(c1,xl),xu)
                c2 = min(max(c2,xl),xu)
                if flip(0.5):
                    return c2,c1
                else:
                    return c1,c2
            else:
                return v1,v2
        else:
            return v1,v2

    def tournament_selection(self):
        """
        role:锦标赛选择，随机选择两个父代比较进行胜出
        """
        # 随机选取父代
        ind1_id = np.random.randint(0,self.pops.get_pop_size())
        ind2_id = np.random.randint(0,self.pops.get_pop_size())
        ind1 = self.pops.get_individual_at(ind1_id)
        ind2 = self.pops.get_individual_at(ind2_id)
        winner = self.selection(ind1,ind2) # 松弛二进制竞标赛选择
        return winner

    def selection(self,ind1,ind2):
        """
        role:松弛二进制锦标赛选择(Slack Binary Tournament Selection)
        """
        mean_threshold = 0.05 # 平均值阈值
        complexity_threshold = 100 # 参数数量阈值
        if ind1.mean > ind2.mean: # ind1比ind2的准确率平均值大
            if ind1.mean - ind2.mean > mean_threshold: # 平均值大于阈值选择ind1
                return ind1
            else: # 平均值在阈值范围内，查看参数数量关系
                if ind1.complexity - ind2.complexity > complexity_threshold:
                    return ind2
                else:
                    if ind1.std < ind2.std:
                        return ind1
                    else:
                        return ind2
        else:
            if ind2.mean - ind1.mean > mean_threshold: # 平均值大于阈值选择ind1
                return ind2
            else: # 平均值在阈值范围内，查看参数数量关系
                if ind2.complexity - ind1.complexity > complexity_threshold:
                    return ind1
                else:
                    if ind2.std < ind1.std:
                        return ind2
                    else:
                        return ind1

def main():
    """
    role:main function
    """
    print("hello world")


if __name__ == '__main__':
    main()