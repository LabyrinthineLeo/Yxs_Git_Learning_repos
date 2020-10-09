# coding  : utf-8
# ResNet代码复现
# @Author : Labyrinthine Leo
# @Time   : 2020.10.08

from tensorflow.keras import layers, Model, Sequential

class BasicBlock(layers.Layer):
	"""
	role:构建2层的残差结构(用于18/34)，具有实线结构和虚线结构(虚线结构在维度升高的情况下使用，论文中有Option A和B，一般用B)
	"""
	expansion = 1  # 对于浅层和深层的residual结构中是否扩张的系数

	def __init__(self,out_channel,strides=1,downsample=None,**kwargs):
		"""
		role:构造函数
		param out_channel:卷积层所使用的卷积核个数，即输出数据的通道数
		param strides:步长默认值1，在虚线结构中会为2则传入2
		param downsample:下采样函数，shortcut中使用
		param kwargs:字典参数
		"""
		super(BasicBlock,self).__init__(**kwargs)
		# -----------------conv1------------------
		# 由于论文中每层卷积后和激活函数前使用BN层操作，因此不需要设置bias
		self.conv1 = layers.Conv2D(out_channel,kernel_size=3,strides=strides,
									padding="SAME",use_bias=False)
		self.bn1 = layers.BatchNormalization(momentum=0.9,epsilon=1e-5) # 动量和ε
		# -----------------conv2------------------
		# 第2层卷积层实线和虚线结构步长均为1
		self.conv2 = layers.Conv2D(out_channel,kernel_size=3,strides=1,
									padding="SAME",use_bias=False)
		self.bn2 = layers.BatchNormalization(momentum=0.9,epsilon=1e-5) # 动量和ε
		# ----------------------------------------
		self.downsample = downsample # 下采样函数
		self.relu = layers.ReLU() # 激活函数
		self.add = layers.Add() # 逐元素加操作

	def call(self,inputs,training=False):
		"""
		role:定义residual结构正向传播(数据流方向)
		param inputs:输入数据
		param training:用于控制BN在训练过程和验证过程展示不同的状态
		"""
		identity = inputs # 用于identity mapping
		# shortcut选择实线还是虚线结构则取决于下采样函数
		if self.downsample is not None:
			identity = self.downsample(inputs)

		x = self.conv1(inputs)
		x = self.bn1(x,training=training)
		x = self.relu(x)

		x = self.conv2(x)
		x = self.bn2(x,training=training)

		x = self.add([x,identity])
		x = self.relu(x)

		return x

class Bottleneck(layers.Layer):
	"""
	role:构建3层的残差结构(用于50/101/152)，具有实线结构和虚线结构(虚线结构在维度升高的情况下使用，论文中有Option A和B，一般用B)
	"""
	expansion = 4  # 对于浅层和深层的residual结构中卷积核个数是否扩张的系数

	def __init__(self,out_channel,strides=1,downsample=None,**kwargs): 
		"""
		role:构造函数
		param out_channel:第一层卷积层的filter个数
		param strides:步长默认为1
		param downsample:shortcut上的下采样操作
		param kwargs:字典参数
		"""
		super(Bottleneck,self).__init__(**kwargs)
		# -----------------conv1------------------
		# 设置name为与预训练模型权重参数name匹配，filter size为1则padding使用valid
		self.conv1 = layers.Conv2D(out_channel,kernel_size=1,strides=1,
									use_bias=False,name="conv1")
		self.bn1 = layers.BatchNormalization(momentum=0.9,epsilon=1e-5,name="conv1/BatchNorm")
		# -----------------conv2------------------
		self.conv2 = layers.Conv2D(out_channel,kernel_size=3,strides=strides,
									use_bias=False,padding="SAME",name="conv2")
		self.bn2 = layers.BatchNormalization(momentum=0.9,epsilon=1e-5,name="conv2/BatchNorm")
		# -----------------conv3------------------
		# 通道会扩张
		self.conv3 = layers.Conv2D(out_channel*self.expansion,kernel_size=1,strides=1,
									use_bias=False,name="conv3")
		self.bn3 = layers.BatchNormalization(momentum=0.9,epsilon=1e-5,name="conv3/BatchNorm")
		# ----------------------------------------
		self.relu = layers.ReLU()
		self.downsample = downsample
		self.add = layers.Add()

	def call(self,inputs,training=False):
		"""
		role:定义residual结构正向传播(数据流方向)
		param inputs:输入数据
		param training:用于控制BN在训练过程和验证过程展示不同的状态
		"""
		identity = inputs # 用于identity mapping
		# shortcut选择实线还是虚线结构则取决于下采样函数
		if self.downsample is not None:
			identity = self.downsample(inputs)

		x = self.conv1(inputs)
		x = self.bn1(x,training=training)
		x = self.relu(x)

		x = self.conv2(x)
		x = self.bn2(x,training=training)
		x = self.relu(x)

		x = self.conv3(x)
		x = self.bn3(x,training=training)

		x = self.add([x,identity])
		x = self.relu(x)

		return x

# 使用functional API方法搭建模型，相对subclass方法在print summary时更加方便
def resnet(block,blocks_num,im_width=224,im_height=224,num_classes=3,include_top=Ture):
	"""
	role:搭建resnet模型结构
	noted:tensorflow中的tensor顺序NHWC(即数量、高、宽、通道)，here(None,224,224,3)
	param block:block类型(BasicBlock/Bottleneck)
	param block_num:对应论文中conv2_x ~ conv5_x中每部分block的个数	，列表类型(ex.:resnet34[3,4,6,3])
	param im_width:width of image
	param im_height:height of image
	param num_classes:类别数
	param include_top:是否使用顶层结构(即最后的平均池化下采样层和全连接层)
	"""
	# 输入图片数据
	input_image = layers.Input(shape=(im_height,im_width,3),dtype="float32")
	# 进行初始的卷积和池化操作
	x = layers.Conv2D(filters=64,kernel_size=7,strides=2,
						padding="SAME",use_bias=False,name="conv1")(input_image)
					




		
