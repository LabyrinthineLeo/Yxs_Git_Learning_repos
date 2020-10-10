# coding  : utf-8
# role    : ResNet代码复现
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
def resnet(block,blocks_num,im_width=224,im_height=224,num_classes=1000,include_top=True):
	"""
	role:搭建resnet模型结构
	noted:tensorflow中的tensor顺序NHWC(即数量、高、宽、通道)，here(None,224,224,3)
	param block:block类型(BasicBlock/Bottleneck)
	param blocks_num:对应论文中conv2_x ~ conv5_x中每部分residual结构的个数，列表类型(ex.:resnet34[3,4,6,3])
	param im_width:width of image
	param im_height:height of image
	param num_classes:类别数,默认1000类
	param include_top:是否使用顶层结构(即最后的平均池化下采样层和全连接层)
	"""
	# 输入图片数据
	input_image = layers.Input(shape=(im_height,im_width,3),dtype="float32")
	# 进行初始的卷积和池化操作
	x = layers.Conv2D(filters=64,kernel_size=7,strides=2,
						padding="SAME",use_bias=False,name="conv1")(input_image)
	x = layers.BatchNormalization(momentum=0.9,epsilon=1e-5,name="conv1/BatchNorm")(x)		
	x = layers.ReLU()(x)
	# 这里池化层步长为2，尺寸折半
	x = layers.MaxPool2D(pool_size=3,strides=2,padding="SAME")(x)

	# 使用blocks_num数据构建block结构
	x = parse_layer(block,x.shape[-1],64,blocks_num[0],name="block1")(x)
	x = parse_layer(block,x.shape[-1],128,blocks_num[1],strides=2,name="block2")(x)
	x = parse_layer(block,x.shape[-1],256,blocks_num[2],strides=2,name="block3")(x)
	x = parse_layer(block,x.shape[-1],512,blocks_num[3],strides=2,name="block4")(x)

	if include_top: # 包含顶层结构
		x = layers.GlobalAvgPool2D()(x) # 全局平均池化+展平操作
		x = layers.Dense(num_classes,name="logits")(x) # 全连接层操作
		predict = layers.Softmax()(x) # 进行类别概率分布预测
	else:
		predict = x # 方便自己添加其他的层结构

	model = Model(inputs=input_image,outputs=predict)

	return model


def parse_layer(block,in_channel,channel,blocks_num,name,strides=1):
	"""
	role:利用blocks_num解析为block结构
	param block:残差结构类型(2层/3层)
	param in_channel:输入数据的通道(深度)
	param channel:每个block第1层的卷积核个数
	param blocks_num:每一个block的残差结构数量
	param name:
	param strides:
	"""
	downsample = None
	# 判断第1个block的类型(64*1和64*4,所以深层结构该条件成立;即basic block则跳过该条件)
	# 因为在浅层(18/34)中第1个block无虚线残差结构，但是在深层(50/101/152)中第1个block有虚线残差结构，并且此时第1个block中的虚线结构不会降尺寸
	# 同时在后面3个block，两者shortcut参数相同
	if strides != 1 or in_channel != channel * block.expansion:
		# bottleneck中shortcut的下采样函数
		downsample = Sequential([
			# shortcut中的降尺寸升维卷积操作
			layers.Conv2D(out_channel*block.expansion,kernel_size=1,strides=strides,
						use_bias=False,name="conv1"),
			layers.BatchNormalization(momentum=0.9,epsilon=1.001e-5,name="BatchNorm")
		],name="shortcut")

	# 存放一个block的layers列表
	layers_list = []
	# 首先添加虚线残差结构
	layers_list.append(block(out_channel,downsample=downsample,strides=strides,name="unit_1"))

	# 添加实现残差结构层
	for inde in range(1,blocks_num):
		layers_list.append(block(out_channel,name="unit_"+str(index+1)))

	return Sequential(layers_list,name=name)


# 构建不同层网络结构
def resnet18(im_width=224,im_height=224,num_classes=1000):
	"""
	role:构建resnet18
	"""
	return resnet(BasicBlock,[2,2,2,2],im_width,im_height,num_classes)

def resnet34(im_width=224,im_height=224,num_classes=1000):
	"""
	role:构建resnet34
	"""
	return resnet(BasicBlock,[3,4,6,3],im_width,im_height,num_classes)

def resnet50(im_width=224,im_height=224,num_classes=1000,include_top=True):
	"""
	role:构建resnet50
	"""
	return resnet(BasicBlock,[3,4,6,3],im_width,im_height,num_classes,include_top)

def resnet101(im_width=224,im_height=224,num_classes=1000,include_top=True):
	"""
	role:构建resnet101
	"""
	return resnet(BasicBlock,[3,4,23,3],im_width,im_height,num_classes,include_top)

def resnet152(im_width=224,im_height=224,num_classes=1000,include_top=True):
	"""
	role:构建resnet152
	"""
	return resnet(BasicBlock,[3,8,36,3],im_width,im_height,num_classes,include_top)


if __name__ == '__main__':
	resnet_my = resnet101(num_classes=3)