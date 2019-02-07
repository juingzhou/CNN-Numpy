# 使用Numpy从头开始实现简单的CNN
- 在本项目中，仅使用Numpy创建CNN，会创建卷积层(conv)、ReLU层和最大池化层(MaxPooling)
###主要步骤有如下:
1. 读取输入图像
2. 准备filter
3. 卷积层： 使用filter对输入图像执行卷积操作
4. ReLU层： 将ReLU激活函数应用于特征图(卷积层的输出)
5. 最大池化层： 在ReLU层的输出上应用池化操作
6. 堆叠卷积层,ReLU层和最大池化层