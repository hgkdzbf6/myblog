---
layout: post
title:  "Python OpenCV常用方法"
date:   2019-04-10 08:55:05 +0800
categories: cv
---

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## 主要是博客的整理

<!-- ## 更细分的工具函数，文件操作： -->

### 从文件夹当中读取所有文件

{% codeblock lang:python %}
# 返回所有图片文件名
def readAll(dirname, filter = {}):
    names = os.listdir(dirname)
    lists = [name for name in names 
        if name.split('.')[-1] in filter]
    print(lists)
    return lists
# 使用方法：list = readAll('aokeng',{'jpg'})
{% endcodeblock %}

## 图像操作：

### python的图像处理工具

参考[这里](https://www.jianshu.com/p/3977d674da85)
一种有这么几种：

1. numpy + cv2
2. matplotlib.pyplot
3. PIL.Image

很头疼的一点是他们之间怎么转换，转换关系如下所示：

#### cv2转PIL.Image

{% codeblock lang:python %}
import cv2
from PIL import Image
import numpy as np
image = Image.open("plane.jpg")  # 这是一个PIL.Image图像
img = np.asarray(image)          # 这就转化成了cv2图像
img = np.array(image)            # 这个也行

img = cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR)   # 针对彩色图像
{% endcodeblock %}

#### PIL.Image转化成cv2

{% codeblock lang:python %}
import cv2
from PIL import Image
import numpy as np
img = cv2.imread("plane.jpg")     # 读入文件
image = Image.fromarray(img)      # 转化成功

image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  # 针对彩色图像
{% endcodeblock %}

#### 判断图像数据是否是OpenCV格式

{% codeblock lang:python %}
isinstance(img, np.ndarray)
{% endcodeblock %}

`plt.imread`和`PIL.Image.open`读入的都是RGB顺序，而`cv2.imread`读入的是BGR顺序。使用时需要倍加注意。

不过灰度图没这个要求。

#### 重要！灰度图转化成彩色图，经常用！因为要在黑白图上面圈东西，cv2的版本

{% codeblock lang:python %}
img = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
{% endcodeblock %}

#### 画奇奇怪怪的形状：

{% codeblock lang:python %}
cv2.line(canvas, (300, 0), (0, 300), (255,255,0), 3) # 画线，起点，终点
cv2.rectangle(canvas, (10, 10), (60, 60), (0,0,255)) # 画长方形
# 第二个参数是圆心，第三个参数是半径，最后的一个-1表示填充内部
cv2.circle(canvas, tuple(pt), radius, color, -1)     
# 添加文字
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,’OpenCV’,(80,90), font, 2,(255,255,255),3)

# 画椭圆，第二个参数是中心坐标，第三个参数是长轴和短轴，第四个参数是旋转的角度
# 第五个是颜色，最后一个是线宽或者填充。
cv2.ellipse(img, (150,150),(10,5),0,0,180,(0,127,0),-1)

# 画多边形，第三个参数是是否闭合的意思，然后是颜色和线宽
Pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
Pts=Pts.reshape((-1,1,2))
cv2.polylines(img,[Pts],True,(0,255,255),3)

{% endcodeblock %}

#### 图像显示说明：

`1.` opencv当中的图像显示需要先建立一个窗口，这么建立：
{% codeblock lang:python %}
    cv2.namedWindow('the_image',cv2.WINDOW_NORMAL)
    cv2.imshow('the_image',img)
{% endcodeblock %}

`2.`  matplotlib建立的时候非常好，能够嵌入在notebook当中，前提是要写magic的`%`
{% codeblock lang:python %}
%matplotlib inline
plt.rcParams['figure.figsize'] = (20.0, 16.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2
plt.imshow(img)
{% endcodeblock %}

还能够添加文字标题栏，标签栏等，参考[这里](https://blog.csdn.net/helunqu2017/article/details/78659490)
{% codeblock lang:python %}
plt.title('Interesting Graph',fontsize='large'，fontweight='bold') 设置字体大小与格式
plt.title('Interesting Graph',color='blue') 设置字体颜色
plt.title('Interesting Graph',loc ='left') 设置字体位置
plt.title('Interesting Graph',verticalalignment='bottom') 设置垂直对齐方式
plt.title('Interesting Graph',rotation=45) 设置字体旋转角度
plt.title('Interesting',bbox=dict(facecolor='g', edgecolor='blue', alpha=0.65 )) 标题边框
{% endcodeblock %}

{% codeblock lang:python %}
# 添加标注
plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),arrowprops=dict(facecolor='black', shrink=0.05))
# 添加文字
plt.text(x,y,string,fontsize=15,verticalalignment="top",horizontalalignment="right")
# 还能添加数学公式，niubility
plt.text(0.6, 0.6, r'$\mathcal{A}\mathrm{sin}(2 \omega t)$',fontsize=20)
{% endcodeblock %}

### 返回图像的卷积操作

很多运算都能表示为图像的卷积操作，比如说差分算子，sobel算子等。参考[这里](https://blog.csdn.net/m0_38032942/article/details/82230059)

{% codeblock lang:python %}

def filter(img, kernel):
    # 参考： https://blog.csdn.net/m0_38032942/article/details/82230059
    # sobel算子，横向边缘检测： [[-1, -2, -1],[0,0,0],[1,2,1]]
    # prewitt算子，横向边缘检测： [[-1, -1, -1],[0,0,0],[1,1,1]]
    # sobel算子，纵向边缘检测： [[-1, 0, 1],[-2,0,2],[-1,0,1]]
    # prewitt算子，纵向边缘检测： [[-1, 0, 1],[-1,0,1],[-1,0,1]]
    # laplacian算子： [[-1, -1, -1],[-1,8,-1],[-1,-1,-1]]

    # 锐化算子： [[-1, -1, -1],[-1,9,-1],[-1,-1,-1]]
    # 更加强调边缘的卷积核： [[-1, -1, -1],[-1,7,-1],[-1,-1,-1]]
    # 浮雕效果的卷积核 [[-6, -3, 0],[-3,  1, 3],[0,  3, 6]]

    # 均值滤波  [[1/9, 1/9, 1/9],[1/9, 1/9, 1/9],[1/9, 1/9, 1/9]]
    kernel = [[-1, -1, -1],[-1,7,-1],[-1,-1,-1]]
    return cv2.filter2D(img,-1,kernel)
    
{% endcodeblock %}

### 阈值操作，也就是二值化

这个参考这里，[图像的阈值处理](https://blog.csdn.net/on2way/article/details/46812121)：

#### 经典的阈值处理方法：

{% codeblock lang:python %}
    ret,img_threshold = cv2.threshold(img,70,255, cv2.THRESH_BINARY)
{% endcodeblock %}

这里要记住第一个参数并不是处理之后的图像，而且这个实际采用的阈值，像上面的这个例子里面就是70，ret也是70，但是有一些例子当中会根据图像的一些信息来计算出一个自适应的图像，所以这个结果有可能和第2个参数不一样。

参数说明如下：
{% codeblock lang:python %}
cv2.THRESH_BINARY(黑白二值)
cv2.THRESH_BINARY_INV(黑白二值反转)
cv2.THRESH_TRUNC (得到的图像为多像素值)
cv2.THRESH_TOZERO
cv2.THRESH_TOZERO_INV
{% endcodeblock %}

#### 自适应阈值：

函数是这样的：
{% codeblock lang:python %}
cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) -> dst{% endcodeblock %}

- 第一个原始图像
- 第二个像素值上限
- 第三个自适应方法Adaptive Method：
  - `cv2.ADAPTIVE_THRESH_MEAN_C`：领域内均值
  - `cv2.ADAPTIVE_THRESH_GAUSSIAN_C`：领域内像素点加权和，权重为一个高斯窗口
- 第四个值的赋值方法：只有`cv2.THRESH_BINARY`和`cv2.THRESH_BINARY_INV`
- 第五个Block size:规定领域大小（一个正方形的领域）
- 第六个常数C，阈值等于均值或者加权值减去这个常数（为0相当于阈值 就是求得领域内均值或者加权值）

这种方法理论上得到的效果更好，相当于在动态自适应的调整属于自己像素点的阈值，而不是整幅图像都用一个阈值。
`但是实际上这个方法用起来很不好，会引入很多其他的杂点`。

#### Otus方法

原理：基于图像的背景和前景的像素差很大。首先计算灰度直方图，得到双峰图像，然后Otsu’s算法会找到一个"理论上的"最佳分割，也就是阈值所以就有了给的阈值为0，然后ret输出一个阈值这种情况，用法和经典方法差不多，注意最后一项里面加上`cv2.THRESH_OTSU`方法。如下所示：

{% codeblock lang:python %}
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
{% endcodeblock %}

然后ret2当中会返回计算出来的阈值。

### 腐蚀和膨胀操作：

这属于形态学处理的范畴，一般都先把图像转化为二值化图像再进行操作。

首先需要新建一个
{% codeblock lang:python %}
kernel = np.ones((5,5),np.uint8)  
{% endcodeblock %}

这个kernel可能是任何形状，常见的就是圆形，正方形等形状，如果核越大那么被处理之后的响应也就越大。

然后就是腐蚀和膨胀操作了。腐蚀和膨胀都是针对白色的点为准的，腐蚀是白色区域变少，膨胀是白色区域增多。`膨胀`可以`连接两个分离的区域`，而`腐蚀`可以把两个`粘在一起的物体断开`。

{% codeblock lang:python %}
腐蚀
erosion = cv2.erode(img,kernel,iterations = 1)
膨胀
dilate = cv2.dilate(img,kernel,iterations = 1)
腐蚀之后膨胀
mid = cv2.erode(img,kernel,iterations = 1)
res = cv2.dilate(mid,kernel,iterations = 1)
膨胀之后腐蚀
mid = cv2.dilate(img,kernel,iterations = 1)
res = cv2.erode(mid,kernel,iterations = 1)
{% endcodeblock %}

当然两次形态学变换的时候，核可以进行改变。最后一个是腐蚀次数，可以同时进行好几次腐蚀。

也有开运算和闭运算。开运算用来`除去图像上的小洞`，`删除黑色小斑点`，如下所示：

{% codeblock lang:python %}
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
{% endcodeblock %}

`闭运算`用来`填充前景物体中的小洞`，或者`填充前景物体上的小黑点`：

{% codeblock lang:python %}
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
{% endcodeblock %}

### 图像增强操作

直方图均衡化

{% codeblock lang:python %}
dst = cv2.equalizeHist(gray)
{% endcodeblock %}

比如我得到了一个灰度图像，在0-255的范围内。

我现在想要增强，映射到以最小值为0，最大值为255的区间之内，这个可以使用OpenCV进行如下操作：

{% codeblock lang:python %}
def norm(img):
    min = np.min(np.min(img))
    max = np.max(np.max(img))
    img = 255*(img-min) *(max - min) + min
{% endcodeblock %}

### 找轮廓：

参考[这里](https://blog.csdn.net/hjxu2016/article/details/77833336)

前提操作：二值化之后的图像。
{% codeblock lang:python %}
contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
{% endcodeblock %}

然后把轮廓画出来：
{% codeblock lang:python %}
cv2.drawContours(img,contours,-1,(0,0,255),3)  
{% endcodeblock %}

参数详解：

第一个参数是寻找轮廓的图像；

第二个参数表示轮廓的检索模式，有四种（本文介绍的都是新的cv2接口）：
    `cv2.RETR_EXTERNAL`表示只检测外轮廓
    `cv2.RETR_LIST`检测的轮廓不建立等级关系
    `cv2.RETR_CCOMP`建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
    `cv2.RETR_TREE`建立一个等级树结构的轮廓。

第三个参数method为轮廓的近似办法
    `cv2.CHAIN_APPROX_NONE`存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
    `cv2.CHAIN_APPROX_SIMPLE`压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
    `cv2.CHAIN_APPROX_TC89_L1`，`CV_CHAIN_APPROX_TC89_KCOS`使用teh-Chinl chain 近似算法

返回值：
cv2.findContours()函数返回两个值，一个是轮廓本身，还有一个是每条轮廓对应的属性。

`注意！假设轮廓有100个点，OpenCV返回的ndarray的维数是(100, 1, 2)！！！而不是我们认为的(100, 2)。切记！！`

### 绘制轮廓矩形框

绘制轮廓外矩形框，常用绘制轮廓外形状的函数： 
{% codeblock lang:python %}
cv::boundingRect(InputArray points)绘制一个矩形（轮廓周围最小矩形左上角点和右下角点） 
cv::minAreaRect(InputArray points)绘制轮廓周围最小旋转矩形 
cv::minEnclosingCircle(InputArray points, Point2f& center, float& radius)//绘制轮廓周围最小圆形 
cv::fitEllipse(InputArray points)绘制轮廓周围最小椭圆
{% endcodeblock %}

## 动态调整参数

有时候我们需要动态调整参数，这时候有一个滚动条，开关啥的就很好了。opencv能够建立一个这样的对话框一样的东西。

一般步骤如下：

{% codeblock lang:python %}
# 首先要先建立一个window
# 虽然默认是auto，但是不能够调整大小，一般改成normal
cv2.namedWindow('the_image',cv2.WINDOW_NORMAL)
cv2.createTrackbar(trackbarname, winname, value, count, onChange, userdata)
# value 是当前值
# count 是最大值
# onChange是回调函数，函数的形式是：onChange(int, userdata)，第一个是位置值，第二个是用户的参数
# 也能够在循环当中实时获取这个值
a = cv2.getTrackbarPos(trackbarname, winname)
# 注意如果要在循环里面写的话，要写退出的东西，比如waitkey啥的。
{% endcodeblock %}

### 拟合椭圆


### matplotlib调整图像大小：

{% codeblock lang:python %}
plt.imshow(img,aspect='equal')
fig = plt.gcf()
fig.set_size_inches(10.5, 10.5)
plt.title(title,loc ='left')
{% endcodeblock %}
