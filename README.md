# 
看点视频，看点专业书，随机做点笔记存过来  
  
    
***单变量线性回归***  
线性回归预测值，分类问题预测结果偏向于0或1的二值问题  
回归问题常用模型：平方误差代价函数（假设函数与实际值的均差方）  
线性回归的目标函数：min（代价函数）  
Gradient descent 梯度下降法 **注意对每个参数要同步更新**  
  
线性回归算法：梯度下降法与平方代价函数结合 （Batch梯度下降法）  
  
在单变量线性回归中，假设函数的参数有两个，a0和a1，其实质就是那一条可见的直线  
代价函数是一个针对m个点而言的误差平方和再求平均，计算量展开式庞大，表达式是个求和式  
而梯度下降函数是在前值和学习率（步长）一定时，对代价函数对应求偏导，整个更新过程是并行的，  
    也就是说整个过程代价函数的参数都是未更新之前的参数  
       
 求导式因为参数只有a0 a1，当对a0求导时，平方项直接拉下来，但是内部还是累加和  
                        当对a1求导时，a1有参数x(i),故求导式不仅是累加和，还有乘积项参数x(i)  
           
           
***多变量线性回归***  
单变量假设函数：h(x) = a0 + a1x;  
多变量假设函数：h(x) = a0 + a1x + a2x +...+ anx;  
在多变量线性回归中，对于参数的更新，单变量中带有参数的a1项为一般式aj的映射式  

梯度下降法实用技巧  
**1.特征缩放**(控制影响因素差不多，从而使得梯度下降速度更快)  一般用均值归一化方法，(xi - ave) / （max - min）  
**2.学习率的选择** 绘制J（0）与迭代次数的函数，判断是否是学习率选择过大  经验：3倍扩大  

特征和多项式回归  
可以对已有特征处理，作为新的特征来计算 多项式回归是对假设函数的更改，可能是二次函数，三次函数，也可能是开方函数  

***正规方程***p69过程推导  求0的最优值  不经过迭代，直接找到最优解  
对于x列向量，多加一列x0全为1，然后构造X矩阵，已知y矩阵，0矩阵最优解公式为 0 = （X转置·X）逆 ·X转置y  
**对于（X转置·X）可能涉及到的不可逆的问题，先看是否有多余特征，比如倍数关系，有的话就删除，或是再用伪逆解决**  
  
数据较小时用正规方程比较方便 数据较大时用梯度下降法  分界在10000左右  
  
    
***Logisitic回归***  个人觉得这个名字主要是因为log函数  
线性回归不适用于分类问题，适用于预测值  
Logisitic函数也叫sigmoid函数 g(z) = 1/(1 + e^(-z))    
**决策边界**原函数中映射到值0.5的所有点组成的线，其将平面划分成为两个平面，分别映射到0/1  

用训练集的目的是拟合参数0，而不是用训练集确定决策边界    
对于代价函数中的假设函数使用sigmoid函数会陷入过多的局部最优解，而无法形成一个凹形函数，  
故代价函数差方和被Cost(h0(x),y)代替， 含义为：在输出的预测值是h(x)而实际的标签是y的情况下，我们希望算法付出的代价  
if(y = 1) Cost 为 -log(h0(x))  图像横轴h0(x)为0时，纵轴Cost为正无穷，意思是我们预测值与实际值相反，付出巨大的代价 而h0(x)为1时，Cost为1  
if(y = 0) Cost 为 -log(1-h0(x))  图像横轴h0(x)为0时，纵轴Cost为0，意思是我们预测值与实际值完全相同，代价为零， 而h0(x)为1时，Cost为正无穷  
合并后Cost 代价函数为：-ylog(h0(x))-(1-y)log(1-h0(x))
找出让J(0)取得最小值的参数0  梯度下降法迭代  
一些其他的算法 复杂度更高，例如共轭梯度法，BFGS，L-BFGS  
"一对多"分类算法  多类别分类问题 对于分为n类问题，依次将某一类化为正类，剩下n-1类为负类，训练出一个识别各种类的概率，然后将测试集例子带入各个分类器，取得最大概率分类器即为预测y值  


***正则化***  
过度拟合会使算法性能不佳，解决方法就是正则化  
算法没有很好的拟合训练集，比如后期差距越来越大(高偏差)，为欠拟合  
算法具有高方差，高度拟合每一个训练样本但是图像本身不够圆滑，可以说是崎岖不平(高阶函数参数过多，什么函数都能拟合，代价函数值几近于0，图像就是丑)，就是过度拟合  
防止过度拟合两个方法：  
1.尽量减少选取变量的数量  
2.正则化  保留所有变量，但是减少量级  
正则化的实质是代价函数的更改  在代价函数中高阶参数给个大的权值，要使代价函数最小，对应参数取值就小，回到假设函数中的值就小，影响就小  
例如假设函数 y = 0o + 01x + 02x^2 + 03x^3 + 04x^4; 代价函数 J(0) = 差方和 + 555503^2 + 666604^2;  想代价函数尽量小，03和04就尽量小。（称为加入惩罚）  
推广出去，对于所有参数(1-n)设立惩罚项 即修改代价函数 J(0) = 差方和 + t(求和1-n)0^2  t就是正则化参数 作用是控制两个不同目标之间的取舍 目标1是使拟合尽量的好，目标2是使参数尽量小    正则化参数过大，使得01及以后都很小，即只有0o起作用，典型的过拟合，所有选择正则化参数就显得尤为重要  


线性回归的正则化
根据多变量线性回归和正规方程，均有一些细节上的不同  
多变量线性回归的梯度下降法，0o不变 0j 在最后加上(正则化参数/m)x0j 化简后可看到每次0j与之前的线性回归梯度下降相比只是减少了一丢丢，学习率后面的一大块没变
而对于正规方程，在x^Tx同级加了一个正则化参数x类似单位矩阵,只是左上角为0，主对角线其他都为1  通常是(n+1维)x(n+1维)  这种情况下，只要正则化参数>0,一定可逆  

Logistic 回归的正则化  
在Cost函数 (未求和时)-ylog(h0(x))-(1-y)log(1-h0(x)) 后面添加 正则化参数/2m x (求和1-n)0j^2  再运用梯度下降法，与多变量线性回归的式子类似 但是假设函数前者是sigmoid函数，后者是log型那个 
  
  
  
  ***神经网络***  
    
**非线性假设**  
特征n的个数太多时，用线性分类，就会有很多的项数，计算量极其庞大   
  
  
  
  
  
  
    
     
     
     
***TensorFlow入门教程相关***  
TensorFlow的数据中央控制单元是tensor(张量)，一个tensor由一系列的原始值组成，这些值被形成一个任意维数的数组。  
一个tensor的列就是它的维度。  
import tensorflow as tf   TensorFlow 程序典型的导入语句，作用是：赋予Python访问TensorFlow类(classes)，方法（methods），符号(symbols)  

TensorFlow核心程序由2个独立部分组成： a:Building the computational graph构建计算图    b:Running the computational graph运行计算图    
一个computational graph(计算图)是一系列的TensorFlow操作排列成一个节点图 。  
node1 = tf.constant(3.0, dtype=tf.float32)  
node2 = tf.constant(4.0)# also tf.float32 implicitly  
print(node1, node2)  
最后打印结果：Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0",shape=(), dtype=float32)  
要想打印最终结果，我们必须用到session:一个session封装了TensorFlow运行时的控制和状态  
sess = tf.Session()  
print(sess.run([node1, node2]))  
我们可以组合Tensor节点操作(操作仍然是一个节点)来构造更加复杂的计算  
node3 = tf.add(node1, node2)  
print("node3:", node3)  
print("sess.run(node3):", sess.run(node3))  
打印结果是：
node3:Tensor("Add:0", shape=(), dtype=float32)  
sess.run(node3):7.0  

用placeholders(占位符)来预先定义公式方法，后面再调用sess.run()函数去填充公式，计算得值  
a = tf.placeholder(tf.float32)  
b = tf.placeholder(tf.float32)  
adder_node = a + b      # + provides a shortcut for tf.add(a, b)  
print(sess.run(adder_node, {a:3, b:4.5}))  
print(sess.run(adder_node, {a: [1,3], b: [2,4]}))  
结果是：  
7.5  
[3.  7.]  
操作的可迭代性，意为定义过的方法可作为新的成员变量在新的方法中被使用  
    add_and_triple = adder_node *3.  
print(sess.run(add_and_triple, {a:3, b:4.5}))  
输出结果是：  
22.5  

在机器学习中，我们通常想让一个模型可以接收任意多个输入，比如大于1个，好让这个模型可以被训练，在不改变输入的情况下,  
我们需要改变这个计算图去获得一个新的输出。变量允许我们增加可训练的参数到这个计算图中，它们被构造成有一个类型和初始值：  
    W = tf.Variable([.3], dtype=tf.float32)  
b = tf.Variable([-.3], dtype=tf.float32)  
x = tf.placeholder(tf.float32)  
linear_model = W*x + b  
**当你调用tf.constant时常量被初始化，它们的值是不可以改变的，而变量当你调用tf.Variable时没有被初始化**  
在TensorFlow程序中要想初始化这些变量，你必须明确调用一个特定的操作，如下：  
    init = tf.global_variables_initializer()  
sess.run(init)  
要实现初始化所有全局变量的TensorFlow子图的的处理是很重要的，直到我们调用sess.run，这些变量都是未被初始化的。  
既然x是一个占位符，我们就可以同时地对多个x的值进行求值linear_model，例如：  
    print(sess.run(linear_model, {x: [1,2,3,4]}))  
求值linear_model   
输出为  
[0.  0.30000001  0.60000002  0.90000004]  

我们已经创建了一个模型，但是我们至今不知道它是多好，在这些训练数据上对这个模型进行评估，我们需要一个  
y占位符来提供一个期望的值，并且我们需要写一个loss function(损失函数)，一个损失函数度量当前的模型和提供  
的数据有多远，我们将会使用一个标准的损失模式来线性回归，它的增量平方和就是当前模型与提供的数据之间的损失，  
linear_model - y创建一个向量，其中每个元素都是对应的示例错误增量。这个错误的方差我们称为tf.square。然后，  
我们合计所有的错误方差用以创建一个标量，用tf.reduce_sum抽象出所有示例的错误。 
    y = tf.placeholder(tf.float32)  
squared_deltas = tf.square(linear_model - y)  
loss = tf.reduce_sum(squared_deltas)  
print(sess.run(loss, {x: [1,2,3,4], y: [0, -1, -2, -3]}))  
输出的结果为  
23.66  
我们分配一个值给W和b(得到一个完美的值是-1和1)来手动改进这一点,一个变量被初始化一个值会调用tf.Variable，  
但是可以用tf.assign来改变这个值，例如：fixW = tf.assign(W, [-1.])  
    fixb = tf.assign(b, [1.])  
sess.run([fixW, fixb])  
print(sess.run(loss, {x: [1,2,3,4], y: [0, -1, -2, -3]}))  
最终打印的结果是：  
0.0  

tf.train APITessorFlow提供optimizers(优化器)，它能慢慢改变每一个变量以最小化损失函数，最简单的优化器是  
gradient descent(梯度下降)，它根据变量派生出损失的大小,来修改每个变量。通常手工计算变量符号是乏味且容易出错的，  
因此，TensorFlow使用函数tf.gradients给这个模型一个描述，从而能自动地提供衍生品，简而言之，优化器通常会为你做这个。例如：  
    optimizer = tf.train.GradientDescentOptimizer(0.01)  
train = optimizer.minimize(loss)  
sess.run(init)# reset values to incorrect defaults.  
for iin range(1000):  
   sess.run(train, {x: [1,2,3,4], y: [0, -1, -2, -3]})  
 
print(sess.run([W, b]))  
输出结果为  
[array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)]  

---------------------  

本文来自 lgx06 的CSDN 博客 ，全文地址请点击：https://blog.csdn.net/lengguoxing/article/details/
