# -
看点视频，看点专业书，随机做点笔记存过来

  
***单变量线性回归***  
线性回归预测值，分类问题预测结果偏向于0或1的二值问题  
回归问题常用模型：平方误差代价函数（假设函数与实际值的均差方）  
线性回归的目标函数：min（代价函数）  
Gradient descent 梯度下降法 **注意对每个参数要同步更新**  
  
线性回归算法：梯度下降法与平方代价函数结合（Batch梯度下降法）  
  
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

***正规方程*** 求0的最优值  不经过迭代，直接找到最优解  
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
