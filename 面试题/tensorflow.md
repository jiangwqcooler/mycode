### Tensorflow的工作原理

Tensorflow是用数据流图来进行数值计算的,而数据流图是描述有向图的数值计算过程。在有向图中,节点表示为数学运算,边表示传输多维数据,节点也可以被分配到计算设备上从而并行的执行操作。

### Tensorflow中interactivesession和session的区别
Tf.Interactivesession()默认自己就是用户要操作的会话,而tf.Session()没有这个默认,所以eval()启动计算时需要指明使用的是哪个会话。

