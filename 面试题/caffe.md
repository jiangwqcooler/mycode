### 1. 写solver文件需要注意的点
- 需要导入的包：
```python
from caffe.proto import caffe_pb2
```
- 声明solver结构
```python
s = caffe_pb2.SolverParameter()
```
- 保存solver
```python
with open(slover_file, 'w') as f:
f.wirte(str(s))
``` 

### 参数
- net / {train_net/test_net}：指定定义模型的prototxt文件
- test_iter:这个次数和测试批量大小的乘积应该为测试集的大小
- test_interval:迭代test_interval次进行一次测试
- display:每迭代display次显示一次结果
- max_iter:最大迭代次数
- lr_policy:学习率策略
- caffe框架中的策略包括：fixed、step、exp、inv、multistep、poly、
sigmoid
- gamma:学习率调整过程中的一个参数
- momentum：动量参数
- weight_decay：权重衰减，可以防止过拟合
- snapshot：迭代多少次存储一次模型
- anapshot_prefix:
- slover_mode：GPU/CPU
- device_id：GPU id
- type：学习策略，如梯度下降

### fine_tuning:
差别性的精调（差异学习）：在训练期间为网络设置不同的学习速率。
前几个层通常会包含非常细微的数据细节，比如线和边，我们一般不希望改变这些细节并想保留它的信息

### 启动一个caffe训练需要的文件：
- train_val.prototxt
- solver.prototxt：假设在路径path下
- 运行：/caffe-master/build/tools/caffe --solver=path/solver.prototxt

### caffe能使用几种数据源
数据库：LevelDB、LMDB
数据层为Data
必须配置的参数有：
- source：数据库名称（带路径）
- batch_size：每次处理的数据个数（受内存限制）
可选参数：
- rand_skip：在开始的时候路过某个数据的输入。在异步SGD很有用
- backend：选择是采用LevelDB还是LMDB

