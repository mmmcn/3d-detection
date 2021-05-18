# 模型表现
基于SUNRGBD数据集  

|method | AP@0.25 | AP@0.5 |
| :----: | :----: | :----: |
| votenet-official | 57.7 |  |
| votenet-mmdet3d | 59.07 | 35.77 |
| votenet-manhattan | 60.45 | 37.03 | 
# 环境

- sys.platform：Ubuntu20.04
- python：3.7.9
- Pytorch：1.5.0+cu101
- Torchvision：0.6.0+cu101
- GCC：7.5.0
- MMCV：1.1.5
- MMDetection：2.5.0

# 环境安装及配置
a. 创建conda虚拟环境并激活  
```shell
conda create -n votenet python=3.7
conda activate votenet
```
  
b. 根据[Pytorch官方](https://pytorch.org/get-started/locally/)安装对应版本的torch及torchvision  
```shell
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1
```
  
c. 安装[MMCV](https://github.com/open-mmlab/mmcv)  
```shell
 pip install mmcv-full==1.1.5 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.5.0/index.html
```
  
d. 安装[MMDetection](https://github.com/open-mmlab/mmdetection)  
```shell
git clone -b v2.5.0 https://github.com/open-mmlab/mmdetection.git
cd mmdetection-2.5.0
pip install -r requirements/build.txt
pip install -v -e . # or “python setup.py develop”
```
  
e. 将本仓库克隆到本地  
```shell
git clone https://github.com/mmmcn/votenet-manhattan.git
cd 3d-detection
```
  
f. 编译相关文件  
```shell
./compile.sh
```
  
g. 下载相关依赖(with `pip`)  
**如果只进行前向推理以及只使用提供的数据,可以不安装以下依赖包**
```tex
tensorboard
scipy
```

# 运行方式
首先将`lib`及`mmdet3d`添加到`PYTHONPATH`  
```shell
export PYTHONPATH=$PYTHONPATH:/path/to/3d-detection/lib:/path/to/3d-detection/mmdet3d
```
## 数据准备
`./data`目录中数据仅支持使用预训练模型infer及测试模型推断速度  
如果想要训练自己的模型  
1. 请参考[votenet](https://github.com/facebookresearch/votenet/tree/master/sunrgbd)中步骤1和步骤2
2. 执行以下命令生成训练数据:
  ```shell
  python lib/utils/create_data/create_data.py sunrgbd --root-path ./data/sunrgbd --out-dir ./data/sunrgbd --extra-tag sunrgbd
  ```

## Train阶段 
如果想要训练自己的模型还需要安装`tensorboard`  
```shell
pip install tensorboard
```
之后执行以下命令可进行模型训练:  
```shell
python lib/core/train_v2.py configs/votenet/manhattan_votenet_2x32_sunrgbd-3d-10class.py --work-dir ${YOUR_WORK_DIR}
```  

## Test阶段
**如果仅使用部分数据集, 则需要指定`data_root`参数**
1. 不保存结果,仅评估mAP
```shell
python lib/core/test.py configs/votenet/manhattan_votenet_2x32_sunrgbd-3d-10class.py checkpoints/manhattan_votenet_0.05_0.05_0.90_epoch34.pth --data_root ./data --eval mAP
```
2. 保存预测结果并评估mAP
```shell
python lib/core/test.py configs/votenet/manhattan_votenet_2x32_sunrgbd-3d-10class.py checkpoints/manhattan_votenet_0.05_0.05_0.90_epoch34.pth --data_root ./data --eval mAP --options 'show=True' 'out_dir=./data/sunrgbd/show_results'
```
**保存结果目前存在bug,待修复**
  
## 测试Infer速度
```shell
python lib/core/benchmark_v2.py configs/votenet/manhattan_votenet_2x32_sunrgbd-3d-10class.py checkpoints/manhattan_votenet_0.05_0.05_0.90_epoch34.pth --data_root ./data
```
