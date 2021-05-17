# 环境

- sys.platform：Ubuntu20.04
- python：3.7.9
- Pytorch：1.5.0+cu101
- Torchvision：0.6.0+cu101
- GCC：7.5.0
- MMCV：1.1.5
- MMDetection：2.5.0
- MMDetection3D：0.6.1

# 环境安装及配置

参照MMDetection3D[官方安装文档](https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation)

# 运行方式(Test阶段)

以下步骤只支持基于sunrgbd数据集的votenet模型。

首先需将`lib`添加到`PYTHONPATH`    

```shell
export PYTHONPATH=$PYTHONPATH:/path/to/point-3d/lib
```
  

之后编译相关文件:  
```shell
./compile.sh
```


1. 不保存结果，仅评估mAP

   ```shell
   python lib/core/test.py configs/votenet/manhattan_votenet_2x32_sunrgbd-3d-10class.py checkpoints/manhattan_votenet_0.05_0.05_0.90_epoch34.pth ./data --eval mAP
   ```

2.  保存预测结果并评估mAP

   ```shell
   python lib/core/test.py configs/votenet/manhattan_votenet_2x32_sunrgbd-3d-10class.py checkpoints/manhattan_votenet_0.05_0.05_0.90_epoch34.pth ./data --eval mAP --options 'show=True' 'out_dir=./data/sunrgbd/show_results'
   ```

3. 测试inference time：  

   ```shell
   python lib/core/benchmark.py configs/votenet/manhattan_votenet_2x32_sunrgbd-3d-10class.py checkpoints/manhattan_votenet_0.05_0.05_0.90_epoch34.pth ./data
   ```

