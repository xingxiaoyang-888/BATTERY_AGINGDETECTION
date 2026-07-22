# 钠离子电池服务器训练

## 1. 同步代码

```bash
git clone https://github.com/xingxiaoyang-888/BATTERY_AGINGDETECTION.git
cd BATTERY_AGINGDETECTION
git switch codex/sodium-data-pipeline
python -m pip install -r requirements-data.txt
```

GPU 版 PyTorch 需要按服务器 CUDA 版本单独安装，不写死在通用依赖中。

## 2. 数据目录

```text
data/raw/sodium_ion/
|-- mendeley_nfm/
|   |-- Dataset 1-1.xlsx    # 25 C 钠电
|   |-- Dataset 1-2.xlsx    # -15 C 钠电
|   `-- Dataset 2.xlsx      # 锂电对照，默认不加载
`-- rwth_67_cells/
    `-- <解压后的官方整包>
```

Wenzhou H 系列继续放在仓库原有 Wenzhou 总目录下。所有原始数据均被 Git 忽略。

## 3. 数据传输

Mendeley NFM 只有约 14 MB，可直接同步：

```bash
rsync -avP data/raw/sodium_ion/mendeley_nfm/ USER@SERVER:/path/BATTERY_AGINGDETECTION/data/raw/sodium_ion/mendeley_nfm/
```

Wenzhou 钠电约 450 MB，也适合用 `rsync -avP` 续传。RWTH 整包约 3.44 GB：

1. 服务器能访问 RWTH 时，优先在服务器直接下载。
2. 若命令行触发站点挑战，在本地浏览器下载一次。
3. 使用 `rsync -avP` 传输，便于断点续传和校验。

## 4. 预处理与审计

当前已实测通过 Wenzhou 和 Mendeley NFM：

```bash
python train_soh_model.py --pipeline --datasets wenzhou mendeley-nfm
python -m utils.audit_sodium_data
```

处理产物分别位于：

```text
models/data/processed/sodium_ion/
models/weights/sodium_ion/
```

RWTH 整包尚未在本地落地，字段映射没有实测依据。当前加载器会在发现整包数据时停止并给出错误，必须先对照官方读取脚本核验通道名称、单位、循环聚合和电芯编号，再把 `rwth` 加入训练命令。不要用猜测字段生成论文训练标签。

## 5. 训练原则

先跑树模型基线，再跑深度模型：

```bash
python -m models.soh_ai.train --model xgb --datasets wenzhou mendeley-nfm
python -m models.soh_ai.train --model lstm --epochs 200
```

训练、验证和测试必须按电芯或数据实体划分，不能把同一实体的循环随机拆到多个集合。当前 6 个实体足够完成工程基线，但 Mendeley 两组是温度工况数据，不能在论文中表述为 2 只经过独立重复实验的电芯。论文实验应在 RWTH 67 电芯落地后增加跨电芯、跨温度和跨数据集验证。
