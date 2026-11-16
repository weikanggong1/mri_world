# 配对连续目标的 GLMnet 预测

本模块比较一个基线特征矩阵和多个模态矩阵对同一连续目标的预测。每名受试者占一行；外层五折与内层三折在目标缺失和模态可用性筛选之前共享固定。每个训练折独立进行均值填补、删除全缺失列，并按训练目标的绝对相关性选择最多 200 列。`glmnet==2.2.1` 负责 elastic-net 拟合及其默认的内部 lambda 选择。

```mermaid
flowchart LR
    A[明确的受试者行顺序] --> B[MRI合格域与基线可用域交集]
    B --> C[固定共享 outer5 / inner3 folds]
    C --> D[应用目标缺失与预期模态掩码]
    D --> E[训练折填补与top200 / alpha选择]
    E --> F[每人一次外层OOF预测]
    F --> G[原target单位的模态预测算术均值]
    G --> H[同一配对人群比较基线与模态均值]
```

## 输入和 Python 调用

所有输入矩阵必须使用同一个显式受试者行顺序。`baseline_features` 为 `[N,P]`，可使用 IDP 或其他预先定义的完整基线特征。`feature_blocks` 将模态名映射到 `[N,D_m]` 矩阵；各模态列数可以不同。受试者 ID 是调用者已经确定的 canonical ID，不从 MRI 文件名推断，也不通过 target 表头选择基线列。

```python
import numpy as np
from brain_mri_world_model.paired_glmnet import fit_paired, summary

# 文件由用户在自己的数据环境中准备；所有数组具有相同受试者行顺序。
with np.load("data/prediction_matrices.npz", allow_pickle=False) as matrices:
    subject_ids = matrices["subject_ids"].tolist()
    target = matrices["target"]
    baseline_features = matrices["baseline"]
    modality_names = [key.removeprefix("feature__")
                      for key in matrices.files if key.startswith("feature__")]
    feature_blocks = {name: matrices["feature__" + name] for name in modality_names}
    expected_availability = {name: matrices["expected__" + name] for name in modality_names}
    completed_availability = {name: matrices["completed__" + name] for name in modality_names}
    result = fit_paired(
        subject_ids, target, baseline_features, feature_blocks,
        expected_availability=expected_availability,
        completed_availability=completed_availability,
        scientific_eligible=matrices["scientific_eligible"],
        baseline_available=matrices["baseline_available"],
        baseline_complete=matrices["baseline_complete"],
        minimum_subjects=1001,
    )

print(summary(result))  # 聚合指标和计数
# result还含逐人target、fold和OOF数组，供调用者在私有数据目录保存。
```

| 输入 | 格式与含义 |
|---|---|
| `subject_ids` | 长度 N 的唯一非空字符串序列；跨数据来源的 ID 应先明确消歧 |
| `target` | 浮点 `[N]`；NaN 表示目标缺失，Infinity 报错；不作全局目标变换 |
| `baseline_features` | 数值 `[N,P]`，P>0；NaN 由每个训练折的列均值处理 |
| `feature_blocks` | `{模态名: 数值[N,D_m]}`；`baseline` 为保留名称 |
| `expected_availability` | `{模态名: bool[N]}`；根据独立的合格影像清单确定是否应有该模态，不由已完成特征反推 |
| `completed_availability` | `{模态名: bool[N]}`；特征是否已完成。预期却未完成时该 arm 标为 pending，不静默删人 |
| `scientific_eligible` | 可选 bool[N]，MRI 等预先确定的科学纳入域；默认全 True |
| `baseline_available` | 可选 bool[N]，基线预期可用域；默认全 True。与上项求交后再分 folds |
| `baseline_complete` | 可选 bool[N]，基线实际完成掩码；默认全 True |
| `outer_fold_ids` | 可选整数[N]；必须与本模块在完整 CV 域上生成的 shuffle42 五折完全相同；域外为 -1 |
| `minimum_subjects` | 每个 arm 的最小有效目标人数，默认 1001。可显式调整，分析报告应记录所用阈值 |

如果用户将某个可用/完成掩码省略为默认全 True，就表示用户已确认这些条件对所有输入行成立。程序不会用“当前已经提取了哪些特征”代替科学队列的定义。

## 数值协议与结果

- 外层 5 折、内层 3 折，均使用唯一受试者首次出现顺序与 NumPy `RandomState(42)` shuffle。所有 arms 共享这些 folds；目标缺失和模态差异不重新划分 folds。
- 每个内/外训练折独立填补 predictor NaN 并删除训练中全缺失列；按训练目标的绝对 Pearson 相关性选最多 200 列。相同相关性按原列顺序打破平局。
- alpha 网格固定为 `[0.01,0.1,0.5,0.8,0.99]`，用三次内层验证 Pearson 的均值选择；有限分数完全相同时选择网格中先出现的值。
- 每次拟合调用 `ElasticNet(alpha=alpha, n_lambda=30)`，保留 2.2.1 wrapper 的内部 lambda CV、默认 predict 和预处理默认设置。模块保存实际 lambda/default 参数摘要。
- 不增加 confounds、不残差化目标或 predictors，也不作全局 target 变换。每次拟合的 NumPy RNG 状态在调用后恢复。
- 每名可评估受试者获得一次真正的外层 OOF 预测。模态融合为其所有**预期**模态的原 target 单位 OOF 预测的算术均值；不平均各模态的相关系数，不对 OOF 分数作 z-score。
- 只有固定目标域中的所有预期模态与基线均完成时，`paired_comparison_complete=True` 并给出同一人群的配对 Pearson。预期缺失计算保留 pending；常量目标等未定义 Pearson 保留未评估状态，不替换为 0。

`shared_subject_folds(subject_ids, cv_scope)` 返回长度 N 的外层标签和五组长度 N 的内层标签；域外/相应外层测试行为 -1。`mean_raw_oof(predictions, expected, common_mask)` 接收各模态长度 N 的预测、预期掩码和共同目标掩码，返回原单位融合向量、完整性掩码和每人的预期模态数。

返回字典包含 `outer_fold_ids`、`shared_inner_fold_ids`、`predictions_by_arm`、`feature_mean_predictions`、`paired_mask`、各 arm 状态和计数。`summary(result)` 去除逐人数组及每折详细字段，保留聚合状态、人数和指标。

MRI 三轴和重复扫描应在输入本模块之前按预先定义规则合并。与当前表示协议一致的规则为：同一明确 MRI 的三轴向量先作 CPU FP32 均值；同受试者/同模态多个明确扫描再作 float64 累加的等权扫描均值。空间/模态/图像身份应先验证，不能将不同模态或不同 checkpoint 的向量合并为一次扫描。

## 命令行、环境与原实现调用

在独立 GLMnet 环境运行，主 Torch 环境只负责图像/latent/feature 数值生产。该拟合模块不导入 Torch。实际算法要求 `glmnet==2.2.1`；其余依赖为 NumPy、SciPy、scikit-learn 等 wrapper 所需包，应使用项目发布的独立环境文件。主 Torch 环境的 NumPy 约束不能直接替代旧 GLMnet 环境。已核对的成功拟合环境使用 NumPy 1.26.4、SciPy 1.11.4、scikit-learn 1.7.1 和 GLMnet 2.2.1（Conda 构建 `py311h4397e4c_6`）；应在独立环境中保留这些版本，而非在主 Torch 环境中直接降级 NumPy。

NPZ 文件键：必需 `subject_ids`、`target`、`baseline`，每个模态配套 `feature__<name>`、`expected__<name>`、`completed__<name>`；可选 `scientific_eligible`、`baseline_available`、`baseline_complete`、`outer_fold_ids`。ID 使用 Unicode 字符串数组，掩码使用 bool，加载禁用 pickle。

```bash
python -m brain_mri_world_model.paired_glmnet \
  --input data/prediction_matrices.npz \
  --output-dir outputs/paired_prediction \
  --minimum-subjects 1001
```

输出为新目录中的 `summary.json` 和 `private_oof.npz`。后者包含逐人 ID、target、fold 与 OOF，应保留在用户的数据环境中。

对应原 `python-glmnet` 的模型调用：

```python
from glmnet import ElasticNet

# training_predictors / training_target 已在该训练折内完成填补与列筛选。
regressor = ElasticNet(alpha=selected_alpha, n_lambda=30)
regressor.fit(training_predictors, training_target)
heldout_prediction = regressor.predict(heldout_predictors)
```

这里使用 Python wrapper 的真实 elastic-net 后端。R 的 `cv.glmnet` 有独立的 wrapper 默认值和折分配，不能仅凭同名 alpha/lambda 将其输出视作本接口的逐值等价结果。

## 软件测试、benchmark 与更新记录

`tests/test_paired_glmnet.py` 检查共享 folds 的先后顺序、训练折缺失值处理、确定性 top200 排序、预期模态完整性、原单位 OOF 均值及默认后端版本门槛。小型 fixtures 只用于软件测试，不作为真实连续表型 benchmark。

本次公共导出未附带目标表、逐人 OOF、真实脑图或真实数据端到端精度/运行时间表。可合法复现时，应在完全相同的个体、特征、目标、folds 和期望模态域上，对比原 wrapper 调用与本模块的每折预测/alpha/lambda，另报告矩阵准备、逐 arm 拟合和融合耗时。当前不提供未执行的数字或由软件 fixtures 推导的科学效果。

首次公共版本保留了共享 outer5/inner3、seed42、训练折 top200、原 wrapper lambda 行为和原单位 OOF 融合；输入改为通用矩阵/掩码接口，移除了私有数据布局、固定队列人数、具体表型字段和服务器控制协议。

参考：[python-glmnet 官方实现](https://github.com/civisanalytics/python-glmnet)、[glmnet 2.2.1 发布页](https://pypi.org/project/glmnet/2.2.1/)、Friedman, Hastie & Tibshirani, *Regularization Paths for Generalized Linear Models via Coordinate Descent*, JSS 33(1), 2010, [DOI](https://doi.org/10.18637/jss.v033.i01)。
