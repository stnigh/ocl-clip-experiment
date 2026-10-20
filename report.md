# OCL 对象概念学习推理实验报告

## 一. 具体实现

### 1.1 实验环境

| 项目 | 配置 |
|---|---|
| Python | 3.8.20 |
| PyTorch | 1.8.1+cu111 |
| torchvision | 0.9.1+cu111 |
| CUDA | 11.1 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8GB |
| CLIP | OpenAI CLIP 1.0，ViT-B/16 |


### 1.2 数据处理

实验使用 OCL 的 val 和 test 标注，合并为 `valtest`。根据对象类别样本数筛选出数量最多的 10 类：

```text
car, broccoli, boat, horse, dog,
banana, motorcycle, truck, zebra, sheep
```

筛选后共有 10,442 张图片和 22,526 个 Top-10 对象。由于部分网络图片的原始链接已经失效，使用 `from_web.tgz` 和 Wayback Machine 补充后，最终可用：

| 项目 | 数量 |
|---|---:|
| 可用图片 | 10,138 |
| 排除图片 | 304 |
| 可用对象 | 22,074 |
| 对象覆盖率 | 97.99% |

排除的 304 张图片包括 299 张缺失图片和 5 张小于 32×32 的异常图片，不参与指标计算。

`src/dataset.py` 中的 `OCLDataset` 完成以下处理：

1. 读取 val/test 标注并统计 Top-10 类别。
2. 只读取可用图片清单中的样本。
3. 将图片转换为 RGB。
4. 根据标注中的对象框裁剪每个 Top-10 对象。
5. 将对象裁剪图调整为 CLIP 的 `224×224` 输入。
6. 返回类别编号、114 维属性多热向量、170 维可供性多热向量和对象框。

使用 PyTorch `DataLoader`，设置 `batch_size=4`、`shuffle=False`、`num_workers=0`，并通过自定义 `collate_fn` 合并一个 batch 中的所有对象裁剪图。

### 1.3 模型调用

测试文档中，Step 2 要求使用 available API 加载预训练 CLIP，并引用 OpenAI CLIP 官方仓库。我没有找打另外提供的接口地址，因此实验中使用的是 OpenAI CLIP 官方 Python API：

```python
model, preprocess = clip.load(
    "ViT-B/16",
    device=device,
    download_root=download_root,
)
```

模型加载后设置：

```python
model.eval()
for parameter in model.parameters():
    parameter.requires_grad = False
```

依据测验，模型参数全部冻结，不定义优化器和损失函数。

### 1.4 推理流程

类别文本使用多个 prompt 模板：

```text
a photo of a {class}.
an image of a {class}.
a picture of a {class}.
```

属性使用正负 prompt：

```text
an object that is {attribute}.
an object that is not {attribute}.
```

可供性使用正负 prompt：

```text
an object that can {affordance}.
an object that cannot {affordance}.
```

同一标签的多个 prompt 分别编码后取平均，再进行 L2 归一化。对象裁剪图通过 CLIP 图像编码器得到图像特征，并与文本特征计算余弦相似度：

```python
class_scores = image_features @ class_text_features.T
binary_score = positive_similarity - negative_similarity
```

类别分类时，对 10 个类别分数排序并取 Top-k。属性与可供性对每个标签独立计算正负分数差。

### 1.5 评估指标

对象类别分类使用对象级 Top-k accuracy：

```text
Top-k = 真实类别出现在前 k 个预测中的对象数 / 总对象数
```

属性和可供性使用逐标签 AP：

```python
AP[label] = average_precision_score(ground_truth, prediction_score)
mAP = np.nanmean(AP) * 100
```

没有正样本的标签不参与 mAP 平均。最终属性使用 95 个有效标签，可供性使用 164 个有效标签。

## 二. 实验结果

### 2.1 类别分类
按指标：

| 指标 | 正确对象 | 总对象 | 准确率 |
|---|---:|---:|---:|
| Top-1 | 19,845 | 22,074 | 89.90% |
| Top-3 | 21,666 | 22,074 | 98.15% |
| Top-5 | 21,908 | 22,074 | 99.25% |

按图片来源：

| 来源 | 对象数 | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|---:|
| COCO | 19,003 | 89.78% | 98.31% | 99.31% |
| ImageNet | 1,016 | 81.40% | 93.90% | 97.64% |
| from_web | 2,055 | 95.28% | 98.83% | 99.51% |

### 2.2 属性与可供性

| 任务 | 有效标签数 | 全部标签数 | mAP |
|---|---:|---:|---:|
| 属性 | 95 | 114 | 11.97% |
| 可供性 | 164 | 170 | 28.61% |

属性 mAP 较低，主要因为属性更细粒度，且当前使用固定正负文本 prompt 进行 zero-shot 打分。
而可供性类别相对集中，因此 mAP 较高。

### 2.3 结果分析

Top-3 和 Top-5 分别达到 98.15% 和 99.25%，说明正确类别通常能在候选排序中进入前列。同时，Top-1 的主要错误集中在相似类别：

| 真实类别 | 预测类别 | 次数 |
|---|---|---:|
| car | truck | 375 |
| truck | car | 274 |
| boat | car | 110 |
| motorcycle | car | 91 |
| car | motorcycle | 91 |
| boat | truck | 87 |
| dog | sheep | 84 |
| sheep | dog | 82 |

与论文 Table 2 相比，本实验属性 mAP 为 11.97，可供性 mAP 为 28.61，而论文 Vanilla CLIP 分别为 23.6 和 49.6，OCRN 分别为 31.6 和 53.3。分析可知，这是因为论文中的 Vanilla CLIP 是在 OCL 上训练的，而本实验使用完全冻结的预训练 CLIP 并只保留 Top-10 子集。但是本实验和论文的数据都表明可供性 mAP 高于属性 mAP。

## 三. 遇到的问题

### 问题一：环境配置

实验要求的 Python、PyTorch、torchvision、NumPy 和 Pillow 版本与原有环境不一致。
另外，第一次安装 CLIP 时调用了默认 Python 3.10 的 `pip`，包被装到了错误环境，
而实验实际使用的 `carprt` 中仍然没有 CLIP。

**解决方案：** 更新实验环境中的相关版本；将 Pillow 改为官方 Windows wheel；
从错误环境中卸载 CLIP 相关包，使用 `carprt` 的完整解释器路径重新安装 CLIP，
并添加 `--no-deps`，避免 pip 改动已经固定好的 torch。

### 问题二：数据丢失与脚本适配

OCL 的部分网络图片需要根据原始 URL 重新下载。实际准备数据时发现大量链接已经
失效，返回 404、410、403 或 502。`from_web.tgz` 只包含数据集发布前已经失效的
图片，无法补齐之后继续失效的链接。

对剩余失败 URL 使用 Wayback Machine 查询后，又补回了 36 张历史图片，但最终
仍有大量图片无法获取：

```text
缺失图片：299 张
尺寸异常：5 张
不可用合计：304 张
最终可用：10,138 / 10,442 张
对象覆盖：97.99%
```

数据丢失还带来了脚本适配问题： `prepare_data.sh` 面向完整数据准备，
会下载 SUN、aPY 和完整 COCO，且依赖 Linux 工具；并且
`download_web_images.py` 遇到第一个失败链接便终止，无法处理当前大量失效 URL。
如果继续使用完整标注和完整文件清单，DataLoader 会在读取缺失图片时直接报错。

**解决方案：** 保留官方数据源和目录结构，不修改原始数据内容，在
`src/` 中增加测试子集适配脚本：

1. `prepare_coco_subset.py`：只从 COCO 压缩包提取所需的 7,896 张图片，避免解压完整数据。
2. `prepare_from_web.py`：在官方 URL 映射基础上增加跳过已有文件、超时重试、失败记录和图片有效性检查，避免单张失败终止全部任务。
3. `query_wayback_from_web.py`、`query_wayback_curl.py`：查询失效 URL 的历史快照，并生成补充下载清单。
4. `verify_dataset.py`：对照标注、可用清单和实际文件，检查缺失、损坏和尺寸异常。
5. `dataset.py`：改为读取 Top-10 可用清单，自动排除缺失和异常图片，避免 DataLoader 读取不存在的文件。
6. ImageNet 按清单从完整训练包和验证包中提取所需图片，没有保留完整 ImageNet 数据集。

最终评测基于 10,138 张有效图片和 22,074 个可用对象。原始网络图片无法全部
恢复是当前唯一未完全解决的数据问题。

