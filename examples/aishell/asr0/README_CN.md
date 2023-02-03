# DeepSpeech2 使用 Aishell 离线/在线 ASR
此示例包含用于使用 [Aishell 数据集](http://www.openslr.org/resources/33) 训练 DeepSpeech2 离线或在线模型的代码

## 概述
您需要的所有脚本都在 `run.sh` 中。 `run.sh`中有几个阶段，每个阶段都有它的功能。

| 阶段 | 功能                                                                                                                  |
|:---- |:--------------------------------------------------------------------------------------------------------------------|
| 0     | 处理数据. 这包括: <br>       (1) 下载数据集 <br>       (2) 计算训练数据集的CMVN <br>       (3) 获取词汇文件 <br>       (4) 获取训练、开发和测试数据集的清单文件 |
| 1     | 训练模型                                                                                                                |
| 2     | 通过对 top-k 模型求平均得到最终模型，设 k = 1 表示选择最好的模型                                                                             |
| 3     | 测试最终模型性能                                                                                                            |
| 4     | 导出静态图模型                                                                                                             |
| 5     | 测试静态图模型                                                                                                             |
| 6     | 推断单个音频文件                                                                                                            |

您可以通过设置 `stage` 和 `stop_stage` 来选择运行一系列阶段.

比如你要执行阶段 2 和阶段 3 的代码，可以运行这个脚本:
```bash
bash run.sh --stage 2 --stop_stage 3
```
或者你可以设置 `stage` 等于 `stop-stage` 只运行一个阶段.

例如，如果你只想运行阶段 0，你可以使用下面的脚本:
```bash
bash run.sh --stage 0 --stop_stage 0
```
下面的文档将详细描述 `run.sh` 中的脚本.

## 环境变量
path.sh 包含环境变量. 
```bash
source path.sh
```
这个脚本需要先运行。  

还需要另一个脚本:
```bash
source ${MAIN_ROOT}/utils/parse_options.sh
```
它将支持在 shell 脚本中使用 `--variable value` 的方式设置参数.

## 局部变量
一些局部变量设置在 `run.sh`. 

- `gpus` 表示您要使用的 GPU 编号. 如果你设置 `gpus=`，这意味着你只使用 CPU. 
- `stage` 表示您要在实验中开始的阶段数.
- `stop stage` 表示您希望在实验中结束的阶段数. 
- `conf_path` 表示模型的配置路径.
- `avg_num` 表示要平均得到最终模型的 top-K 模型的数量 K.
- `model_type` 表示模型类型: offline 或 online
- `audio file` 表示要在阶段 6 中推断的单个文件的文件路径
- `ckpt` 表示模型的检查点前缀, 例如 "deepspeech2"

您可以在使用 `run.sh` 时设置局部变量（`ckpt` 除外）

例如，您可以在使用命令行时设置 `gpus` 和 `avg_num`:
```bash
bash run.sh --gpus 0,1 --avg_num 1
```
## 阶段 0: 数据处理

要使用此示例，您需要先处理数据，您可以使用 `run.sh` 中的阶段 0 来执行此操作。代码如下所示:

```bash
 if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
     # 准备资料
     bash ./local/data.sh || exit -1
 fi
```
阶段 0 用于处理数据.

如果你只想处理数据。你可以跑
```bash
bash run.sh --stage 0 --stop_stage 0
```
您也可以在命令行中运行这些脚本.
```bash
source path.sh
bash ./local/data.sh
```
处理数据后，`data` 目录将如下所示:
```bash
data/
|-- dev.meta
|-- lang_char
|   `-- vocab.txt
|-- manifest.dev
|-- manifest.dev.raw
|-- manifest.test
|-- manifest.test.raw
|-- manifest.train
|-- manifest.train.raw
|-- mean_std.json
|-- test.meta
`-- train.meta
```
## 阶段 1: 模型训练
如果你想训练模型。您可以在 `run.sh` 中使用阶段 1。代码如下所示. 
```bash
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
     # 训练模型，所有 `ckpt` 都在 `exp` 目录下
     CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path}  ${ckpt}
 fi
```
如果要训练模型，可以使用下面的脚本执行阶段 0 和阶段 1:
```bash
bash run.sh --stage 0 --stop_stage 1
```
或者您可以在命令行中运行这些脚本（仅使用 CPU）.
```bash
source path.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/deepspeech2.yaml deepspeech2
```
## 阶段 2:  Top-k 模型平均
训练完模型后，我们需要得到最终的模型进行测试和推理。在每个 epoch 中，模型检查点都被保存下来，因此我们可以根据验证损失从中选择最好的模型，或者我们可以对它们进行排序并对 top-k 模型的参数进行平均以获得最终模型。我们可以使用阶段 2 来做到这一点，代码如下所示:
```bash
 if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
     # 平均最佳模型
     avg.sh best exp/${ckpt}/checkpoints ${avg_num}
 fi
```
`avg.sh` 位于 `../../../utils/`, 它在 `path.sh` 中定义.

如果想得到最终的模型，可以使用下面的脚本分别执行阶段 0、阶段 1、阶段 2:
```bash
bash run.sh --stage 0 --stop_stage 2
```
或者您可以在命令行中运行这些脚本（仅使用 CPU）.

```bash
source path.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/deepspeech2.yaml deepspeech2
avg.sh best exp/deepspeech2/checkpoints 1
```
## 阶段 3: 模型测试
测试阶段是评估模型的性能。测试阶段的代码如下所示:
```bash
 if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
     # 测试 ckpt avg_n
     CUDA_VISIBLE_DEVICES=0 ./local/test.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
 fi
```
如果你想训练一个模型并测试它，你可以使用下面的脚本来执行 stage 0、stage 1、stage 2 和 stage 3:
```bash
bash run.sh --stage 0 --stop_stage 3
```
或者您可以在命令行中运行这些脚本（仅使用 CPU）.
```bash
source path.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/deepspeech2.yaml deepspeech2
avg.sh best exp/deepspeech2/checkpoints 1
CUDA_VISIBLE_DEVICES= ./local/test.sh conf/deepspeech2.yaml exp/deepspeech2/checkpoints/avg_1
```
## 预训练模型
您可以从 [this](../../../docs/source/released_model_cn.md) 获得预训练模型.

使用 `tar` 脚本解压模型，然后您可以使用脚本测试模型.

例如:
```shell
wget https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_aishell_ckpt_0.1.1.model.tar.gz
tar xzvf asr0_deepspeech2_aishell_ckpt_0.1.1.model.tar.gz
source path.sh
# 如果你已经处理过数据并得到 manifest 文件，你可以跳过下面 2 个步骤
bash local/data.sh --stage -1 --stop_stage -1
bash local/data.sh --stage 2  --stop_stage 2

CUDA_VISIBLE_DEVICES= ./local/test.sh conf/deepspeech2.yaml exp/deepspeech2/checkpoints/avg_1
```
已发布模型的性能显示在[这里](./RESULTS_CN.md)

## 阶段 4: 静态图模型导出
这个阶段是将动态图转换为静态图.
```bash
 if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
     # 导出 ckpt avg_n
     CUDA_VISIBLE_DEVICES=0 ./local/export.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} exp/${ckpt}/checkpoints/${avg_ckpt}.jit ${model_type}
 fi
```
如果你已经有一个动态图模型，你可以运行这个脚本:
```bash
source path.sh
./local/export.sh deepspeech2.yaml exp/deepspeech2/checkpoints/avg_1 exp/deepspeech2/checkpoints/avg_1.jit offline
```
## 阶段 5: 静态图模型测试
和stage 3类似，也可以测试静态图模型.
```bash
 if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
     # 测试导出 ckpt avg_n
     CUDA_VISIBLE_DEVICES=0 ./local/test_export.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt}.jit ${model_type}|| exit -1
 fi
```
如果您已经导出静态图，则可以运行此脚本:
```bash
CUDA_VISIBLE_DEVICES= ./local/test_export.sh conf/deepspeech2.yaml exp/deepspeech2/checkpoints/avg_1.jit offline
```
## 阶段 6: 单个音频文件推理
在某些情况下，您希望使用经过训练的模型对单个音频文件进行推理. 您可以使用阶段5. 代码如下所示
```bash
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
     # 测试单个 .wav 文件
     CUDA_VISIBLE_DEVICES=0 ./local/test_wav.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} ${model_type} ${audio_file}
 fi
```
你可以自己训练模型，也可以通过下面的脚本下载预训练好的模型:
```bash
wget https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_aishell_ckpt_0.1.1.model.tar.gz
tar xzvf asr0_deepspeech2_aishell_ckpt_0.1.1.model.tar.gz
```
您可以下载音频演示:
```bash
wget -nc https://paddlespeech.bj.bcebos.com/datasets/single_wav/zh/demo_01_03.wav -P data/
```
您需要准备一个音频文件或使用上面的音频演示，请确认音频的采样率为16K。您可以通过运行下面的脚本来获得音频演示的结果.
```bash
CUDA_VISIBLE_DEVICES= ./local/test_wav.sh conf/deepspeech2.yaml conf/tuning/decode.yaml exp/deepspeech2/checkpoints/avg_1 data/demo_01_03.wav
```
