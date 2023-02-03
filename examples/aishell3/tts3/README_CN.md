# FastSpeech2 with AISHELL-3

此示例包含用于使用 [AISHELL-3](http://www.aishelltech.com/aishell_3) 训练 [Fastspeech2](https://arxiv.org/abs/2006.04558) 模型的代码。

AISHELL-3 是一个大规模、高保真多说话人普通话语音语料库，可用于训练多说话人文本到语音 (TTS) 系统。

我们在这里使用 AISHELL-3 训练多说话人 fastspeech2 模型。

## 数据集
### 下载解压
从其[官方网站](http://www.aishelltech.com/aishell_3)下载 AISHELL-3 训练数据并将其解压缩到 `~/datasets`。然后数据集位于目录 `~/datasets/data_aishell3` 中。
 
### 获取 MFA 结果并提取
我们使用 [MFA2.x](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) 获取 aishell3_fastspeech2 的持续时间。

您可以从此处下载 [aishell3_alignment_tone.tar.gz](https://paddlespeech.bj.bcebos.com/MFA/AISHELL-3/with_tone/aishell3_alignment_tone.tar.gz)，或参考我们仓库的 [mfa 示例](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/other/mfa)(现在使用 MFA1.x)训练您的 MFA 模型。

## 开始

假设数据集的路径是`~/datasets/data_aishell3`。

假设 AISHELL-3 的 MFA 结果路径是 `./aishell3_alignment_tone`

运行下面的命令

1. **设置环境变量**
2. 预处理数据集
3. 训练模型
4. 合成波形
    - jsonl`. 从 `metadata.jsonl` 合成波形
    -  从文本文件合成波形
```bash
./run.sh
```
您可以选择要运行的一系列阶段，或将 `stage` 设置为 `stop-stage` 以仅使用一个阶段，例如，运行以下命令将仅预处理数据集。
```bash
./run.sh --stage 0 --stop-stage 0
```

### 数据预处理
```bash
./local/preprocess.sh ${conf_path}
```
当它完成时。在当前目录中创建一个`dump`文件夹。转储文件夹的结构如下所示。
```text
dump
├── dev
│   ├── norm
│   └── raw
├── phone_id_map.txt
├── speaker_id_map.txt
├── test
│   ├── norm
│   └── raw
└── train
    ├── energy_stats.npy
    ├── norm
    ├── pitch_stats.npy
    ├── raw
    └── speech_stats.npy
```
数据集分为 3 个部分，即 `train`, `dev` 和 `test`，每个部分包含一个 `norm` 和 `raw` 子文件夹。 raw 文件夹包含每个话语的语音、音调和能量特征，而 norm 文件夹包含归一化的特征。用于规范化特征的统计数据是从位于 `dump/train/*_stats.npy` 中的训练集计算得出的。

此外，每个子文件夹中都有一个 `metadata.jsonl`。它是一个类似表格的文件，包含音素(phones)、文本长度(text_lengths)、讲话长度(speech_lengths)、持续时间(durations)、语音特征路径、音调特征路径、能量特征路径、说话者(speaker)和每个话语(utterance)的 id。

### 模型训练
`./local/train.sh` calls `${BIN_DIR}/train.py`.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path}
```
这是完整的帮助信息:
```text
usage: train.py [-h] [--config CONFIG] [--train-metadata TRAIN_METADATA]
                [--dev-metadata DEV_METADATA] [--output-dir OUTPUT_DIR]
                [--ngpu NGPU] [--phones-dict PHONES_DICT]
                [--speaker-dict SPEAKER_DICT] [--voice-cloning VOICE_CLONING]

训练 FastSpeech2 模型。

optional arguments:
  -h, --help            显示此帮助信息并退出
  --config CONFIG       fastspeech2 配置文件.
  --train-metadata TRAIN_METADATA
                        训练数据.
  --dev-metadata DEV_METADATA
                        dev data.
  --output-dir OUTPUT_DIR
                        输出目录.
  --ngpu NGPU           如果 ngpu=0, 使用 cpu.
  --phones-dict PHONES_DICT
                        音素字典文件.
  --speaker-dict SPEAKER_DICT
                        多说话者模型的说话者 ID 映射文件.
  --voice-cloning VOICE_CLONING
                        是否训练语音克隆模型.
```
1. `--config` 是 yaml 格式的配置文件，用于覆盖默认配置, 可以在 `conf/default.yaml` 找到.
2. `--train-metadata` 和 `--dev-metadata` 应该是 `dump` 文件夹中 `train` 和 `dev` 的标准化子文件夹中的元数据文件。
3. `--output-dir` 是保存实验结果的目录。检查点保存在此目录内的 `checkpoints/` 中。
4. `--ngpu` 是要使用的 gpu 数量，如果 ngpu == 0，则使用 cpu。
5. `--phones-dict` 是因素词汇文件的路径。
6. `--speaker-dict` 是训练多说话人 FastSpeech2 模型时说话人 ID 映射文件的路径。

### 合成
我们使用[并行 wavegan](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/voc1) 作为神经声码器。

从 [pwg_aishell3_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_aishell3_ckpt_0.5.zip) 下载预训练的并行 wavegan 模型并解压。
```bash
unzip pwg_aishell3_ckpt_0.5.zip
```
Parallel WaveGAN 检查点包含下面列出的文件:
```text
pwg_aishell3_ckpt_0.5
├── default.yaml                   # 用于训练并行 wavegan 的默认配置
├── feats_stats.npy                # 训练并行 wavegan 时用于归一化频谱图的统计数据
└── snapshot_iter_1000000.pdz      # generator parameters of parallel wavegan 并行wavegan生成器参数
```
`./local/synthesize.sh` 调用 `${BIN_DIR}/../synthesize.py`, 它可以从 `metadata.jsonl` 合成波形。
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name}
```
```text
usage: synthesize.py [-h]
                     [--am {speedyspeech_csmsc,fastspeech2_csmsc,fastspeech2_ljspeech,fastspeech2_aishell3,fastspeech2_vctk,tacotron2_csmsc,tacotron2_ljspeech,tacotron2_aishell3}]
                     [--am_config AM_CONFIG] [--am_ckpt AM_CKPT]
                     [--am_stat AM_STAT] [--phones_dict PHONES_DICT]
                     [--tones_dict TONES_DICT] [--speaker_dict SPEAKER_DICT]
                     [--voice-cloning VOICE_CLONING]
                     [--voc {pwgan_csmsc,pwgan_ljspeech,pwgan_aishell3,pwgan_vctk,mb_melgan_csmsc,wavernn_csmsc,hifigan_csmsc,hifigan_ljspeech,hifigan_aishell3,hifigan_vctk,style_melgan_csmsc}]
                     [--voc_config VOC_CONFIG] [--voc_ckpt VOC_CKPT]
                     [--voc_stat VOC_STAT] [--ngpu NGPU]
                     [--test_metadata TEST_METADATA] [--output_dir OUTPUT_DIR]

使用声学模型和声码器合成

可选参数:
  -h, --help            显示此帮助信息并退出
  --am {speedyspeech_csmsc,fastspeech2_csmsc,fastspeech2_ljspeech,fastspeech2_aishell3,fastspeech2_vctk,tacotron2_csmsc,tacotron2_ljspeech,tacotron2_aishell3}
                        选择 tts 任务的声学模型类型。
  --am_config AM_CONFIG
                        声学模型的配置。
  --am_ckpt AM_CKPT     声学模型的检查点文件。
  --am_stat AM_STAT     mean and standard deviation used to normalize spectrogram when training acoustic model.
                        训练声学模型时用于归一化声谱图的均值和标准差。
  --phones_dict PHONES_DICT
                        音素字典文件。
  --tones_dict TONES_DICT
                        tone vocabulary file.
                        音调字典文件。
  --speaker_dict SPEAKER_DICT
                        说话人 ID 映射文件。
  --voice-cloning VOICE_CLONING
                        是否训练语音克隆模型。
  --voc {pwgan_csmsc,pwgan_ljspeech,pwgan_aishell3,pwgan_vctk,mb_melgan_csmsc,wavernn_csmsc,hifigan_csmsc,hifigan_ljspeech,hifigan_aishell3,hifigan_vctk,style_melgan_csmsc}
                        选择 tts 任务的声码器类型.
  --voc_config VOC_CONFIG
                        voc的配置.
  --voc_ckpt VOC_CKPT   voc的检查点文件.
  --voc_stat VOC_STAT   mean and standard deviation used to normalize spectrogram when training voc.
                        训练 voc 时用于归一化频谱图的均值和标准差.
  --ngpu NGPU           if ngpu == 0, use cpu.
  --test_metadata TEST_METADATA
                        测试元数据.
  --output_dir OUTPUT_DIR
                        输出目录.
```
`.localsynthesize_e2e.sh` 调用 `{BIN_DIR}..synthesize_e2e.py`，可以从文本文件合成波形.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize_e2e.sh ${conf_path} ${train_output_path} ${ckpt_name}
```
```text
usage: synthesize_e2e.py [-h]
                         [--am {speedyspeech_csmsc,speedyspeech_aishell3,fastspeech2_csmsc,fastspeech2_ljspeech,fastspeech2_aishell3,fastspeech2_vctk,tacotron2_csmsc,tacotron2_ljspeech}]
                         [--am_config AM_CONFIG] [--am_ckpt AM_CKPT]
                         [--am_stat AM_STAT] [--phones_dict PHONES_DICT]
                         [--tones_dict TONES_DICT]
                         [--speaker_dict SPEAKER_DICT] [--spk_id SPK_ID]
                         [--voc {pwgan_csmsc,pwgan_ljspeech,pwgan_aishell3,pwgan_vctk,mb_melgan_csmsc,style_melgan_csmsc,hifigan_csmsc,hifigan_ljspeech,hifigan_aishell3,hifigan_vctk,wavernn_csmsc}]
                         [--voc_config VOC_CONFIG] [--voc_ckpt VOC_CKPT]
                         [--voc_stat VOC_STAT] [--lang LANG]
                         [--inference_dir INFERENCE_DIR] [--ngpu NGPU]
                         [--text TEXT] [--output_dir OUTPUT_DIR]

使用声学模型和声码器合成

可选参数:
  -h, --help            显示此帮助信息并退出
  --am {speedyspeech_csmsc,speedyspeech_aishell3,fastspeech2_csmsc,fastspeech2_ljspeech,fastspeech2_aishell3,fastspeech2_vctk,tacotron2_csmsc,tacotron2_ljspeech}
                        选择 tts 任务的声学模型类型.
  --am_config AM_CONFIG
                        声学模型配置.
  --am_ckpt AM_CKPT     声学模型的检查点文件.
  --am_stat AM_STAT     训练声学模型时用于归一化声谱图的均值和标准差.
  --phones_dict PHONES_DICT
                        因素字典文件.
  --tones_dict TONES_DICT
                        音调字典文件.
  --speaker_dict SPEAKER_DICT
                        说话人 ID 映射文件.
  --spk_id SPK_ID       多说话人声学模型的说话人 id
  --voc {pwgan_csmsc,pwgan_ljspeech,pwgan_aishell3,pwgan_vctk,mb_melgan_csmsc,style_melgan_csmsc,hifigan_csmsc,hifigan_ljspeech,hifigan_aishell3,hifigan_vctk,wavernn_csmsc}
                        选择 tts 任务的声码器类型.
  --voc_config VOC_CONFIG
                        voc的配置.
  --voc_ckpt VOC_CKPT   voc的检查点文件.
  --voc_stat VOC_STAT   训练 voc 时用于归一化频谱图的均值和标准差.
  --lang LANG           选择模型语言. zh 或 en
  --inference_dir INFERENCE_DIR
                        保存推理(inference)模型的目录
  --ngpu NGPU           if ngpu == 0, use cpu.
  --text TEXT           要合成的文本，每行一个 'utt_id 句子' 对.
  --output_dir OUTPUT_DIR
                        输出目录.
```
1. `--am` 是格式为 `{model_name}_{dataset}` 的声学模型类型
2. `--am_config`, `--am_ckpt`, `--am_stat`, `--phones_dict` `--speaker_dict` 是声学模型的参数，对应于 fastspeech2 预训练模型中的 5 个文件。
3. `--voc` 是格式为 `{model_name}_{dataset}` 的声码器类型
4. `--voc_config`, `--voc_ckpt`, `--voc_stat` 是声码器的参数，对应于并行 wavegan 预训练模型中的 3 个文件。
5. `--lang` 是模型语言，可以是 `zh` 或 `en`。
6. `--test_metadata` 应该是 `dump` 文件夹中 `test` 的规范化(归一化)子文件夹中的元数据文件。
7. `--text` 是文本文件，其中包含要合成的句子。
8. `--output_dir` 是保存合成音频文件的目录。
9. `--ngpu` 是要使用的 gpu 数量，如果 ngpu == 0，则使用 cpu。

## 预训练模型
音频边缘无静音的预训练 FastSpeech2 模型(Pretrained FastSpeech2 model with no silence in the edge of audios):
- [fastspeech2_aishell3_ckpt_1.1.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_aishell3_ckpt_1.1.0.zip)
- [fastspeech2_conformer_aishell3_ckpt_0.2.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_conformer_aishell3_ckpt_0.2.0.zip) (Thanks for [@awmmmm](https://github.com/awmmmm)'s contribution)

静态模型可以在这里下载：
- [fastspeech2_aishell3_static_1.1.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_aishell3_static_1.1.0.zip)

ONNX 模型可以在这里下载：
- [fastspeech2_aishell3_onnx_1.1.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_aishell3_onnx_1.1.0.zip)

Paddle-Lite 模型可以在这里下载：
- [fastspeech2_aishell3_pdlite_1.3.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_aishell3_pdlite_1.3.0.zip)

FastSpeech2 检查点包含下面列出的文件。

```text
fastspeech2_aishell3_ckpt_1.1.0
├── default.yaml            # 用于训练 fastspeech2 的默认配置
├── energy_stats.npy        # 训练 fastspeech2 时用于标准化能量的统计数据
├── phone_id_map.txt        # 训练 fastspeech2 时的因素字典文件
├── pitch_stats.npy         # 训练 fastspeech2 时用于标准化(normalize)音调的统计数据
├── snapshot_iter_96400.pdz # 模型参数和优化器状态
├── speaker_id_map.txt      # 训练多说话人 fastspeech2 时的说话人 ID 映射文件
└── speech_stats.npy        # 训练 fastspeech2 时用于归一化(normalize)声谱图的统计数据
```
您可以使用以下脚本合成 `${BIN_DIR}/../sentences.txt` 使用预训练的 fastspeech2 和并行 wavegan 模型:
```bash
source path.sh

FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ${BIN_DIR}/../synthesize_e2e.py \
  --am=fastspeech2_aishell3 \
  --am_config=fastspeech2_aishell3_ckpt_1.1.0/default.yaml \
  --am_ckpt=fastspeech2_aishell3_ckpt_1.1.0/snapshot_iter_96400.pdz \
  --am_stat=fastspeech2_aishell3_ckpt_1.1.0/speech_stats.npy \
  --voc=pwgan_aishell3 \
  --voc_config=pwg_aishell3_ckpt_0.5/default.yaml \
  --voc_ckpt=pwg_aishell3_ckpt_0.5/snapshot_iter_1000000.pdz \
  --voc_stat=pwg_aishell3_ckpt_0.5/feats_stats.npy \
  --lang=zh \
  --text=${BIN_DIR}/../sentences.txt \
  --output_dir=exp/default/test_e2e \
  --phones_dict=fastspeech2_aishell3_ckpt_1.1.0/phone_id_map.txt \
  --speaker_dict=fastspeech2_aishell3_ckpt_1.1.0/speaker_id_map.txt \
  --spk_id=0 \
  --inference_dir=exp/default/inference
```
