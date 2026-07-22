# Source对照矩阵（固定官方 commit）

- mor-reproduce commit: `d4fa8c470ba943ae4f0eabfb0ecc70ddcd3839ea`
- MiniOneRec official commit: `0c64b955ecb8e3d7a9ae9f1fa88cf938f129b0ed`
- Official root: `/home/sheng/proj/MiniOneRec-official`

| 模块 | 论文描述 | 官方源码实现 (0c64b955) | 官方启动脚本实际参数 | 当前复现(旧) | 决定采用的实现 |
| --- | --- | --- | --- | --- | --- |
| SFT任务 | 推荐 + 对齐等多任务 | `SidSFTDataset` + `SidItemFeatDataset`(title↔SID) + `FusionSeqRecDataset`(SID→title)；Preference等注释掉 | `sft.sh` 调 `sft.py`（无额外任务开关） | 自定义 chat JSONL 多任务 | **official_source: 三任务 ConcatDataset** |
| SFT Loss | response NTP/CE | HF Trainer + labels=-100 on prompt | 同 | HF Trainer CE | **official_source** |
| Early stopping | patience 1 | `EarlyStoppingCallback(patience=3)` | 未覆盖 | patience 0（关） | **official_source=3；paper_aligned=1** |
| RL数据任务 | 两类任务继续 | `SidDataset`+`RLTitle2Sid`+`RLSeqTitle2Sid`(sample=10000)；Sid2Title 注释掉 | `rl.sh` | 旧 JSONL RL | **官方启用的三项** |
| Rollout | 约束 beam G=16 | `beam_search` + ConstrainedLogitsProcessor；RepeatRandomSampler | `--num_generations 16 --beam_search True` | sample G=2 | **official_source** |
| GRPO Loss | old-policy clip | `exp(logπ-logπ.detach())*A`（ratio≡1）+ β KL | dapo/gspo False | 序列 mean logπ×A | **official_source 复刻假ratio；paper_aligned 真 ratio+clip** |
| KL | β 未给数值 | Schulman KL；β CLI | `--beta 1e-3` | β=0.001 | **β=1e-3（官方脚本）** |
| Reward | R_rule+R_rank | ranking=[rule, ndcg_rule] | `--reward_type ranking` | hybrid 近似 | **逐行官方** |
| Advantage | (R-μ)/σ | eps=1e-4；无 clip | 同 | +adv_clip=5 | **官方无 clip** |
| Constrained decoding | 合法 SID trie | hash_dict + prefix_index=3 | evaluate/rl 使用 | 自研 SIDTrie | **官方 ConstrainedLogitsProcessor** |
| Evaluation | HR/NDCG@3/5/10 | evaluate.py beam50 + calc.py | `--num_beams 50` | beam50；缺 JSON | **官方协议；0.5B smoke 可用较小 beam 并标注** |

## 关键超参来源

| 参数 | 值 | 来源 |
| --- | ---: | --- |
| SFT lr | 3e-4 | 论文 + sft.py 默认 |
| SFT global batch | 1024 | sft.sh `--batch_size 1024`（再 / micro / world） |
| SFT epochs | 10 | 论文 + sft.py |
| SFT early stop | 3 | sft.py（非论文 1） |
| RL lr | 1e-5 | rl.sh |
| RL β | 1e-3 | rl.sh |
| RL G | 16 | rl.sh / 论文 |
| RL beam_search | True | rl.sh |
| RL epochs | 2 | rl.sh |
| weight_decay | 0.0 | 官方 TrainingArguments 未设 |
