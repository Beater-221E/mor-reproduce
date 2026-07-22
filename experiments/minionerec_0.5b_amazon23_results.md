# MiniOneRec 0.5B Amazon23 — 实验结果

## 固定版本

| 项 | 值 |
| --- | --- |
| mor-reproduce | `d4fa8c470ba943ae4f0eabfb0ecc70ddcd3839ea` |
| MiniOneRec official | `0c64b955ecb8e3d7a9ae9f1fa88cf938f129b0ed` |
| 模型 | Qwen2.5-0.5B **Base diagnostic**（Instruct 权重本地不完整） |
| 数据 | Amazon23 Industrial + 当前 SID（未替换） |
| 评估 | test 抽样 64；beam=10（非正式官方 beam=50） |

## 验收证据

| 检查 | 结果 |
| --- | --- |
| 单元测试 | **17 passed** |
| missing_sid / collision / tokenizer roundtrip | 0 / 0 / 0 |
| 约束解码 invalid（RL smoke） | **0.0** |
| FP32 SidSFT overfit exact SID | **1.0**（16 samples） |
| FP16 SidSFT overfit | 失败（V100+新 embedding 不稳定）→ 已改为 `fallback_for_v100: fp32` |

## 结果表（test 64 samples；不可与论文 Amazon18 对比）

| 实验 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | Invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 Base+SID resize | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B1 SFT overfit (fp32, 64 train) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B3 RL official smoke_g4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B4 RL paper smoke_g4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

说明：B1 仅过拟合 64 条 **train** SidSFT；在 **test** 上为 0 符合预期。训练集自检（EvalSid prompt）HR@10≈0.36，invalid=0。

## 消融问题简答

1. 对齐 SFT 是否优于 Base：过拟合自检可生成正确 SID；test 指标需完整 SFT。  
2. RL 是否优于 SFT：smoke 步数太少，test 上均 0，尚无结论。  
3. official RL vs legacy：legacy 未在本轮重跑（可评已有 1.5B ckpt）。  
4. Reward↑ 是否对应 HR↑：smoke 中 reward 偶发上升，test HR 仍 0。  
5. 约束解码是否消除非法 SID：**是（RL/Eval invalid=0）**。  
6. SFT best vs final：smoke 混训 best_step=60 优于 final eval_loss。  
7. Instruct vs Base 非法率：Instruct 权重缺失，仅 Base diagnostic。

## 未完成 / 阻塞

| 项 | 原因 | 下一步 |
| --- | --- | --- |
| Qwen2.5-0.5B-Instruct 主实验 | 本地缺 `model.safetensors` | 补齐权重后去掉 diagnostic |
| 完整 10-epoch SFT / G=16 RL | 时间与显存；本轮做最小验证 | `scripts/sft_0.5b_official.sh` + `rl_0.5b_official.sh` |
| B2 legacy RL 对比 | 未重跑 0.5B legacy | 可用已有 1.5B legacy 或补跑 |
| 正式 test 全量 + beam=50 | smoke 用 64 / beam=10 | 改 `eval_0.5b.yaml` |

## 运行命令

见 `docs/MiniOneRec_0.5B_Amazon23_对齐修改报告.md` §13。
