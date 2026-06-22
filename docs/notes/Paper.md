## RL 训练 RAG — 核心方法

### 论文摘要（1–20）

|  # | 论文                | 简要说明                                                                                      | 链接                                                                    |
| -: | ------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
|  1 | R3-RAG              | 两阶段：冷启动迭代推理+检索 → RL 探索检索环境；奖励：结果(答案正确性) + 过程(相关文档验证)。 | https://arxiv.org/abs/2505.23794                                        |
|  2 | UR2                 | 统一检索与推理的 RL 框架；难度感知课程学习、混合知识访问（离线语料 + LLM 摘要）。             | https://arxiv.org/html/2601.21912                                       |
|  3 | R-Search            | 多奖励（答案质量/证据质量/格式），支持 token 级触发检索并整合证据。                           | https://arxiv.org/pdf/2506.04185                                        |
|  4 | Search-P1           | 路径中心奖励塑形：双轨路径评分 + 软结果评分，提升准确率。                                     | https://arxiv.org/pdf/2602.22576                                        |
|  5 | ProRAG              | 过程监督 RL for RAG：策略预热 → MCTS 构建过程奖励模型 → PRM 引导精炼 → 双粒度优势 RL。     | https://arxiv.org/html/2601.21912                                       |
|  6 | TreePS-RAG          | 将推理 rollout 建模为树，使用蒙特卡洛估计节点效用并相似性剪枝。                               | https://arxiv.org/pdf/2601.06922                                        |
|  7 | ReasonRAG           | MCTS + 最短路径奖励估计(SPRE)，少量样本达成高效训练与性能。                                   | https://arxiv.org/pdf/2505.14069                                        |
|  8 | ReaLM-Retrieve      | 步级不确定性检测器(RSUS) + 检索干预策略 + 效率优化，减少调用提高 F1。                         | https://arxiv.org/html/2604.26649                                       |
|  9 | GRIP                | 将检索决策嵌入 token 级解码（[RETRIEVE]/[ANSWER]/[INTERMEDIARY]/[SOLVED] 控制 token）。       | https://arxiv.org/pdf/2604.11407                                        |
| 10 | GraphRAG-R1         | 图 + 文本混合检索，过程约束奖励(PRA/CAF)，三阶段训练提高推理容量。                            | https://arxiv.org/pdf/2507.23581                                        |
| 11 | MMOA-RAG            | 将 RAG 管道视为多智能体协作任务，用 MAPPO 联合优化模块。                                      | https://arxiv.org/pdf/2501.15228                                        |
| 12 | MAO-ARAG            | 多智能体编排：Planner + 多 Executor，Planner 用 PPO 动态组装工作流。                          | https://arxiv.org/pdf/2508.01005                                        |
| 13 | A-RAG               | 三级检索接口 (keyword/semantic/chunk_read)，模型自发泛化工作流。                              | https://arxiv.org/pdf/2602.03442                                        |
| 14 | RRPO                | 将重排序建模为 MDP，使用 LLM 反馈作为奖励，优化文档“有用性”。                               | https://arxiv.org/pdf/2604.02091                                        |
| 15 | PyRAG               | 将多跳推理重构为 Python 程序合成与执行，三 Agent 架构（Decompose/Plan/Answer）。              | https://arxiv.org/html/2605.12975                                       |
| 16 | Self-Correcting RAG | 上下文选择为 MMKP，输出端用 NLI 引导 MCTS 验证忠实性并自纠正。                                | https://arxiv.org/pdf/2604.10734                                        |
| 17 | ReflectiveRAG       | 自反思检索 + 对比噪声消除，轻量延迟下提升 EM 并降低冗余。                                     | https://aclanthology.org/2026.eacl-industry.27.pdf                      |
| 18 | SEAKR               | 从 LLM 内部状态提取不确定性以触发检索并进行重排序/自感知推理。                                | https://aclanthology.org/anthology-files/pdf/acl/2025.acl-long.1312.pdf |
| 19 | SIM-RAG             | 系统自我练习多轮检索以生成过程监督数据，训练轻量 Critic 评估信息充分性。                      | https://arxiv.org/pdf/2505.02811                                        |
| 20 | PAR2-RAG            | 广度优先锚定 + 深度优先精炼，提升准确率与 NDCG。                                              | https://arxiv.org/pdf/2603.29085                                        |

---

### 开源/仓库信息

|  # | 论文                | arXiv ID   | GitHub 仓库                                               | 备注                                           |
| -: | ------------------- | ---------- | --------------------------------------------------------- | ---------------------------------------------- |
|  1 | R3-RAG              | 2505.23794 | https://github.com/Yuan-Li-FNLP/R3-RAG                    | 论文声明，HuggingFace 有模型权重               |
|  2 | UR2                 | 2508.06165 | https://github.com/Tsinghua-dhy/UR2                       | 127 stars，Apache-2.0；Modelscope 有模型       |
|  3 | R-Search            | 2506.04185 | https://github.com/QingFei1/R-Search                      | HuggingFace 有训练好的模型和数据集             |
|  4 | ProRAG              | 2601.21912 | https://github.com/lilinwz/ProRAG                         | 论文声明                                       |
|  5 | ReasonRAG           | 2505.14069 | https://github.com/Applied-Machine-Learning-Lab/ReasonRAG | NeurIPS'25；HuggingFace 有 RAG-ProGuide 数据集 |
|  6 | ReaLM-Retrieve      | 2604.26649 | https://github.com/bettyguo/realm-retrieve                | 含步骤分割器+策略网络+检索器                   |
|  7 | GRIP                | 2604.11407 | https://github.com/WisdomShell/GRIP                       | ACL'26，项目页包含更多资料                     |
|  8 | GraphRAG-R1         | 2507.23581 | https://github.com/ycygit/GraphRAG-R1                     | HuggingFace 有模型权重；Zenodo DOI             |
|  9 | MMOA-RAG            | 2501.15228 | https://github.com/chenyiqun/MMOA-RAG                     | 论文声明                                       |
| 10 | MAO-ARAG            | 2508.01005 | https://github.com/chenyiqun/Agentic-RAG                  | 同一作者不同仓库                               |
| 11 | A-RAG               | 2602.03442 | https://github.com/Ayanami0730/arag                       | 240 stars，MIT license，含评估套件             |
| 12 | PyRAG               | 2605.12975 | https://github.com/GasolSun36/PyRAG                       | 论文声明                                       |
| 13 | Self-Correcting RAG | 2604.10734 | https://github.com/xjiacs/Self-Correcting-RAG             | 论文声明                                       |
| 14 | SEAKR               | ACL 2025   | https://github.com/thu-keg/seakr                          | GPL-3.0；清华 KEG 出品                         |
| 15 | SIM-RAG             | 2505.02811 | https://github.com/ucscirkm/SIM-RAG                       | 论文声明 "All code and data are available"     |

### 暂无开源代码（5 篇）

| # | 论文          | arXiv ID   | 状态说明                                                  |
| -: | ------------- | ---------- | --------------------------------------------------------- |
| 1 | Search-P1     | 2602.22576 | 附录标注 "(anonymized)"，可能仍在匿名评审期               |
| 2 | TreePS-RAG    | 2601.06922 | 论文未提及代码链接，搜索未找到仓库                        |
| 3 | RRPO          | 2604.02091 | 论文未提及代码链接，仅有第三方 (AutoRAG) 请求实现的 issue |
| 4 | ReflectiveRAG | EACL 2026  | Amazon 行业论文，无公开代码仓库                           |
| 5 | PAR2-RAG      | 2603.29085 | 论文未提及代码链接，搜索未找到仓库                        |

---

## LLaMA + RAG — 近两年高引用/期刊论文

### 方法类（基于 LLaMA 架构的 RAG 创新）

| # | 论文     | 简要说明                                                                                                      | 会议/期刊                | 估计引用 | 链接                             |
| -: | -------- | ------------------------------------------------------------------------------------------------------------- | ------------------------ | -------: | -------------------------------- |
| 1 | Self-RAG | 训练单一 LM 按需检索、生成并自反思；引入 reflection token 实现可控推理；基于 LLaMA-2-7B/13B。                 | **ICLR 2024 Oral** |    ~600+ | https://arxiv.org/abs/2310.05530 |
| 2 | RankRAG  | 统一排序与生成的指令微调框架；LLaMA-3 骨干，加少量排序数据即可同时做重排+生成；9 基准超 ChatQA-1.5 和 GPT-4。 | **NeurIPS 2024**   |    ~350+ | https://arxiv.org/abs/2407.02485 |
| 3 | RA-DIT   | 轻量级双指令微调，分别微调 LLM 和检索器；基于 LLaMA + DRAGON+，RA-DIT-65B 达知识密集任务 SOTA。               | **ICLR 2024**      |    ~300+ | https://arxiv.org/abs/2310.01352 |
| 4 | ChatQA   | NVIDIA 出品，LLaMA-3 对话式 RAG；ChatQA-1.5-70B 接近 GPT-4 水平。                                             | ACL 2024                 |    ~150+ | https://arxiv.org/abs/2401.10225 |
| 5 | ImpRAG   | 无需显式查询的 RAG；基于 LLaMA-3 将层分为检索/生成专用组，同一前向传播完成检索+生成；EM 提升 3.6–11.5。      | 预印本 2025              |       — | https://arxiv.org/abs/2506.02279 |
| 6 | CoRAG    | o1 风格 RAG，拒绝采样自动生成检索链，多种测试时解码策略；KILT 基准新 SOTA，多跳 QA EM+10。                    | Microsoft 2025           |       — | https://arxiv.org/abs/2501.14342 |
| 7 | ReaRAG   | 面向大推理模型的事实性增强 RAG，限制推理链长度避免过度思考；多跳 QA 超越 Search-o1。                          | 预印本 2025              |       — | https://arxiv.org/abs/2503.21729 |

### 图增强 RAG（GraphRAG 系列）

|  # | 论文                 | 简要说明                                                                            |       估计引用 | 链接                             |
| -: | -------------------- | ----------------------------------------------------------------------------------- | -------------: | -------------------------------- |
|  8 | GraphRAG (Microsoft) | 知识图谱+社区摘要，全局/局部双模式查询，可结合 LLaMA。                              | **1531** | https://arxiv.org/abs/2404.16130 |
|  9 | LightRAG             | 双层检索（实体级+关系级），速度远超 GraphRAG，支持增量更新。                        |          ~200+ | https://arxiv.org/abs/2410.05779 |
| 10 | GFM-RAG              | 首个图基础模型 for RAG；8M 参数 GNN 在 60 个 KG 上预训练，零样本迁移无需领域微调。  |         新发表 | https://arxiv.org/abs/2502.01113 |
| 11 | RAS                  | 动态构建问题特定知识图谱，迭代检索+结构化推理；LLaMA-2/3 骨干，7 基准提升 7–9.7%。 |         新发表 | https://arxiv.org/abs/2502.10996 |

### 综述类

|  # | 论文                             | 简要说明                                                                                                  | 会议/期刊                     | 估计引用 | 链接                                              |
| -: | -------------------------------- | --------------------------------------------------------------------------------------------------------- | ----------------------------- | -------: | ------------------------------------------------- |
| 12 | RAG for LLMs Survey (Gao et al.) | RAG 领域最广泛引用综述；系统梳理 Naive → Advanced → Modular RAG 范式演进。                              | arXiv 2024                    |   ~1000+ | https://arxiv.org/abs/2312.10997                  |
| 13 | RAG for AIGC Survey (PKU-DAIR)   | 全面综述 RAG 基础范式（query-based / logits-based / latent / parametric），覆盖多模态。                   | arXiv 2024                    |    ~400+ | https://arxiv.org/abs/2402.19473                  |
| 14 | RAG Meets LLMs Survey            | 从架构、训练策略、应用三维综述 RA-LLMs。                                                                  | **ACM 出版** 2024       |    ~200+ | https://arxiv.org/abs/2405.06211                  |
| 15 | RAG for NLP Survey               | 检索融合四类分类法（query/logits/latent/parametric），含教程代码。                                        | arXiv 2024                    |    ~150+ | https://arxiv.org/abs/2407.13193                  |
| 16 | Systematic Lit Review of RAG     | PRISMA 2020 系统文献综述，128 篇论文，发现 RAG 从 DPR+seq2seq 向模块化策略驱动转型。                      | **MDPI BDCC** 📰 2025   |   新发表 | https://doi.org/10.3390/bdcc9120320               |
| 17 | RAG: Architectures & Robustness  | 按 retriever-centric / generator-centric / hybrid / robustness 分类，含 LLaMA 2/3 比较表。                | arXiv 2025                    |   新发表 | https://arxiv.org/abs/2506.00054                  |
| 18 | Agentic RAG + Deep Reasoning     | 统一 Reasoning-Enhanced RAG → RAG-Enhanced Reasoning → Synergized RAG-Reasoning 三阶段框架，200+ 论文。 | **EMNLP Findings** 2025 |   新发表 | https://github.com/DavidZWZ/Awesome-RAG-Reasoning |

### 顶刊/顶会代表作

|  # | 论文        | 简要说明                                                          | 会议/期刊             | 链接                                           |
| -: | ----------- | ----------------------------------------------------------------- | --------------------- | ---------------------------------------------- |
| 19 | OpenScholar | 基于 LLaMA-3.1 的文献综述 RAG；45M 论文数据存储，8B 超越 GPT-4o。 | **Nature 2025** | https://arxiv.org/abs/2411.14199               |
| 20 | LaRA        | RAG vs 长上下文基准；2326 测试用例，11 个 LLM 评估，无银弹。      | **ICML 2025**   | https://proceedings.mlr.press/v267/li25dv.html |

### 开源/仓库信息

|  # | 论文                 | GitHub 仓库                                        | 备注           |
| -: | -------------------- | -------------------------------------------------- | -------------- |
|  1 | Self-RAG             | https://selfrag.github.io/                         | 模型+代码      |
|  2 | RankRAG              | 模型权重已发布（HuggingFace）                      | NeurIPS 2024   |
|  3 | RA-DIT               | 开源                                               | ICLR 2024      |
|  4 | ChatQA               | NVIDIA 开源                                        | 模型权重+代码  |
|  5 | ImpRAG               | 预印本，待开源                                     | —             |
|  6 | CoRAG                | https://github.com/microsoft/LMOps/tree/main/corag | Microsoft      |
|  7 | ReaRAG               | 预印本，待开源                                     | —             |
|  8 | GraphRAG             | https://github.com/microsoft/graphrag              | 引用 1531      |
|  9 | LightRAG             | https://github.com/HKUDS/LightRAG                  | 港大           |
| 10 | GFM-RAG              | 预印本，待确认                                     | —             |
| 11 | RAS                  | https://github.com/pat-jj/RAS                      | LLaMA-2/3 骨干 |
| 12 | RAG Survey (AIGC)    | https://github.com/PKU-DAIR/RAG-Survey             | PKU-DAIR       |
| 13 | RAG Tutorials        | https://github.com/luffy06/RAG-Tutorials           | 含教程代码     |
| 14 | RAG+Reasoning Survey | https://github.com/DavidZWZ/Awesome-RAG-Reasoning  | 200+ 论文收录  |
| 15 | OpenScholar          | 全部开源（代码+模型+数据存储）                     | Nature 2025    |
| 16 | LaRA                 | https://github.com/Alibaba-NLP/LaRA                | ICML 2025      |
