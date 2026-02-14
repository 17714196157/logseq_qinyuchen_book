- **Self-Evolution（自我演化）定义** 是指 AI 系统能够**自主地、持续地改进自身**的完整能力——包括生成训练数据、评估输出质量、优化推理策略、调整模型参数，甚至修改系统架构——形成一个**无需人工干预的闭环进化循环**
-
- Self-Evolution（自我演化） [最广义的定义]
      ├── 自我训练（Self-Training） [数据层面：用伪标签扩展数据]
      ├── 迭代自举训练 (Iterative Bootstrapping Training) [过程层面：迭代纠错改进]
      ├── 自我奖励（Self-Rewarding） [评估层面：自己判断输出质量]
      └── ...（其他自我优化技术）
-
- ## **迭代自举训练 (Iterative Bootstrapping Training)**
- ### 核心思想
  这个方法通常包含三个关键步骤的循环：
- 1. 生成（Generation）：模型自己生成训练数据（如数学问题、解题过程）
- 2. 验证（Verification）：通过某种方式判断生成数据的质量（如使用奖励模型、规则验证、多数投票等）
- 3. 训练（Training）：用验证后的高质量数据继续训练模型
-
- #### 具体技术名称
  根据不同的侧重点，这种方法还有几种叫法：
  | 名称                                       | 说明                       |  首推经典  |
  |--|--|--|
  | **RFT (Rejection Sampling Fine-Tuning)** | 拒绝采样微调，只保留高质量样本进行训练      | [[Rejection Sampling for Language Model Alignment ]] |
  | **STaR (Self-Taught Reasoner)**          | 早期的自举推理方法，通过生成-验证-训练循环提升|[[Self-Taught Reasoner Bootstrapping Reasoning With Reasoning]] |
  | **SPIN (Self-Play Fine-Tuning)**         | 自对弈微调，模型与自己博弈生成数据        | [[Self-Play Fine-Tuning]]  |
  | **Self-Rewarding **         | 打破"奖励瓶颈"        | [[Self-Rewarding Language Model]]  |
  | **Meta-Rewarding**         | 引入"元评判"升级        | [[Meta-Rewarding Language Models]]  |
-
- #### 关键挑战与理论分析
- ##### 1. 崩溃模式（Collapse Modes）
- | 类型       | 表现        | 缓解策略         |
  | -------- | --------- | ------------ |
  | **同质化**  | 输出多样性丧失   | 温度采样、多样性奖励   |
  | **退化**   | 模型能力随迭代下降 | 质量过滤、与原始数据混合 |
  | **幻觉固化** | 错误知识被强化   | 事实核查、检索增强    |
- ##### 2. 能力边界
  **"不可能三角"**：完全自主（无外部监督）VS 持续进化（多轮迭代）VS 保证收敛（不崩溃）
-