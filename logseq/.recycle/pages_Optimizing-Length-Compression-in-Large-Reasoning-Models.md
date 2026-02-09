-
- ### **抽象**
  
  大型推理模型 ( LRM ) 取得了显著的成功, 但它们经常产生不必要和冗长的推理链。我 们将这个问题的一个核心方面确定为 " 无效 思维 "—— 模型在得出正确答案后往往会反 复仔细检查他们的工作。为了解决这种特定 的低效率问题,我们超越了效率和效率的一 般原则,提出了两个新的细粒度原则:简洁, 主张消除冗余,以及充分性,确保保留关键 推理步骤。在这些原则的指导下,我们引入 了 LC‑R1,这是一种基于群体相对策略优化 ( GRPO ) 的训练后方法。LC‑R1 采用了 一种新颖的组合,即 长度奖励 ( Length Reward ) 用于整体简洁性,而压缩奖励 ( Compress Reward ) 则专门用于消除思 考过程中的无效部分。对多个推理基准的广 泛实验表明,LC‑R1 实现了
  
  ![image.jpeg](../assets/Optimizing-Length-Compression-in-Large-Reasoning-Models/_page_0_Picture_5.jpeg)
  
  图 1:低效推理模型和高效模型的比较。前者倾向于 在得出与给定问题相对应的正确答案后进行冗长的自 我检查过程,从而导致推理效率低下。用 LC‑R1 训练 的模型获得了更高效的推理过程来获得正确答案,没 有任何无效的思维过程。
- ## **1. 引言**
  
  最近的 " 长期思考 " 大型推理模型 ( LRM ),例如 OpenAI 的 O1 ( Jaech 等人,2024 年)和 Deepseek‑R1 ( DeepSeek‑AIet al.,2025 年),代表了 基础思维链 ( CoT ) 技术的重要范式扩展( Wei et al., 2023 )。这些模型通过强化学习 ( RL ) 进行微调,迭代 地改进解决方案,以在复杂的原因中实现前所未有的性能 ‑
  
  1 马里兰大学。通信对象:天一 周 < tianyi.david.zhou@gmail.com>。 预印本。审核中。
  
  ing 数学和编程等任务( Sun et al., 2025;Gu et al., 2024 )。然而,随着 " 深度思考 " 能力的 提高,一个突出的问题是推理过程中对计算资源的过 度消耗( Chen et al., 2025;Aggarwal 和 Welleck,2025 年)。具体来说,现有模型在解决复 杂度低或解决方案路径清晰的问题时,往往会产生冗 长甚至不必要的推理链。这种现象被称为 " 过度思考 ", 表现为模型消耗的计算资源远远超过问题本身得出正 确结论所需的计算资源( Chen et al., 2024;Sui 等 人,2025 年 ;Cuadron 等人,2025 年)。那里 ‑
  
   ![image.jpeg](../assets/Optimizing-Length-Compression-in-Large-Reasoning-Models/_page_1_Figure_1.jpeg)
  
  图 2:两种推理模型上不同方法的功效 ‑ 效率权衡的帕累托分析。x 轴表示推理长度变化,y 轴表示相对于原始 模型(在方程 12 中定义)的精度变化,左上角表示理想位置。较小和较暗的标记表示较高的有效思维 ( VT ) 率(在方程 1 中定义),表示更高效的思维过程。与同样处于帕累托前沿的其他方法相比,LC‑R1 实现了更有 利的权衡,以最小精度下降为代价获得了更高的压缩率,并且它还实现了更高的 VT 率。我们的消融变体(无 C‑ 奖励,无 L‑ 奖励)的次优性能进一步证明了我们的双重奖励设计的重要性。
  
  在此之前,会出现一个关键问题:
## 我们如何在显著提高推理效率的同时保持高推理 效能?

以前的工作通过微调较短的演示 ( SFT ) 来解决这 个问题 ( Chen et al., 2024 ),构建简洁性优先数 据集 ( Luo et al., 2025a;Shen et al., 2025 ), 或将长度惩罚整合到 RL 中( Hou et al., 2025;Luo 等人,2025b;Team et al., 2025 )。然 而,这些方法经常将推理过程视为一个黑匣子,在不 分析思想本身的内部结构的情况下惩罚长度。

为了解决这一差距,我们深入研究了 " 过度思考 " 的结构,并确定了一种特定的模式:模型在已经得出 正确答案后,经常进行冗余的 " 双重检查 "。我们 将这种现象称为 " 无效思维 ",如图 1 所示。为了 量化它,我们引入了一个新的指标,即有效思维 ( VT ) 率,它衡量推理过程中对得出初始正确结论至 关重要的比例。

在这一见解的指导下,我们提出了两个细粒度的原则: 简洁(消除冗余)和充足性(保留必要的步骤)。然 后,我们介绍了 LC‑R1,这是一种基于 GRPO 的训练 后方法,可实施这些原则。LC‑R1 独特地将整体简洁 性的长度奖励与新颖的压缩奖励相结合,以直接指导 模型在得出正确答案后终止思考过程。

我们在 7 个基准测试中对 2 个推理模型进行了全面的 实验。实证结果表明,LC‑R1 在功效和效率之间实现 了比以前的方法更有利的权衡,如图 2 所示。具体来 说,在准确率仅下降 2% 的情况下,我们的方法在平 均序列长度上减少了 50%。消融研究还证明了长度奖 励和压缩奖励对于实现有效推理的必要性。进一步的 研究表明,该方法在不影响模型爆炸能力的情况下实 现了高效压缩,其效率可以推广到各种难度问题。总 之,我们的贡献可以总结如下:

• 本文分析了当前竞争推理模型的思维过程,发现 " 无效思维 " 现象:在得出正确答案后,需要花 费很大一部分思维过程进行仔细检查,导致推理冗 长且效率低下。

• 我们提出了两个新颖的原则:简洁和充足,并为 LRM 后训练设计了一种基于 GRPO 的方法 LC‑R1,以在简洁和充分之间取得平衡,在压缩整 体序列的同时修剪无效思维。

• 通过综合实验,我们验证了 LC‑R1 在功效和效率 之间取得更好权衡的有效性,并进一步分析了压缩 的深层影响,证明了 LC‑R1 对各种困难的稳健性, 并为未来的工作提供了见解。

表 1:当前最先进的大型推理模型的有效思维率。Nemotron 表示 Llama‑3.3‑Nemotron‑Super‑49b‑v1。结 果显示所有这些模型的 VT 率都很低,突出了 " 无效思维 " 的现象。

| 型          | Avg. | AIME25 | AMC  | GSM8K 型 | MATH500 | 奥林匹克 |
|------------|------|--------|------|---------|---------|------|
| Qwen‑3‑32B | 57.5 | 73.8   | 58.8 | 53.8    | 46.6    | 51.5 |
| QwQ‑32B 型  | 59.2 | 70.8   | 58.2 | 54.1    | 53.1    | 59.6 |
| 深度搜索 ‑R1   | 65.3 | 66.5   | 71.8 | 64.2    | 59.8    | 64.0 |
| Ne         | 60.8 | 62.1   | 64.1 | 63.1    | 56.6    | 58.1 |
| motron     |      |        |      |         |         |      |
### **2. 初步:压缩和有效推理模型**
#### **2.1. 动机:量化冗余推理**

大型推理模型 ( LRM ) 的常见范例涉及最终答案之 前的思考过程(即逐步的基本原理)。虽然对准确性 有效,但我们观察到一贯的低效率:模型通常在思考 过程的早期就得出正确答案,但会继续进行冗长而冗 余的验证步骤。我们将这个后续的、非必要的推理称 为 " 冗余序列 "。

为了正式化这一点,我们定义了有效思维 ( VT ) 率, 这是一个专注于模型思维过程的指标:

$$
VT = \frac{|\text{Tokens in Valid Thinking}|}{|\text{Total tokens in Thinking Process}|}
$$
 (1)

其中 "Valid Thinking" 包括从思考过程开始到首 次得出正确答案的标记。为了自动化这种测量,我们 使用了一个轻量级解析器 LC‑Extractor,其实现细 节在第 4 节中提供。

我们评估了四种最先进的 LRM——Qwen3‑32b (团队, 2025a )、 QwQ‑32b (团队,2025b )、 Deepseek‑R1 ( DeepSeek‑AI 等人,2025 年)和 Llama‑3.3‑nemotron‑super‑ 49b‑v1 ( Bercovich 等 人,2025 年) —— 跨越五个数学基准:AIME25 、 MATH500 、 GSM8K 、 AMC 、奥林匹克工作台。我 们的分析揭示了一个普遍而严重的过度思考问题。如表 1 所示,所有测试的模型都表现出低 VT 率,这表明在找 到解决方案后,它们的大部分计算工作(通常为 35‑45% ) 都花在了冗余推理上。这种普遍的低效率证实了压缩的 巨大潜力,并激励了我们的工作。
#### **2.2. 有效推理的原则**

传统上,对推理模型的评估基于两个支柱:效率(计 算成本,通常与输出长度相近)和效能(正确解决问 题的能力)。然而,简单地缩短输出是一种粗略的方 法,可能会无意中去除 critical

思考步骤。为了创建一个更具针对性的框架,我们通 过引入两个新的互补原则来改进这些概念:

• 简洁通过将重点从通用长度减少转移到具体消除

" 冗余序列 " 来改进效率。虽然传统方法可能仍然 会产生一个包含不必要的双重检查的压缩序列,但简 洁性主张模型在找到正确答案后立即终止其推理过程。

• 充足性是效能的重要保障。它要求,在追求简洁的 过程中,不遗漏任何对得出正确答案至关重要的关键 逻辑步骤。它确保压缩的推理保持完整和逻辑合理。

因此,理想的推理模型必须驾驭这些原则之间的紧张 关系:它应该通过去除所有非必要的思维来最大限度 地简短,但始终保持足够以保证正确性。我们的工作 LC‑R1 明确旨在优化这种平衡。
### **3. LC‑R1: 具有高效推理原理的长度压缩**

在本节中,我们提出了 LC‑R1,这是一种基于 GRPO 的训练后算法,旨在解决 " 无效思维 " 现象 并提高推理效率。在第 2.2 节中引入的简洁和充分原 则的指导下,LC‑R1 采用了一种新颖的双重奖励系统。 该系统将整体简洁性的全局长度奖励与专门删除冗余 推理的有针对性的压缩奖励相结合。LC‑R1 的完整管 道如图 3 和算法 1 所示。
#### **3.1. 问题表述**

让 M 成为模型,让 q 成为给定的查询。输出为 o ∼ M ( q ),其中 o = cat ( R,A ) 由推理部分 R 和答案部 分 A 组成,由标记 </think> 分割,该标记被视为 A 的一 部分。对于推理部分 R,我们将其有效前缀 R′ 表示为

![image.jpeg](..\assets\Optimizing-Length-Compression-in-Large-Reasoning-Models\_page_3_Figure_1.jpeg)

图 3:LC‑R1 训练三阶段管道概述。( 1 ) 有效片段提取:首先,提取器模型处理原始推理轨迹,识别有效思 考部分并生成压缩序列。( 2 ) 奖励计算:接下来,这些压缩序列用于计算我们的双重奖励 —— 长度奖励和压 缩奖励,后者仅作为最终 </think> 代币的奖励或惩罚。然后将这些组合起来计算最终优势。( 3 ) 策略优化: 最后,利用压缩序列和相应的优势计算 GRPO 损失,引导模型朝着更简洁、更高效的推理方向发展。

R 开始,直到与查询 q 对应的正确答案第一次出现。 如果 R 不包含正确答案,则我们定义 R′ = R。我们 定义两个函数如下:

$$
t({R, A}) = R, \quad f({R, A}) = {R', A}
$$
 (2)

函数 t 从输出 o 中提取推理过程 R ,函数 f 提取 简洁推理部分 R′ 并将其与答案 A 连接起来。我们将 o<sup>i</sup> 表示为原始模型输出,将 o ′ <sup>i</sup> = f ( o<sup>i</sup> ) 表示为优 化的压缩输出。

LC‑R1 是一种基于 GRPO 的方法,可以有效地压缩推 理过程。在一个组内,设 C 表示索引集 i ,其中序 列 o<sup>i</sup> 导致与查询 q 对应的正确答案, W 为索引集 j, 其中 o<sup>j</sup> 导致错误答案,总组大小为 G = |C| + |W|。
#### **3.2. 奖励和目标设计**

我们方法的奖励系统由两个核心组成部分组成:用于 减少整体输出长度的长度奖励和针对模型推理的冗余 部分的压缩奖励。

长度奖励。为了压缩模型输出的总长度,我们建议在 GRPO 训练过程中添加长度惩罚。利用 GRPO 的基于 组的抽样,我们可以计算相对长度,以自动调整问题 的难度。我们定义 Length Reward 如下:

$$
r_{i,\text{length}} = \begin{cases} 1 - \frac{|o'_i|}{\max_{j \in C} |o'_j|}, & \text{if } i \in C \\ 0, & \text{if } i \in \mathcal{W} \end{cases} \tag{3}
$$

此公式使用组中正确的压缩序列的最大长度作为归一 化器。最终奖励将其与格式和准确性的基本奖励相结 合,并通过减去组均值进行归一化,按照 Liu 等人 ( 2025 年)获得无偏梯度:

$$
\tilde{r}_i = r_{i,\text{base}} + \alpha \cdot r_{i,\text{length}} \tag{4}
$$

$$
r_{i,\text{combine}} = \tilde{r}_i - \text{mean}(\{\tilde{r}_j\}_{j=1}^G) \tag{5}
$$

哪里

$$
r_{i,\text{base}} = r_{i,\text{format}} + r_{i,\text{accuracy}} \tag{6}
$$

根据之前的工作,ri,format 和 ri,accuracy 是二元视图, 以判断模型是否将其思维过程置于 <think> 和 </think> 之间,以及样本是否

算法 1 LC‑R1:R1 样式模型的长度压缩 输入:初始策略模型 π<sup>θ</sup> 、压缩函数 f (·)、任务提示符 D 、超参数 α,β, µ 输出:经过训练的策略模型 π<sup>θ</sup> *1*:对于步骤 = 1, *. . .* ,M *do* 2: 对批处理 D<sup>b</sup> from D 进行采样 3: 更新旧策略模型 πθold ← π<sup>θ</sup> 4: 每个问题的示例 G 输出 {oi} G <sup>i</sup>=1 ∼ πθold ( ·|q ) q ∈ D<sup>b</sup> 5: 将压缩应用于所有输出:o ′ <sup>i</sup> ← f (o<sup>i</sup> ) 6: 计算组合奖励 ri,combine (方程 5 )和压缩奖励 ri,compress (方程 11 ) 7: Compute Token 级优势 Aˆi,<sup>t</sup> 对于每个压缩输出 o ′ <sup>i</sup> (式 10 ) 8: 对于迭代 = 1, . . . ,µ do 9: 通过最大化目标 JGRPO (方程 7 )来更新策略模型 π<sup>θ</sup> 10: **end 为 11:结束 12: return**π<sup>θ</sup>

分别导致与 Math‑Verify1 验证的查询相对应的正确 答案。α 是控制 Length Reward 权重的超参数。

压缩奖励。对于原始的 GRPO 方法,损失计算基于模 型自己的抽样结果。为了驱动模型在获得正确答案以 实现简洁时终止思考过程,我们修改了 GRPO 目标, 如下所示:

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q)} \tag{7}
$$
\n
$$
\left[ \frac{1}{\sum_{i=1}^G |o'_i|} \sum_{i=1}^G \sum_{t=1}^{|o'_i|} \left\{ \min[R_t(\theta) \cdot \hat{A}_{i,t}, \operatorname{clip}(R_t(\theta), \Delta_{i,t}) \right] \right\}
$$
\n
$$
1 - \epsilon, 1 + \epsilon \cdot \hat{A}_{i,t} \right] - \beta D_{\text{KL}}(\pi_{\theta}(\cdot|q) || \pi_{\text{ref}}(\cdot|q)) \left\}
$$

哪里

$$
\mathbb{D}_{\text{KL}}(\pi_{\theta} \mid \pi_{ref}) = \frac{\pi_{ref}(o'_i|q)}{\pi_{\theta}(o'_i|q)} - \log \frac{\pi_{ref}(o'_i|q)}{\pi_{\theta}(o'_i|q)} - 1 \quad (8)
$$

$$
o'_{i} = f(o_{i}), \quad R_{t}(\theta) = \frac{\pi_{\theta}(o'_{i,t}|q, o'_{i,(9)
$$

我们对标准 GRPO 目标的关键修改是,损失是在压缩 轨迹 o ′ <sup>i</sup> 上计算的,而不是在原始的完整轨迹 o<sup>i</sup> 上计 算的。我们定义 ˆ token 级优势 Ai,<sup>t</sup> 如下:

Aˆ i,t = ri,combine + γ ·I(o ′ i,t = </think>)· ri,compress (10) 哪里

$$
r_{i, \text{compress}} = \begin{cases} 1 - \frac{|t(o'_i)|}{|t(o_i)|}, & \text{if } i \in \mathcal{C} \& \text{ans}(q) \in t(o'_i) \\ -1, & \text{if } i \in \mathcal{C} \& \text{ans}(q) \notin t(o'_i) \\ 0, & \text{if } i \in \mathcal{W} \end{cases} \tag{11}
$$

1https://github.com/huggingface/Math‑Verify

设 ans ( q ) 为给定查询 q 的真实答案。在这种设置 中,我们把重点放在思考过程中,当得到正确答案时 ( o ′ <sup>i</sup> 结尾)引导模型输出 </think> token,实现对冗 长 token 的压缩,符合简洁的原则。我们只给这个代 币额外的奖励,避免不必要地强调其他代币,使训练 过程更加高效和稳定。我们将奖励定义为冗余序列的 一部分,由 1− |t(o ′ <sup>i</sup>) | <sup>|</sup><sup>t</sup> <sup>o</sup><sup>i</sup> <sup>|</sup>( ) 表示,表示压缩前后 序列之间的效率差异。hyperparameterγ 缩放此加 成。

基于 Adequate 原则,模型应进行充分的推理过程, 避免以精度降低为代价的开销压缩。因此,如果模型 在找到正确答案之前终止了推理,我们会对标记 </ think> 施加很大的惩罚 ( ‑1 ),这阻止了有害的过 度压缩并为训练过程提供了鲁棒性。

为了进一步验证我们方法的有效性,我们遵循 DAPO ( Yu et al., 2025 ) 来计算一组中所有标记 的反对意见,而不是在单个序列中平均标记重复,这 消除了原始 GRPO 方法对短正确序列和长错误序列的 偏好,有助于验证我们方法的有效性。
## **4. 实验**
## **4.1. 实验设置**

主干模型。我们选择了两个具有代表性的推理模型: DeepSeek‑R1‑Distill‑Qwen‑7B/1.5B ( DeepSeek‑AI et al., 2025 ) 作为我们的主干模型, 它们在数学和编码推理任务上表现出了强大的性能。

表 2:七个基准中所有方法的准确度(上图)和序列长度(下图)。AVG 显示与 Base 模型相比,准确率和长度的相 对变化 ( + increase, ‑ decrease )。GPQA‑D 表示 GPQA‑Diamond 基准测试,LCB 表示 LiveCodeBench 上 的 pass@10 分数。VT 代表有效思维比率。对于每列,表现最好的分数以粗体标记,第二好的分数用下划线标记。

| 方法                       |                    | AIME25 MATH500 GSM8K |                   | 奥运会                        | AMC                | GPQA‑D 系列          | LCB                | Avg                  | VT     |
|--------------------------|--------------------|----------------------|-------------------|----------------------------|--------------------|--------------------|--------------------|----------------------|--------|
| DeepSeek‑R1‑ 蒸馏 ‑Qwen‑7B |                    |                      |                   |                            |                    |                    |                    |                      |        |
| Base                     | 37.7<br>(11007)    | 92.6<br>(3832)       | 91.6<br>(1842)    | 59.7<br>(7342)             | 81.2<br>(6715)     | 46.6<br>(6508)     | 68.8<br>(8878)     | –                    | 58.72% |
| SFT                      | 36.6<br>(9457)     | 90.2<br>(2497)       | 91.9<br>(946)     | 56.0<br>(6329)             | 78.7<br>(5231)     | 39.8<br>(8217)     | 67.3<br>(8739)     | –4.46%<br>(–15.54%)  | 95.64% |
| DPO                      | 36.9<br>(9718)     | 91.4<br>(2277)       | 90.3<br>(980)     | 56.2<br>(6338)             | 78.6<br>(5122)     | 37.2<br>(8109)     | 66.9<br>(8755)     | –5.26%<br>(–16.18%)  | 96.34% |
| O1‑ 修枝剪                  | 35.0<br>(8263)     | 91.5<br>(2268)       | 91.1<br>(1012)    | 59.6<br>(4712)             | 77.1<br>(4510)     | 45.5<br>(5012)     | 66.7<br>(5901)     | –2.79%<br>(–33.71%)  | 69.30% |
| ThinkPrune<br>西梅         | 38.0<br>(9309)     | 93.1<br>(3253)       | 91.2<br>(1546)    | 60.8<br>(6225)             | 82.7<br>(5510)     | 50.3<br>(6508)     | 67.8<br>(7180)     | +1.58%<br>(–14.13%)  | 77.16% |
| SFT+O1‑Pruner            | 35.5<br>(9466)     | 91.0<br>(2245)       | 89.7<br>(920)     | 56.0<br>(5807)             | 76.6<br>(5133)     | 43.9<br>(6425)     | 66.8<br>(7267)     | –4.31%<br>(–24.19%)  | 85.22% |
| LC‑R1 (我们的)              | 36.2<br>(7150)     | 90.4<br>(1568)       | 88.1<br>(450)     | 58.7<br>(4041)             | 79.1<br>(3453)     | 47.2<br>(4604)     | 69.0<br>(6059)     | –1.84%<br>(–46.32%)  | 97.14% |
|                          |                    |                      |                   | DeepSeek‑R1‑ 蒸馏 ‑Qwen‑1.5B |                    |                    |                    |                      |        |
| Base                     | 22.8<br>(12129)    | 83.7<br>(4869)       | 83.4<br>(2294)    | 44.2<br>(9258)             | 61.2<br>(8696)     | 34.5<br>(8516)     | 43.1<br>(10120)    | –                    | 56.06% |
| SFT                      | 20.5<br>(10639)    | 81.4<br>(3045)       | 81.3<br>(1134)    | 42.7<br>(7637)             | 59.7<br>(6608)     | 22.4<br>(10217)    | 39.8<br>(10597)    | –9.13%<br>(–16.74%)  | 95.54% |
| DPO                      | 19.4<br>(10316)    | 79.0<br>(2749)       | 80.9<br>(855)     | 41.1<br>(6544)             | 56.7<br>(5912)     | 19.8<br>(9438)     | 39.2<br>(10287)    | –12.79%<br>(–24.30%) | 98.38% |
| O1‑ 修枝剪                  | 23.2<br>(8731)     | 84.3<br>(2913)       | 82.7<br>(1162)    | 47.1<br>(5960)             | 65.1<br>(5131)     | 32.1<br>(6173)     | 42.5<br>(7305)     | +0.89%<br>(–35.64%)  | 78.20% |
| ThinkPrune<br>西梅         | 24.1<br>(7960)     | 84.5<br>(3518)       | 84.1<br>(1690)    | 44.9<br>(6250)             | 63.4<br>(5897)     | 33.6<br>(5576)     | 42.7<br>(7226)     | +1.31%<br>(–30.89%)  | 65.62% |
| SFT+O1‑Pruner            | 17.5<br>(9075)     | 80.2<br>(2769)       | 81.5<br>(919)     | 40.0<br>(6411)             | 58.7<br>(5553)     | 25.0<br>(7410)     | 39.4<br>(8488)     | –11.34%<br>(–32.04%) | 91.38% |
| LC‑R1 (我们的)              | 21.2<br>(<br>6434) | 82.5<br>(<br>2233)   | 82.7<br>(<br>841) | 43.2<br>(<br>4333)         | 61.7<br>(<br>3947) | 33.6<br>(<br>4489) | 42.4<br>(<br>5722) | –2.14%<br>(–51.86%)  | 98.64% |

LC 提取器。为了准确识别和提取有效的推理部分,我 们开发了一个专门的解析器来实现方程 2 中提到的提 取函数 f ,称为 LC‑Extractor。我们对 Qwen2.5‑3B‑Instruct 进行了微调,因为它重量轻且 易于运行。附录 B 中提供了详细的实验设置。

数据。我们使用了一个混合难度的数据集,将过去的 AIME 竞赛问题与 MATH 数据集以大约 1:2 的比例 组合在一起,以创建 2500 个训练样本。这种方法使 模型能够学习不同难度问题的长度压缩。

评估。我们在 7 个数据集上测试了我们的模型性能, 包括 AIME25 、 MATH500 、 GSM8K 、 AMC 、

OlympiadBench 、 GPQA‑Diamond 和 LiveCodeBench,跨数学、通用和代码任务,全面评 估推理的效率。我们使用平均 Pass@1 作为主要指标。 对于每个测试,我们采样 N 次,设置 top‑p = 0.95 和温度 = 0.7。对于 AIME25,我们设置 N = 64, 而对于其他测试集,我们设置 N = 8。我们将最大长 度设置为 16384。此外,我们计算了每个基准与基本 模型相比精度和长度的平均波动率,其公式如下:

$$
Avg_{acc} = mean_{i=1}^{7} \left\{ \frac{Acc_{i}^{model} - Acc_{i}^{base}}{Acc_{i}^{base}} \right\}
$$
 (12)

$$
Avg_{len} = mean_{i=1}^{7} \left\{ \frac{Len_{i}^{model} - Len_{i}^{base}}{Len_{i}^{base}} \right\}
$$
 (13)

表 3:长度奖励和压缩奖励对压缩过程贡献的消融研究。该研究揭示了它们的次优性能,对它们进行验证对高效 推理做出了很大贡献。

| 方法                         |                |                | AIME25 MATH500 GSM8K 奥林匹克竞赛 |                | AMC            | GPQA‑D 系列      | LCB            | Avg                 | VT     |
|----------------------------|----------------|----------------|-----------------------------|----------------|----------------|----------------|----------------|---------------------|--------|
| DeepSeek‑R1‑ 蒸馏 ‑Qwen‑7B   |                |                |                             |                |                |                |                |                     |        |
| LC‑R1 (我们的)                | 36.2<br>(7150) | 90.4<br>(1568) | 88.1<br>(450)               | 58.7<br>(4041) | 79.1<br>(3453) | 47.2<br>(4604) | 69.0<br>(6059) | –1.84%<br>(–46.32%) | 97.14% |
| 无 L‑reward                 | 36.1<br>(9309) | 91.3<br>(2316) | 90.6<br>(696)               | 59.4<br>(5779) | 79.0<br>(5021) | 45.9<br>(6273) | 68.0<br>(8023) | –1.80%<br>(–25.28%) | 93.16% |
| 无 C 奖励                     | 37.6<br>(8738) | 92.9<br>(2498) | 91.1<br>(1012)              | 59.1<br>(5344) | 80.5<br>(4741) | 48.9<br>(5727) | 68.5<br>(6893) | +0.31%<br>(–27.35%) | 72.24% |
| DeepSeek‑R1‑ 蒸馏 ‑Qwen‑1.5B |                |                |                             |                |                |                |                |                     |        |
| LC‑R1 (我们的)                | 21.2<br>(6434) | 82.5<br>(2233) | 82.7<br>(841)               | 43.2<br>(4333) | 61.7<br>(3947) | 33.6<br>(4489) | 42.4<br>(5722) | –2.14%<br>(–51.86%) | 98.64% |
| 无 L‑reward                 | 21.3<br>(7061) | 81.2<br>(2270) | 83.3<br>(754)               | 43.4<br>(5024) | 62.2<br>(4478) | 30.6<br>(5021) | 41.9<br>(6378) | –3.42%<br>(–47.79%) | 95.16% |
| 无 C 奖励                     | 21.9<br>(7988) | 83.2<br>(2965) | 84.1<br>(1160)              | 44.0<br>(5363) | 63.4<br>(5192) | 30.1<br>(5847) | 43.7<br>(6874) | –1.70%<br>(–38.35%) | 71.10% |

我们还测试了每个模型的 VT 以评估思维过程的简洁 性,以研究这些方法减轻 " 无效思维 " 现象的能力。 我们在五个数学基准上测试 VT 并计算平均值,以便 于从数学问题的思考过程中提取标准和格式化的正确 答案。
#### **4.2. 基线**

监督微调 ( SFT )。受 OVERTHINK ( Chen et al., 2024 ) 的启发,OVERTHINK 提议仅使用初始正确解决 方案进行微调,我们通过从自生成的输出中删除 Redundant Sequence 来构建一个包含 5000 个样本的 SFT 数据集。
#### **直接偏好优化 ( DPO )( Rafailov 等人,**

2023 ) . 我们从 MATH 数据集创建了一个包含 5000 个样本的偏好数据集,其中最短的正确答案被视 为 " 选择 " 响应,最长的正确答案被视为 " 被拒绝 " 响应。此 DPO 训练应用于 SFT 优化模型。

O1 修剪器( Luo 等人,2025b )。一种类似 PPO 的离线 微调方法,可在保持性能的同时显著压缩 CoT 长度。我们 使用 MATHdataset 中的 10000 个样本来遵循其方法。

ThinkPrune‑3K ( Hou 等人,2025 年)。一种强 化学习方法,它使用长度截断奖励进行多阶段压缩。 我们复制了 ThinkPrune‑3k 变体,据报道该变体非 常有效,但准确性略有下降。

SFT + O1‑Pruner. 为了更好地理解同时压缩思维过程 和修剪整体序列的效果,我们还将 SFT 和 O1 Pruner 相结合的两阶段训练方法进行了比较。
#### **4.3. 实验结果**

LC‑R1 优于其他方法,具有竞争力的性能和更少的代 币。如表 2 所示,在 7B 型号上,LC‑R1 的平均长度 减少了 46.32%,大大高于所有其他基线,平均准确 率仅下降了 1.84%。同样,在 1.5B 型号上,它的长 度减少了 51.86%,精度降低了 2.14%。这种效率似 乎并不能保证它的通用性,因为与其他高压缩方法相 比,它在 GPQA‑Diamond 和 LiveCodeBench 等分 布外 ( OOD ) 基准测试中表现出更强大的性能。图 2 显示了我们的方法通过最大压缩比和可忽略不计的 精度下降来实现更有利的功效 ‑ 效率权衡。与 O1‑Pruner ( ~70‑78% ) 和 ThinkPrune ( ~66‑ 77% ) 等其他方法相比,LC‑R1 还实现了显着更高 的 VT 率(超过 77% ),证明了我们方法的卓越效率。

结合长度和压缩奖励为推理带来了卓越的效率。我们 对长度奖励 ( L‑reward ) 和压缩奖励 ( C‑reward ) 的消融研究,如表 3 所示,揭示了它们 的关键互补关系。分析表明,虽然每个成分单独产生 有竞争力的结果 —— 将它们置于性能与压缩效率的帕 累托前沿附近 —— 但将它们结合起来可以实现更理想 的平衡。具体来说,单独使用 L‑reward 可实现 sig‑

![](..\assets\Optimizing-Length-Compression-in-Large-Reasoning-Models\_page_7_Figure_1.jpeg)

图 4:LC‑R1 压缩方法对 AIME25 基准测试的影响。左图:Pass@k 分数表明,与原始模型相比,LC‑R1 模型保持了 有竞争力的性能,从而保留了模型的潜力。右图:对 Deepseek‑R1‑Distill‑Qwen‑7B 的每个问题进行分析,结果表 明 LC‑R1 实现了相似的 Pass@1 精度,同时在不同难度的问题中保持一致的令牌压缩率,展示了普遍的压缩效果。

压缩良好,但 VT 率较低。相反,单独的 C‑reward 通过精确消除冗余来确保高 VT,但整体压缩有限。 我们的全 LC‑R1 方法成功地整合了这些优势,实现了 最高的压缩效率和最高的 VT 率,同时保持了相当的 准确性,证明了这两种奖励之间的协同作用对于实现 最大的推理效率是必不可少的。

SFT 显示泛化的限制。虽然 SFT 实现了非常高的 VT 率(超过 95% ),但其有效性是肤浅的。该模型 的性能在 OOD 基准测试中崩溃,表明它只是过度拟 合了训练数据的结构简洁性,而不是学习可推广的高 效推理策略。混合 SFT+O1‑Pruner 方法的糟糕性能 进一步表明,现成技术的简单组合是不够的。这些发 现强调了 LC‑R1 等基于 RL 的方法的优越性,它可以 培养更强大和真正有效的推理技能。
## **5. 压缩影响分析**

压缩不会影响勘探功能。为了研究压缩对模型解决问 题潜力的更深层次影响,我们在最大长度 = 32, 768 的 AIME25 上采样了 256 次,并在压缩前后对两 个模型进行了 pass@k 分测试。图 4 (左)中的结果 揭示了一个关键现象:在 AIME25 数据集上从 k = 1 到 128 的整个 Pass@k 评估范围内,我们的 LC‑R1 方 法压缩的模型的性能曲线几乎与原始模型的性能曲线 完美重叠。这一结果有力地表明,该模型通过多次尝 试找到正确解决方案的探索能力不会受到

压缩。它表明,被修剪的 " 无效思考 " 片段确实是 多余的,它们的删除不会削弱模型的潜在知识或创造 性的问题解决潜力。

在不同的问题难度下,压缩保持一致。为了在微观层 面上分析我们方法的行为,我们在 AIME25 基准测试 中绘制了每个问题的 pass@1 精度与原始模型的代币 消耗量(图 4 (右))。该图揭示了一个明确的难度 范围,其中需要来自基本模型的更多代币的问题通常 对应于较低的 pass@1 分数。至关重要的是,LC‑R1 在整个范围内应用了均匀且显著的压缩比,每个问题 的结果(即成功或失败)与基本模型的结果保持非常 一致。这提供了强有力的证据,证明 LC‑R1 作为一个 稳健且与难度无关的效率层,成功地简化了推理过程, 而不会改变模型针对任何特定问题的核心问题解决逻 辑。
### **6. 总结**

在本文中,我们解决了当前 LRM 中存在的 " 无效思 维 " 现象,即他们倾向于在得出正确答案后仔细检查 他们的工作。为了解决这个问题,我们引入了简洁和 充分的原则,并提出了 LC‑R1,这是一种基于 RL 的 后训练方法,采用双重奖励系统,压缩整体序列长度 并自发修剪冗余序列。大量实验表明,LC‑R1 实现了 更有利的功效 ‑ 效率权衡。在进一步分析中,LC‑R1 不会降低模型的探索能力,并且压缩效应在不同难度 的问题中保持稳健。
### **影响声明**

在本文中,我们解决了 " 无效思维 " 现象,这是大 型推理模型中效率低下的关键来源,它们在得出正确 答案后进行不必要的验证。我们介绍了 LC‑R1,这是 一种新颖的训练后方法,具有双重奖励系统,既鼓励 整体简洁,又鼓励具体消除这种冗余。我们的实验表 明,与现有方法相比,LC‑R1 在性能和效率之间实现 了更有利的权衡。虽然由于计算限制,我们当前的验 证侧重于高达 7B 规模的模型,但这项工作为开发计 算更节俭的 LRM 提供了一条行之有效的途径。通过 提高高级 AI 推理的效率,我们希望使这些强大的工 具更具可扩展性,并适用于更广泛的应用程序。
## **引用**

AaronJaech 、 AdamKalai 、 Adam Lerer 、 Adam Richardson 、 Ahmed El‑Kishky 、 Aiden Low 、 Alec Helyar 、 Aleksander Madry 、 Alex Beutel 、 Alex Carney 等人。 Openai o1 系统卡。arXiv 预印本 arXiv:2412.16720,2024 年。

DeepSeek‑AI, 郭大亚, 杨德健, 张浩伟, 宋俊晓, 张若宇, 徐润欣, 朱启浩, 马世荣, 王培义, 毕晓, 张晓康, 俞兴 凯, 吴宇, 吴志峰, 苟志斌, 邵志宏, 李卓树, 高子怡 ..., 张振 .Deepseek‑r1:通过强化学习激励 llms 中的推理能力, 2025 年。URL https://arxiv.org/ abs/2501.12948。

Jason Wei 、 Xuezhi Wang 、 Dale Schuurmans 、 Maarten Bosma 、 Brian Ichter 、 Fei Xia 、 Ed Chi 、 Quoc Le 和 Denny 周。在大型语言模型中提示引发推理的思维链,2023 年。 URL https://arxiv.org/abs/2201.11903。

孙浩祥、闵颖倩、陈志鹏、赵鑫、刘正、王中原、方磊和温 继荣。挑战推理的界限:大型语言模型的奥林匹克级别数学 基准,2025 年。URL https://:// arxiv.org/abs/2503.21380.

Alex Gu 、 Baptiste Rozière 、 Hugh Leather 、 Armando Solar‑Lezama 、 Gabriel Synnaeve 和 Sida I. Wang。Cruxeval: 代码推理、理解和执行的基准。arXivpreprint arXiv:2401.03065, 2024.

陈启光、秦立波、刘金浩、彭登云、关建南、王鹏、胡梦康、 周宇航、高特、车万祥。迈向推理时代:推理大型语言模型 的长链思维调查,2025 年。URL https://arxiv.org/abs/2503.09567。

Pranjal Aggarwal 和 Sean Welleck。L1:用强化学习控制推理模型 思考的时间,2025 年。URL https://arxiv.org/abs/2503.04697。

陈星宇, 徐家豪, 梁天, 何志伟, 庞建辉, 俞典, 宋林 峰, 刘秋志, 周梦飞, 张卓生, 等 . 不要对 2+ 3= 想那么 多吗?关于类似 O1 的 LLM 的过度思考。arXiv 预印本 arXiv:2412.21187,2024 年。

遂洋, 庄玉能, 王冠初, 张佳木, 张天义, 袁佳义, 刘 宏毅, 温安德, 陈汉杰, 胡霞, et al. 停止过度思考:关 于大型语言模型有效推理的调查。arXiv 预印本 arXiv: 2503.16419, 2025。

亚历杭德罗 · 夸德隆, 李大成, 马文杰, 王兴耀, 王一川, 庄思源, 刘澍, 路易斯 · 加斯帕 · 施罗德, 夏田, 毛欢志, et al. 过度思考的 危险: Ex a‑ a‑ aposition the reasoning‑action didicment in agentic tasks.arXiv 预印本 arXiv:2502.08235,2025 年。

Haotian Luo, Haiying He, Yibo Wang, Jinluan Yang, Rui Liu, Naiqiang Tan, Xiaochun Cao, Dacheng Tao, and LiShen.Adar1: 通过双层自适应推理优化从 long‑cot 到 hybrid‑cot。arXiv 预印本 arXiv:2504.21659, 2025a。

沈毅、张健、黄洁云、石淑明、张文静、闫江泽、王宁、王 凯和连世国。Dast:大型推理模型的难度适应慢思维,

2025 年。URL https://arxiv.org/abs/2503.04472。

侯柏如、张洋、纪家宝、刘玉剑、钱开志、 Jacob Andreas 和 Shiyu Chang。Thinkprune:通过强化学习修剪 llms 的长 思维链。arXiv 预印本 arXiv:2504.01296, 2025。

Haotian Luo, Li Shen, Haiying He, Yibo Wang, Shiwei Liu, Wei Li, Naiqiang Tan, Xiaochun Cao, 和 Dacheng Tao.O1‑pruner: 长度协调微调,用于类似 o1 的推理修剪,2025 b. URL https://arxiv.org/abs/2501.12570。

Kimi Team, 杜鞍钢, 高博飞, 邢博伟, 江昌久, 陈成, 李成, 肖晨军, 杜晨庄, 廖崇华, 唐楚宁, 王聪聪, 张 德浩, 袁恩明, 卢恩哲, 唐凤祥, 宋洪, 魏光达, 赖国 坤, 郭海青, 朱韩, 鼎浩, 胡浩, 杨浩, 张浩天, 姚浩 天, 赵浩天, 卢浩宇, 李浩泽, 于浩珍, 高洪成, 郑华 斌, 袁欢, 陈佳, 郭建航, 苏建林, 王建洲, 赵杰, 张 进, 刘景元, 闫俊杰, 吴俊艳, 石立东, 叶玲, 于隆辉, 董梦南, 张尼奥, 马宁辰, 潘启伟, 龚曲程, 刘绍伟, 马胜玲, 魏树鹏, 曹思涵, 黄思英, 江涛, 高伟浩, 熊 伟民, 何伟然, 黄伟晓, 徐伟欣, 吴文浩, 何文阳, 习 ‑ 昂辉, 贾贤庆, 吴兴哲, 徐欣然, 祖欣兴, 周欣宇, 潘 雪海, Y. Charles, 李洋, 胡阳阳, 刘洋洋, 陈艳如, 王叶杰, 刘一波, 秦义道, 刘一峰, 杨英, 包一平, 杜 玉伦,

Qwen 团队。Qwen3,2025 年 4 月 a. URLhttps://qwenlm。 github.io/blog/qwen3/。

Qwen 团队。Qwq‑32b:拥抱强化学习的力量,2025 年 3 月 b. URL https://qwenlm.github.io/ blog/qwq‑32b/。

Akhiad Bercovich, Itay Levy, Izik Golan, Mohammad Dabbah, Ran El‑Yaniv, Omri Puny, Ido Galil, Zach Moshe, Tomer Ronen, Najeeb Nabwani, Ido Shahaf, Oren Tropp, Ehud Karpas, Ran Zilberstein, JiaqiZeng, Soumye Singhal, Alexander Bukharin, ...

Yian Zhang 和 Chris Alexiuk。Llama‑nemotron:高效研究 模型,2025 年。URL https://arxiv.org/abs/2505。00949.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. 理解类 r1‑zero 训练:批判性视角,2025 年。URL https:// arxiv.org/abs/2503.20783。

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. Dapo: 一个大规模的开源强化学习 系统。arXiv 预印本 arXiv:2503.14476,2025 年。

拉斐尔 · 拉法洛夫、阿奇特 · 夏尔马、埃里克 · 米切尔、克 里斯托弗 · 曼宁、斯特凡诺 · 埃尔蒙和切尔西 · 芬恩。直接 偏好优化:您的语言模型秘密地是一个奖励模型。神经信息 处理系统进展,36:53728– 53741,2023 年。

打开人工智能。聊天。https://openai.com/o1/,2024 年。

谷歌。双子座 2.5pro。https://cloud.google.com/ vertex‑ai/generative‑ai/docs/models/gemini/ 2‑5‑pro,2025a。

Marah Abdin, Sahaj Agarwal, Ahmed Awadallah, Vidhisha Bal‑ achandran, Harkirat Behl, Lingjiao Chen, Gustavo de Rosa, S uriya Gunasekar, Mojan Javaheripi, Neel Joshi, PieroKauff‑ mann, Yash Lara, Caio César Teodoro Mendes, Arindam Mitra, Besmira Nushi, Dimitris Papailiopoulos, Olli Saarikivi, Shital Shah, Vaishnavi Shrivastava, Vibhav Vineet, Yue Wu, Safoora Yousefi, and Guoqing Zheng.Phi‑4 推理技术报告,2025 年。URL https://arxiv.org/abs/2504.21318。

邵志宏、王培义、朱启浩、徐润欣、宋俊晓、毕小、张浩伟、 张明川、李耀基、吴宇和郭大亚。Deepseekmath:在开 放语言模型中突破数学推理的极限,2024 年。URL https:// arxiv.org/abs/2402.03300。

刘佳伟和张玲明。Code‑r1:为具有可靠奖励的代码复制 r1。2025.

胡健、 Jason Klein Liu 和 Wei Shen。Reinforce++:一种对提示和 奖励模型都具有鲁棒性的高效 rlhf 算法,2025 年。URL https://arxiv.org/abs/2501.03262。

马欣茵, 万光年, 余润鹏, 方功凡, 王欣超 .Cot‑valve:长度 可压缩的思维链调整,2025a. URL https://arxiv.org/abs/2502.09601。

达曼 · 阿罗拉 ( Daman Arora ) 和安德里亚 · 扎内特 ( Andrea Zanette )。训练语言模型进行有效推理,2025 年。 URLhttps://arxiv.org/abs/2502。04463.

Simon A. Aytes 、 Jinheon Baek 和 Sung Ju Hwang。 思想素描:具有自适应认知启发素描的高效 llm 推理, 2025 年。URL https://arxiv.org/abs/ 2503.05179。

韩庭旭、王振霆、方春荣、赵诗宇、马诗青和陈振宇。代币预 算感知 llm 推理。arXiv 预印本 arXiv:2412.18547,2024 年。

马文杰、何静轩、查理 · 斯内尔、泰勒 · 格里格斯、闵世元 和马泰 · 扎哈里亚。推理模型可以不思考就有效,2025b. URL https://arxiv.org/abs/2504。09858.

Qwen 团队。Qwen2.5:基金会模型的一方,2024 年 9 月。 URL https://qwenlm.github.io/blog/qwen2.5/。

谷歌。双子座 2.5 闪光灯。https://developers.googleblog。 com/en/start‑building‑with‑gemini‑25‑flash/,2025b。

Ping Yu 、 Jing Xu 、 Jason Weston 和 Ilia Kulikov。将系统 2 蒸 馏到系统 1 中,2024 年。URLhttps://arxiv.org/abs/ 2407.06023。

医学人工智能国际会议。第 23 届医学人工智能国际会议 ( AIME2025 )。https://aime25.aimedicine.info/。访 问时间:2025 年 6 月 10 日。

亨特 · 莱特曼、维内特 · 科萨拉朱、尤里 · 布尔达、哈里森 · 爱德华兹、鲍文 · 贝克、泰迪 · 李、扬 · 莱克、约翰 · 舒尔 曼、伊利亚 · 萨茨克和卡尔 · 科布。让我们一步一步地验证。 在第十二届学习表征国际会议中,2023 年。

Karl Cobbe 、 Vineet Kosaraju 、 Mohammad Bavarian 、 Mark Chen 、 Heewoo Jun 、 Lukasz Kaiser 、 Matthias Plappert 、 JerryTworek 、 Jacob Hilton 、 Reiichiro Nakano 、 Christopher Hesse 和 John Schulman。训练验证者解决数学单词 问题。arXiv 预印本 arXiv:2110.14168,2021 年。

美国数学协会。美国数学竞赛 ( amc )。https://maa‑amc.org/ student‑programs/amc/。访问时间:2025‑06‑10.

David Rein 、 Betty Li Hou 、 Asa Cooper Stickland 、 Jackson Petty 、 Richard Yuanzhe Pang 、 Julien Dirani 、 Julian Michael 和 Samuel R Bowman。Gpqa:研究生水平的 google 证明 q&a 基准测试。在 2024 年第一届语言建模会议上。

Naman Jain 、 King Han 、 Alex Gu 、温定李、闫凡佳、张天军、王思 达、 Armando Solar‑Lezama 、 Koushik Sen 和 Ion Stoica。 Livecodebench:对代码的大型语言模型进行整体和无污染的评估。 arXiv 预印本 arXiv:2403.07974,2024 年。doi: 10.48550/arXiv.2403.07974.

Leandro von Werra 、 Younes Belkada 、 Lewis Tunstall 、 Edward Beeching 、 Tristan Thrush 、 Nathan Lambert 、 Shengyi Huang 、 Kashif Rasul 和 Quentin Gallouédec。Trl: 变压器加固学习。https://github.com/huggingface/trl,2020 年。
## **A. 相关工作**

用于推理的强化学习。强化学习 ( RL ) 已成为大型语言模型 ( LLM ) 训练后阶段的关键技术,在增强其推 理能力方面显示出巨大的潜力。该领域的一项里程碑式工作是 OpenAI 的 o1 模型( OpenAI,2024 年)。作 为 RL 用于推理的首个大规模应用,o1 在发布时就实现了最先进的推理能力。不久之后,Deepseek‑R1 ( DeepSeek‑AI et al., 2025 )作为第一个与 o1 性能相匹配的开源模型被推出,极大地推动了基于 RL 的推理 技术的发展和普及。这种技术方法导致了许多强大的大型推理模型 ( LRM ) 的出现,例如 Gemini 2.5 ( Google,2025a )、 QwQ ( Team,2025b )和 Phi‑4 ( Abdin et al.,2025 )。最近,具有可验证奖励的 强化学习 ( RLVR ) 已被证明是显着改善模型分辨率的有效方法

高效推理。虽然精心推理更有可能得出正确答案,但其冗长的思维过程会显著增加时间和计算成本,这种现象 称为过度思考( Chen et al., 2024 )。为了缓解这个问题,研究人员从不同的角度提出了各种解决方案( Sui et al., 2025 )。这些方法可以大致分类。第一类涉及通过长度控制直接限制推理过程的冗余,典型的例 子是 CoT‑Valve (马 et al., 2025a ) 和 L1 ( Aggarwal and Welleck, 2025 )。第二类方法侧重于使模 型能够根据查询的难度调整其推理深度。例如,Adar1 ( Luo et al., 2025a ) 和 DAST ( Shen et al., 2025 ) 构建了偏好数据集来训练模型以生成与问题复杂性相匹配的推理序列。第三类将效率考虑因素集成到强 化学习框架中
## **B. LC 萃取仪的详细信息**

我们训练 Qwen‑2.5‑3B‑Instruct ( Team, 2024 ) 作为 LC‑Extractor 模型。我们从 MATH 数据集中构建 了一个由 5,000<问题、思维过程、答案 > 三元组组成的数据集,并使用 Gemini‑2.5‑Flash ( Google, 2025b ) 确定第一个正确标记的位置,然后进行严格的基于规则的过滤。然后,我们通过使用这些精选样本进 行 2 个 epoch 的训练,将这些知识提炼成一个更小的模型。LC‑Extractor 的有效性在 100 个样品的测试集上 进行了验证,通过人工评估确认了 98 个样本,如图 6 所示。LC‑Extractor 模型被图 5 中的提示激活。
## **C. 详细的实验设置**
## **C.1. 模型**

我们在论文中使用了 DeepSeek‑R1 ( DeepSeek‑AI et al., 2025 )、 Qwen3‑32B ( Team, 2025a )、 QwQ‑32B ( Team, 2025b )、 Llama‑3.3‑ Nemotrom‑Super‑49B‑V1 ( Bercovich et al.,2025 )、 Distill‑Qwen‑7B 、 Distill‑Qwen‑1.5B ( Yu et al.,2024 ) 和 Qwen‑2.5‑ 3B‑Instruct ( Team, 2024 ) 模型。我们介绍它们的许可证和主要特性如下:

• DeepSeek‑R1 的。开源 671 B→37B MoEreasoning 模型主要通过强化学习进行训练,在支持 128K 令牌上下文的 同时引发自我验证、反思和冗长的思维链跟踪 ; 它仅使用公共数据在数学 / 代码基准测试上匹配专有的 O1。

• Qwen3‑32B.32.8 B 参数的第三代 Qwen 模型可在 " 思考 " 和 " 非思考 " 模式之间切换,在单个密集检 查点中提供最先进的推理、多语言聊天和高达 131 K 的上下文。
# **提示提取答案前缀**

您是 Qwen,由阿里云创建。你是一个乐于助人的助手。
# **指令:**

Extract Answer . 前缀 你会得到一个问题、 一个思考过程及其 Ground Truth 答案 **您的任务:**

1. 从头开始仔细阅读《思考过程》。2. 找到显示 Ground Truth Answer 的第一句 话。3. 复制从思考过程开始到包括该句子的所有内容。4. 重要提示:请勿在该句子 后包含任何文本。
# **例:**
- 问题:什么是 1 + 1?
- 思考过程:好的,我需要解决 1 + 1。这得到 2。让我再检查一次 —— 是的,是 2。
- 基本实况答案: 2.
- 预期输出:好的,我需要解决 1 + 1。这样就得到 2。
## **提供的输入:**
- 问题: <Problem>
- 思考过程: < 思考过程 >
- 真值答案:< 真值答案 >
## **您的输出:**

"Thinking Process (思考过程) " 的前缀,末尾为 Ground Truth。
## 图 5:我们提取答案前缀的提示。

• QwQ‑32B. 使用 SFT + RL 改进的中型 Qwen 推理变体 ; 提供显式 <think> 跟踪、 131 K 上下文和 DeepSeek‑R1– levelaccuracyonhard 评估。

• Llama‑3.3‑Nemotrom‑Super‑49B‑V1。NVIDIA 的 NAS‑ 修剪的 Llama‑3.3‑70B 的 49 B 衍生物,经过推 理、 RAG 和工具调用的后训练 ; 将 128 K 环境与单 H100 部署效率相结合,适用于成本敏感型生产。

• deepseek‑r1‑distill‑qwen‑7b. 从 DeepSeek‑R1 提炼出来的 7 B 密集检查点到 Qwen2.5 主干上,将小模型 MATH‑500 pass1 推高到 92% 以上,并超越了 o1‑minionseveralreasoningsuites,同时保持了笔记本电脑友好性。

• Deepseek‑R1‑Distill‑Qwen‑1.5B. 从 R1 中提炼出来的超紧凑 1.5 B 模型,保留了思维链,并在 MATH‑500 上实现了 83.9% 的通过率 1,为边缘和移动部署带来了有竞争力的分析能力。

• Qwen‑2.5‑3B‑Instruct. 一个 3.09 B 指令调整模型,具有 128 K 上下文、增强的编码 / 数学技能和多语言支 持,旨在为下游任务提供轻量级但可控的聊天基础。
## **C.2. 数据集**

我们在论文中对 AIME25 (医学人工智能国际会议)、 MATH500 ( Lightman 等人,2023 年)、 GSM8K ( Cobbe 等人,2021 年)、 OlympiadBench ( Sun 等人,2025 年)、 AMC (美国数学协会)、 GPQA Diamond ( Rein 等人,2024 年)和 LiveCodeBench ( Jain 等人,2024 年)基准进行了基准测试。我们

|                                           | Deploy :                                                                                                                                                                                                                                                                  |
|-------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Navigation<br>Displaying Entry: 1 of 2851 | <b>JSON Entry Review Interface</b>                                                                                                                                                                                                                                        |
| $Next \Box$<br><b>El</b> Previous         | Entry 1                                                                                                                                                                                                                                                                   |
|                                           | Question:                                                                                                                                                                                                                                                                 |
|                                           | A line is parameterized by $\binom{x}{y} = \binom{0}{-2} + t\binom{3}{4}$ . A second line is parameterized by $\binom{x}{y} = \binom{-8}{12} + u\binom{1}{3}$ . If $\theta$ is the acute angle formed by the two lines, then find $\cos \theta$ .                         |
|                                           | Solution:                                                                                                                                                                                                                                                                 |
|                                           | $\frac{3}{\sqrt{10}}$                                                                                                                                                                                                                                                     |
|                                           | <b>Reasoning Prefix:</b>                                                                                                                                                                                                                                                  |
|                                           | Okay, so I have this problem where I need to find the cosine of the acute angle between two lines. Both lines are given in parametric form, which I remember is like starting at a point and then moving in a certain directio<br>direction vectors, right? Let me think. |
|                                           | First, let me write down the parameterizations to make it clear.                                                                                                                                                                                                          |
|                                           | The first line is:                                                                                                                                                                                                                                                        |
|                                           | $\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 0 \\ -2 \end{pmatrix} + t \begin{pmatrix} 3 \\ 4 \end{pmatrix}$                                                                                                                                                   |
|                                           | So, the direction vector for the first line is $\begin{pmatrix} 3 \\ 4 \end{pmatrix}$ . I'll call this vector v.                                                                                                                                                          |
|                                           | The second line is:                                                                                                                                                                                                                                                       |
|                                           | $\binom{x}{y}$ = $\binom{-8}{12}$ + u $\binom{1}{3}$                                                                                                                                                                                                                      |
|                                           | So, the direction vector for the second line is $\begin{pmatrix} 1 \\ 2 \end{pmatrix}$ . I'll call this vector w.                                                                                                                                                         |
|                                           | I remember that the angle 0 between two vectors can be found using the dot product formula:                                                                                                                                                                               |
|                                           | $\cos\theta = \frac{\mathbf{v}\cdot\mathbf{w}}{  \mathbf{v}   \,   \mathbf{w}  }$                                                                                                                                                                                         |
|                                           | But since we're dealing with lines, not vectors from the origin, the direction vectors are what matter here. So, I can use the direction vectors v and w to compute this.                                                                                                 |
|                                           | First, I need to compute the dot product of v and w. The dot product is calculated by multiplying the corresponding components and then adding them up.                                                                                                                   |
|                                           | So, $v \cdot w = (3)(1) + (4)(3) = 3 + 12 = 15.$                                                                                                                                                                                                                          |
|                                           | Next, I need to find the magnitudes of v and w.                                                                                                                                                                                                                           |
|                                           | Starting with v:                                                                                                                                                                                                                                                          |
|                                           | $  w   = \sqrt{2^2 + 4^2} = \sqrt{0 + 16} = \sqrt{25} = 5$                                                                                                                                                                                                                |

图 6:用于评估 LC‑Extratcor 的注释工具。

按如下方式介绍它们:

• 艾梅 25. 一个包含 30 个问题的基准,从 25 年的美国数学邀请赛试卷中提炼出来。每个项目都是一个三位数 的简答题,用于探讨高中代数、几何、组合学。

• MATH500。一个包含 500 个问题的评估切片,涵盖了原始 MATH 竞赛语料库的整个学科广度。它在难度等 级和主题之间取得平衡,可作为高级高中和早期本科生数学推理的严格标准,而没有完整 12k 题集的运行负担。

• GSM8K 的。被广泛采用的 Grade‑School Math 8K 基准测试,包含 1,319 个日常单词问题。GSM8K 需要多 步算术和常识,仍然是评估会话数学任务的思维链质量的事实标准。

• 奥运会。大约 3 个国家和国际数学奥林匹克问题的精选集合。该基准主要是证明式或数字答案挑战,衡量大学 预科最高水平的创造性、非常规推理。

• AMC 的。总共 83 个来自 10/12 的美国数学竞赛。跨越 2000 年至 2024 年,它提供了基础中学数学的纵向基准。

• GPQA 钻石。该基准测试包含 198 道研究生水平的 Google 证明多项选择题,需要深厚的领域专业知识和多步推理, 由纽约大学、 CohereAI 和 Anthropic 的研究人员策划 ; 使用准确性作为指标,在闭卷和开卷设置中进行评估。

• LiveCodeBench 的非动态、无污染的编码基准测试最初托管从 LeetCode 、 AtCoder 和 CodeForces 收集的 511 个问题(版本 v2 ),由加州大学伯克利分校、麻省理工学院和康奈尔大学的研究人员设计,用于使用 Pass @K 全面评估 LLM 的代码生成、执行和测试预测能力。
## **C.3. 设置**

我们使用了混合难度的数据集,将过去的 AIME 竞赛问题与 MATH 数据集以大约 1:2 的比例相结合,以创建 2500 个训练数据。我们使用 Trl ( von Werra et al., 2020 ) 框架来训练模型。这两个模型都使用 4 \* A800‑80G GPU 进行训练,超参数如表 4 所示。

表 4:LC‑R1 训练的超参数。

| 超参数<br>R1‑<br>蒸馏 | ‑Qwen‑7B<br>R1‑<br>蒸馏 | ‑Qwen‑1.5B |
|------------------|-----------------------|------------|
| cutoff len_      | 8192                  | 8192       |
| 批量大小 _           | 32                    | 32         |
| learning_rate    | 3.0E‑6                | 2.0E‑6     |
| num_train_epochs | 1.0                   | 1.0        |
| α                | 1.0                   | 1.0        |
| β                | 0.04                  | 0.04       |
| γ                | 1.0                   | 1.0        |
| num_generations  | 6                     | 8          |
| ϵ                | 0.2                   | 0.2        |

基线设置。我们将 LC‑R1 与 5 个基线进行比较 ——SFT 、 DPO 、 O1‑Pruner 、 ThinkPrune 、 SFT+

O1‑Pruner。最后一个 hybrid 方法与每个方法共享相同的设置,因此我们给出了前四个方法的设置。

• SFT 的。我们通过提取有效的思维过程来构建训练数据集,以重建在 MATH 数据集上自行采样的序列的简 洁版本。我们设置了 cutoff\_len=8192, epoch=1, learning\_rate = 3.0e‑6, max\_samples=5000。

• DPO 的。我们通过在 MATHdataset 上采样 8 次来构建偏好训练数据集,并选择最长的样本为负,最短的 样本为正。我们设置了 cutoff\_len=8192 、 epoch=2 、 learning\_rate = 5e‑6 、 max\_samples =5000。

• O1‑ 修剪器。我们使用给定的 python 脚本来构建权重训练数据集,其中包含 cutoff\_len=4096 、 epoch =2 、learning\_rate = 2.0e‑7 、 max\_samples = 10000。

• ThinkPrune‑3K。我们在 ThinkPrune‑length3000 数据集上重现了训练过程,大小为 2470。我们设置 了 cutoff\_len=8192, epoch=2, learning\_rate = 2.0e‑6,num\_generations=8, batch\_size=32。
## **D. 案例研究**

我们做了一些案例研究,将 LC‑R1 与 O1‑Pruner ( Luo etal.,2025b )方法和基本模型进行比较。这些案例 研究如图 7 和图 8 所示。

![jpeg](..\assets\Optimizing-Length-Compression-in-Large-Reasoning-Models\_page_14_Figure_1.jpeg)

图 7:LC‑R1 和 O1‑Pruner 比较的案例研究。

![jpeg](..\assets\Optimizing-Length-Compression-in-Large-Reasoning-Models\_page_15_Figure_1.jpeg)

图 8:LC‑R1 与原始模型比较的案例研究。