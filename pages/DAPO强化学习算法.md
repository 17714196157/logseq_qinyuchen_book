- 1. 移除KL散度
  在RLHF场景中，强化学习的目标是对齐模型的输出，使其输出符合人类的偏好（包括输出的语法语序、有害信息屏蔽，使其说话方式更像人），这个时候不应该偏离原始模型太远，需要KL散度来限制，但是在强化学习训练长尾推理模型时，需要模型更加自由的去探索，模型分布可能与初始模型的分布显著偏离，KL散度的限制是没有必要的。
-
- 2. Clip-Higher(更高的上限)
  在PPO和GPPO中对于新旧策略概率比进行裁剪，限定在固定范围内（1-epsilon, 1+epsilon，epsilon一般取0.2），防止模型更新幅度过大。
- 对于两个动作（token），假设其概率分别为0.9（高概率）和0.01（低概率），那么在更新之后，两个token的最大概率分别为1.2*0.9=1.08，0.01*1.2=0.012，这意味着对于高概率的token，其受到的约束反而更小，低概率token受到的约束更大，想实现概率的显著增加非常困难，限制了模型探索的多样性。
- DAPO对上限和下载剪的范围解耦，增加上限剪的范围，给低概率token更多的探索空间。
-
- 3. 动态采样
  GPPO中，通过组内奖励的均值和标准差计算优势，如果一个样本的组内奖励全为0或者全为1，这个时候的优势为0，零优势导致策略更新时没有梯度，无法优化模型，样本效率不高。
- DAPO的做法是在训练前不断进行采样，直到一个batch内的所有样本的组内奖励既不全为0也不全为1。
-
- 4. Token级策略梯度损失
- 这张图片展示了两种策略梯度方法的损失函数：GRPO（Gradient-based PPO）和 DAPO（Dynamic Averaging PPO）。这些方法用于强化学习中的策略优化。
- GRPO的损失函数定义为：
  \[ J_{\text{GRPO}}(\theta) = \mathbb{E}_{(q, a) \sim \mathcal{D}_r(\theta)} \left[ \frac{1}{G} \sum_{t=1}^{G} \frac{1}{|n_t|} \sum_{i=1}^{|n_t|} \left( \min \left( r_{t,i}(\theta) \hat{A}_{t,i}, \text{clip} \left( r_{t,i}(\theta), 1 - \epsilon, 1 + \epsilon \right) \hat{A}_{t,i} \right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{old}}) \right) \right] \]
  
  其中，\(\theta\) 是策略参数，\(\mathcal{D}_r(\theta)\) 是由策略 \(\theta\) 生成的数据分布，\(G\) 是样本长度，\(n_t\) 是第 \(t\) 个时间步的样本数量，\(r_{t,i}(\theta)\) 是第 \(t\) 步第 \(i\) 个样本的奖励比率，\(\hat{A}_{t,i}\) 是优势函数，\(\epsilon\) 是裁剪参数，\(\beta\) 是KL散度的权重，\(D_{\text{KL}}(\pi_\theta \| \pi_{\text{old}})\) 是策略 \(\pi_\theta\) 和旧策略 \(\pi_{\text{old}}\) 之间的KL散度。
- DAPO的损失函数定义为：
  \[ J_{\text{DAPO}}(\theta) = \mathbb{E}_{(q, a) \sim \mathcal{D}_r(\theta)} \left[ \frac{1}{\sum_{t=1}^{G} |n_t|} \sum_{t=1}^{G} \sum_{i=1}^{|n_t|} \min \left( r_{t,i}(\theta) \hat{A}_{t,i}, \text{clip} \left( r_{t,i}(\theta), 1 - \epsilon_{\text{low}}, 1 + \epsilon_{\text{high}} \right) \hat{A}_{t,i} \right) \right] \]
  \[ \text{s.t. } 0 < \left| \left\{ \alpha_i \mid \text{is\_equivalent}(n, \alpha_i) \right\} \right| < G. \]
  
  DAPO通过修改GRPO的损失函数，对组内所有token求平均，使得长样本因为拥有更多的token，其整体对最终的损失贡献更大。
-