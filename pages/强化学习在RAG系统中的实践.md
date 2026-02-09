- #### 思考与传统RAG系统的区别：
- 1. 传统RAG系统，整个工作流是固定的，检索到的上下文也是固定的，不会因为问题不同而改变
  2. 强化学习将RAG技术的各个组件（如：请求改写，上下文检索）的使用 是根据奖励反馈由模型自己决策的
-
- **强化学习的目标是学习一个策略 π，以最大化预期累积奖励**。
  在RL+RAG这个场景中，我们的目标是通过生成与真实答案相似的回答来最大化奖励值。较高的奖励值表明生成的回答与预期答案更加一致。
  
  在RAG系统的上下文中，强化学习可以用于：
  编写强化学习算法的先是定义三件事：
- **状态（State）**
- **动作空间（Action Space）**
- **奖励（Reward）**
-
- #### 状态（State），我们定义了如下属性：
- `query`：原始用户查询。
- `context_chunks`：从知识库中检索到的上下文片段。
- `rewritten_query`：原始查询的重新表述版本。
- `previous_responses`：之前生成的回答列表。
- `previous_rewards`：之前行动所获得的奖励列表。
-
- #### 动作空间（Action Space），我们定义了四个动作：
- `rewrite_query`：改写原始查询
- `expand_context`：检索额外的上下文片段
- `filter_context`：移除不相关的上下文片段
- `generate_response`：根据当前查询和上下文生成响应
-
-
- #### 奖励（Reward），我们定义如下：
  它将评价生成的响应的质量。我们将基于生成的响应与可能的真实答案集合之间的文本相似度来评价。
  具体如下：
  1. 建立了历史上问答数据集（问题，回答）
  2. 召回当前问题语义最相似(类似：bgm这类向量化模型)的三个问题 （问题A，回答A）， （问题B，回答B）， （问题C，回答C）
  3. ABC三个答案最为标准答案，对比可以给当前生成响应一个得分sorce
  4. 最终的得分由加权求和得到一个平均的sorce
  这个sorce就可以作为奖励（Reward）指导我们的强化学习过程。
-
- #### 策略网络，我们定义如下
- 我们需要创建一个策略网络，根据当前状态选择一个动作。
  策略网络是一个函数，它以当前状态和动作空间作为输入，并根据状态返回所选动作。
  策略网络可以使用简单的启发式方法根据当前状态选择动作。
  
  我们定义策略网络如下的工作方式如下：
  1. 20% 概率随机选择一个动作
  2. 80% 概率按如下策略选择动作
- 如果没有之前的回答，优先选择重写查询。
- 如果存在之前的回答，但奖励较低，则尝试扩展上下文。
- 如果上下文包含过多片段，尝试对上下文进行筛选。
- 否则，生成一个回答。
-
-
- ##### 训练过程
-
  <pre>
  <code class="language-python">
  # Python 伪代码
  进行num_episodes次采样，就是模拟num_episodes次
  for episode in range(params["num_episodes"]):
      # 每个次采样，最多走10步，以防止无限循环。
      for step in range(10):
          # 进行每一步,RL决策进行什么操作：比如：重写请求，重新结合上下文去查询
          state, action, reward, response = rl_step(state, action_space)
          # 如果奖励是历史最佳就保存
          if response:
              episode_reward = reward  # Update the episode reward
              # Track the best response and reward
              if reward > best_reward:
                  best_reward = reward
                  best_response = response
              break  # Exit the loop as the episode ends
  	
          print(f"Episode {episode}: Reward = {episode_reward:.4f}, Actions = {episode_actions}")
  
      # 记录下来本次采样的动作序列和奖励序列
      rewards_history.append(episode_reward)
      actions_history.append(episode_actions)
  </code>
  </pre>
-
-
- #### 结果展示：
- ===============
  Query: 我有点发热，咳嗽，可能是什么问题呀
- ic| f"basic_rag_pipeline relevant_chunks={relevant_chunks}":
- ("basic_rag_pipeline relevant_chunks=[
- ' 肺炎  有症状  发热咳痰和胸膜性胸痛 ',
- ' 流感  有症状  高热寒战肌肉酸痛咽喉痛和干咳 ',
- ' 肺炎  有病史  最近有病毒感染或免疫抑制 ']")
- Simple RAG Output:
- 您可能患有肺炎或流感。肺炎的症状包括发热、咳嗽和胸膜性胸痛，而流感的症状包括高热、寒战、肌肉酸痛、咽喉痛和干咳。建议您及时就医，以便得到正确的诊断和治疗。
-
- RL-enhanced RAG Output:
- 肺炎和支气管炎都是可能的原因，需要进一步诊断以确定具体病症，然后给予相应治疗。对于细菌性肺炎，可以使用抗生素；对于支气管炎，则通常使用止咳药和祛痰药。同时，如果症状严重，可能需要支持性护理。
-