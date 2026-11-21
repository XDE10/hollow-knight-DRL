# ADR 0031. 用真实对局更新躲避时的按键

9 胜版的开局在独立评估中 12 场都打过第一只。此前从该版本继续训练时，保持两套按键网络固定，只改三姐妹阶段由谁操作。在三场历史轨迹里，22 次可见扣血中只有 2 次在扣血前 0.72 秒实际改变了按键。一个危险时刻如果两套网络给出的按键相同，切换网络就没有效果。下一次在线更新直接训练这两套网络输出按键的最后一层。开局、实体编码器和原有的选择规则保留。[9 胜版实战](../../models/sisters-native-iql/timing-learned-options/evaluation.json) · [固定轨迹上的变化](../../models/sisters-native-iql/timing-ninewin-continuation/change-before-hp.json)

## Colab 上的训练结果

从已有开发对局取出 155 场、39,719 个输入时刻，保留原有的按场训练和留出划分。对记录中的实际按键，模型学习随后 0.8 游戏秒内的对敌伤害、骑士失血和终局回报。另一个输出预测随后观测到的骑士与按距离排列的螳螂、飞镖状态变化。后续按键沿用轨迹中实际发生的操作。学生独立操控且随后损血的片段，还可以利用同一状态上记录的教师按键做辅助监督。[压缩后的训练和留出片段](../../models/sisters-native-iql/timing-ninewin-primitive-heads/development-windows.zip) · [训练入口](../../scripts/train_sisters_primitive_heads.py)

L4 训练产生了新权重。留出数据中可对照教师的受击前时刻共 320 个，原模型和更新模型都匹配 152 个。两个价值网络附带的运动预测平均绝对误差分别约为 0.430 和 0.429，保持位置不动的基线约为 0.488 和 0.489。按预先登记的条件，本次离线按键更新结束，权重留作对照。在线 PPO 使用原 9 胜版的按键权重，并沿用这次训练出的价值估计。[留出结果](../../models/sisters-native-iql/timing-ninewin-primitive-heads/policy-stage.json) · [模型和训练记录](../../models/sisters-native-iql/timing-ninewin-primitive-heads/training.json)

## 在线强化学习

训练时，骑士在三姐妹阶段接近螳螂或飞镖、且能受伤时，至多每 0.4 游戏秒采样一次实际按键。只改变这一次输入，下一次仍根据新观察作决定。原首选键继续获得大部分概率。当有其他可用键时，至少 10% 的概率分给不同按键。正式推理始终取神经策略的首选键。格式对照使用一场先前录下的固定轨迹，4,047 个输入中有 140 次采样机会，17 次采到不同按键。游戏插件和 Python 对照的实际按键全部一致。[原生按键重放](../../models/sisters-native-iql/timing-ninewin-primitive-heads/native-exploration-replay.json)

每 8 场新实战后，根据逐次记录的采样概率、实际奖励和终局计算按游戏时间折扣的优势。PPO 更新直接写入两套网络的原生按键分数，并在普通时刻限制相对原版的偏移。预登记的 24 场训练内包含单窗原速与两倍速的比较。若两倍速的每游戏秒决策频率低于原速的 95%，或实际吞吐未提高 12%，用原速继续，并用新的原速对局替换两场倍速数据。冻结的评估使用不同的种子和原速原伤。[训练参数和种子](../../models/sisters-native-iql/timing-ninewin-primitive-heads/online-plan.json) · [PPO 实现](../../scripts/train_sisters_primitive_ppo.py)
