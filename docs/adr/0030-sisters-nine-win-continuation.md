# ADR 0030. 从能通关的双策略模型继续强化学习

已有冻结神经策略在 12 场独立原速原伤实战中完整通关 9 场，无伤 0 场。本组在第三次失利后结束。旧版本、开局网络、三姐妹阶段的两套按键网络和原有选择规则已保存，可作为新模型的起点。[旧权重与逐场成绩](../../models/sisters-native-iql/timing-learned-options/evaluation.json)

在原有选择规则上增加一个有 1,857 个参数的网络。每 0.8 游戏秒，它根据骑士、多个螳螂和飞镖的当前状态，以及原模型的特征，调整接下来由哪套按键网络行动。初始化时在三场历史实战的 12,448 次操作上完整复现旧版。这种改法保留了原模型已经掌握的操作。两套按键网络给出相同动作时，改变选择不会改变按键。[旧版复现记录](../../models/sisters-native-iql/timing-ninewin-continuation/parent-equivalence.json)

Colab 的 L4 在 9 场真实随机分配控制器的历史训练对局上训练双 Q 网络、状态价值网络和策略。Q 网络通过实战回报学习，并预测采取不同选择后骑士、螳螂和飞镖的运动。计算训练动作优势时按整场交叉拟合。在 1,242 个训练时刻中，新策略改变了 149 次选择。三场此前分析过的历史留出对局里共改变 60 次实际按键，独立 Python 计算和游戏插件重放均一致。这些数据只能说明改动的范围，胜负由下一组全新实战决定。[训练与权重](../../models/sisters-native-iql/timing-ninewin-continuation/training.json) · [按键变化](../../models/sisters-native-iql/timing-ninewin-continuation/neural-reference.json) · [游戏重放](../../models/sisters-native-iql/timing-ninewin-continuation/runtime-parity.json)

三场历史轨迹中，三姐妹阶段能从骑士血条观察到 22 次扣血。其中只有 2 次在扣血前 0.72 秒内出现不同按键。原版独立实战的 3 场败局最后一次观察到的剩余敌方总血量分别是 59、465、158。既要保护已有输出，也要让训练能改变危险时刻的实际按键。仅凭固定轨迹不能判断换键后能否躲过攻击。[扣血前的按键](../../models/sisters-native-iql/timing-ninewin-continuation/change-before-hp.json) · [原版实战](../../models/sisters-native-iql/timing-learned-options/evaluation.json)

若本次离线更新未过线，在线 PPO 从已保留的 9 胜版本重新出发。用新对局采样，保存每次选择的模型概率、实际随机数、执行的按键、伤害来源和终局。每 8 场按实际游戏时间计算优势并更新策略、价值网络。开发时比较单窗原速和两倍速的实测吞吐和每游戏秒的按键次数，正式评估始终使用原速原伤。[在线学习入口](../../scripts/train_sisters_champion_ppo.py) · [两轮验收条件](../../configs/experiments/sisters_acceptance.json)

复现实验时从仓库根目录运行以下命令，输出放在一个新目录。原实验使用 CUDA。重新训练的数值结果可能随硬件和软件版本变化。

```powershell
$models = 'models/sisters-native-iql'
python scripts/train_sisters_champion_router.py --device cuda --updates 900 --data "$models/timing-neural-blocks/development.zip" --experiment "$models/timing-neural-blocks/experiment.json" --parent "$models/timing-learned-options/world_iql.bin" --learner "$models/timing-preferences/learner.pt" --output runs/nine-win-reproduction
```
