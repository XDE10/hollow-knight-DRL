# ADR 0024. Learn when to continue an existing focus input

The frozen world-assisted IQL actor selects every opening action and initiates every focus attempt. In some development bouts a continuous focus input consumed 22–26 SOUL without increasing HP. Other bouts completed a one-mask heal after roughly 1.4 game seconds and a 33-SOUL decrease. These measurements use the native player clock. The engine also emits heal requests when no HP change occurs.

We trained two small action-conditioned outcome models on previously executed keys. One predicts the outcome after another focus input and the other predicts the outcome after release under the recorded follow-on behavior. Targets cover the next 0.52 game seconds. They measure positive and negative changes in native HP, boss damage in complete event intervals, and SOUL spent. The respective utility weights are 2, −3, 0.45/21 and −0.12/33. Fits give equal total weight to each distinct focus attempt within each action. The comparison uses current HP, SOUL, input duration and threat geometry. The teacher's proposed keys are excluded from the inputs and targets.

During the trio phase, if the parent requests release, the fitted return difference can keep the ordinary focus key down. The additional time on one attempt is capped at 0.52 seconds. A gap in the input tape or a change in native HP ends eligibility. The game records how many attempts were extended, how many inputs changed, and HP gained or lost within 0.52 seconds of an intervention. These counts allow a win with no intervention to be identified.

The two observed action groups were selected by their respective behavior policies. Differences in danger, timing and subsequent inputs can affect their returns. Replaying the control decisions checks the executable. Independent, frozen-policy fights measure wins and actual mask changes. The original T model is retained.

Run the fit from the repository root.

```sh
m=models/sisters-native-iql
python scripts/train_sisters_focus_returns.py \
  --archive "$m/timing-winning-guidance/development.zip" \
  --archive "$m/timing-neural-blocks/development.zip" \
  --archive "$m/timing-randomized-descent/development.zip" \
  --parent "$m/timing-executed-movement/world_iql.bin" \
  --out runs/reproduced-sisters-focus
```
