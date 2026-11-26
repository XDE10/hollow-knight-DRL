# 组员环境配置与排障指南

本指南帮助小组成员在全新电脑（Windows / Linux）上 **5 分钟内**跑通战斗姐妹冠军模型。

---

## 1. 前置条件

1. **正版《空洞骑士》（Hollow Knight 1.5.78）**（通过 Steam 安装，任意盘符均可）。
2. **安装 Modding API**：
   - 下载并打开 [Lumafly](https://themulhima.github.io/Lumafly/)（原 Scarab），它会自动识别你的《空洞骑士》路径并安装 Modding API（安装后 `hollow_knight_Data/Managed/` 下会出现 `MMHOOK_Assembly-CSharp.dll`）。
3. **准备 3 号存档（Godhome 诸神堂）**：
   - 启动游戏确认 **3 号存档槽（Save Slot 3）** 已进入神居诸神堂，解锁「战斗姐妹（Sisters of Battle）」，建议保持 9 格血、纯粹骨钉（1×）。
   - 如果你希望使用其他存档槽，运行时传入 `--save-slot 1`（或 `2`/`4`）即可。

---

## 2. 安装 Python 依赖与 HKRLEnvMod

```bash
# 1. 安装 Python 包
pip install -e "python[dev]"

# 2. 一键安装预编译 HKRLEnvMod.dll（无需装 .NET SDK）
python scripts/install_mod.py
```

* `install_mod.py` 会通过 Windows 注册表 + Steam `libraryfolders.vdf`（或 Linux `~/.local/share/Steam`）自动查找游戏位置；
* 如果你的游戏安装在非 Steam 目录，可直接传参：
  ```bash
  python scripts/install_mod.py --game-root "/path/to/Hollow Knight"
  ```
* 也可以设置环境变量 `HKRL_GAME_ROOT` 省去每次传参。

---

## 3. 启动演示

```bash
# Windows PowerShell
.\scripts\windows\play_sisters.ps1

# 或直接使用 Python（跨平台）
python scripts/sisters/play.py --episodes 1
```

首次运行会自动从 GitHub Release 下载 `models/sisters_champion/sisters_champion.bin`（`17.38 MB`）并校验 SHA-256；如果因为网络原因下载慢，也可以手动从 [Releases 页面](https://github.com/XDE10/hollow-knight-DRL/releases/tag/v1.0.0-sisters-champion) 下载 `sisters_champion.bin` 放到 `models/sisters_champion/` 目录后再运行。

---

## 4. 从源码自行编译 C# Mod（可选，仅修改 Mod 时需要）

如果不需要改 C# 代码，**直接用第 2 步的预编译 DLL 即可**。如需自行编译：

1. 安装 [.NET 8 SDK](https://dotnet.microsoft.com/download) 与 `flatc 23.5.26`（可用 `conda env create -f environment-mod-build.yml`）；
2. 生成 C# FlatBuffers 绑定并编译：
   ```bash
   flatc --csharp -o mod/HKRLEnvMod/Schema schema/hkrl.fbs
   dotnet build mod/HKRLEnvMod/HKRLEnvMod.csproj -c Release \
     -p:HollowKnightManaged="/path/to/Hollow Knight/hollow_knight_Data/Managed"
   ```
3. 安装刚编好的本地 DLL：
   ```bash
   python scripts/install_mod.py --dll mod/HKRLEnvMod/bin/Release/HKRLEnvMod.dll --force
   ```

---

## 5. 常见报错速查

- **`Could not auto-detect Hollow Knight via Steam`**：Steam 未安装在标准路径或非 Steam 版本，传 `--game-root "/path/to/Hollow Knight"` 即可。
- **`Hollow Knight did not listen on 127.0.0.1:5555`**：
  1. 确认已通过 Lumafly 安装 Modding API（检查 `Managed/MMHOOK_Assembly-CSharp.dll` 是否存在）；
  2. 确认 Steam 客户端已在后台登录运行；
  3. 若 5555 端口被占用，改用 `--port 5566`。
- **提示下载权重失败 / 超时**：从 [GitHub Releases](https://github.com/XDE10/hollow-knight-DRL/releases/tag/v1.0.0-sisters-champion) 手动下载 `sisters_champion.bin` 与 `HKRLEnvMod.dll` 放到对应目录即可离线运行。
