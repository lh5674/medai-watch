# MedAI Watch

> 黑客松研究原型。该系统只监测图像分类模型在受控扰动下的预测与 Grad-CAM 数值稳定性，不诊断患者、不推荐治疗，也不代表任何临床有效性。

## 1. 项目介绍

MedAI Watch 接收一张灰度或 RGB 图像，运行 PyTorch 分类模型，生成 Grad-CAM，并分别施加高斯模糊、高斯噪声和亮度调整。系统比较扰动前后的类别、置信度和解释相似度，计算透明的原型风险分数，生成中文报告并将记录保存到 SQLite。

当前项目没有附带真实医学模型权重。默认运行固定权重、无随机预测的 `DemoConvNet`，类别为“演示类别A/演示类别B”。界面和报告会始终显示：

> 模拟演示案例，不代表真实模型或临床结果。

## 2. 使用场景

- 黑客松演示模型稳定性监测流程；
- 研究阶段比较模型对轻微图像变化的敏感程度；
- 展示预测、解释、风险规则和历史记录之间的数据链路；
- 为后续接入已验证的内部模型提供工程适配层。

本项目不适用于临床诊断、治疗决策、分诊或患者风险判断。

## 3. 系统架构

```text
PIL 图像
  ├─ 图像质量检查
  ├─ 原始模型预测 ── 原始 Grad-CAM
  └─ 模糊 / 噪声 / 亮度扰动
       └─ 扰动预测 ── 原类别目标 Grad-CAM
            └─ 类别稳定性 + 置信度变化 + Cosine/SSIM
                 └─ 透明风险引擎
                      ├─ 中文确定性报告
                      ├─ SQLite + PNG 资产
                      └─ Streamlit 分析页 / 仪表盘
```

`agent/monitoring_agent.py` 只负责编排。模型、质量、扰动、Grad-CAM、相似度、风险、报告、资产和数据库分别位于独立服务中，不使用 LangChain、外部 LLM 或多智能体运行框架。

## 4. 技术栈

- Python 3.11+
- Streamlit
- PyTorch / torchvision
- OpenCV
- Pillow / NumPy / Pandas
- scikit-image
- Plotly
- SQLite
- pytest
- 本地自定义 Grad-CAM

## 5. 安装步骤

项目推荐放在 E 盘：

```powershell
Set-Location E:\medai-watch
.\scripts\setup_e_drive.ps1
```

该脚本使用 `E:\medai-watch\.python314\python.exe` 创建虚拟环境，并把
`TEMP`、`TMP` 和 `PIP_CACHE_DIR` 指向项目内的 `.runtime` 与
`.pip-cache`。本机当前的隔离运行环境位于 `E:\medai-watch\.venv`。
项目依赖、模型、数据库、生成资产和项目安装缓存均保存在 E 盘。

等价的依赖安装核心命令是：

```powershell
E:\medai-watch\.venv\Scripts\python.exe -m pip install -r E:\medai-watch\requirements.txt
```

## 6. 启动命令

```powershell
Set-Location E:\medai-watch
.\scripts\run_app.ps1
```

等价的直接启动命令为：

```powershell
E:\medai-watch\.venv\Scripts\python.exe -m streamlit run E:\medai-watch\app.py
```

打开 Streamlit 显示的本地地址。左侧配置扰动参数，上传图像或选择合成演示图，然后点击“开始安全监控分析”。“监控仪表盘”位于 Streamlit 页面导航中。

## 7. 模型权重接入

默认检查点位置：

```text
E:\medai-watch\models\model.pth
```

在 `config/settings.py` 的 `ModelSettings` 中修改：

```python
ModelSettings(
    mode="checkpoint",
    architecture="resnet18",
    checkpoint_path=PROJECT_ROOT / "models" / "model.pth",
    class_names=("类别一", "类别二"),
    model_version="your-model-v1",
    target_layer="layer4.1.conv2",
)
```

当前适配器支持 `resnet18`、`resnet34` 和 `densenet121` 的严格 `state_dict` 加载，也接受包含 `state_dict` 或 `model_state_dict` 的字典。权重、架构或类别数不匹配会直接报错，不会静默回退到模拟输出。

如需其他架构，在 `services/predictor.py` 的 `_build_torchvision_model()` 增加适配器，并同时提供对应测试。

## 8. 类别名称配置

类别顺序配置位于：

```text
config/settings.py → ModelSettings.class_names
```

顺序必须与模型输出 logits 的列顺序完全一致。真实模型模式至少需要两个非空类别名称。切勿用未经确认的标签顺序解释模型结果。

## 9. 图像预处理配置

训练图像尺寸和归一化参数位于：

```text
config/settings.py → PreprocessingSettings
```

字段包括：

- `image_size`：`(高度, 宽度)`；
- `mean`：RGB 三通道训练均值；
- `std`：RGB 三通道训练标准差。

接入真实模型时必须替换为训练阶段的精确参数。输入会统一转换为 RGB，再执行 resize、tensor 转换和 normalize。

## 10. Grad-CAM 层配置

配置位置：

```text
config/settings.py → ModelSettings.target_layer
```

填写模型中的点路径，例如 `layer4.1.conv2`。设置为 `None` 时，系统自动选择最后一个 `torch.nn.Conv2d`。路径不存在或目标不是卷积层会给出明确错误。所有扰动图像的 Grad-CAM 都使用原始预测类别作为目标，以保证比较口径一致。

Grad-CAM 数值或区域不证明医学正确性。

## 11. 风险评分说明

原型公式：

```text
risk_score =
    0.25 × (1 - 原始置信度)
  + 0.35 × (类别变化测试数 / 测试总数)
  + 0.25 × (1 - 平均解释相似度)
  + 0.15 × (1 - 图像质量分 / 100)
```

单次扰动的解释相似度为 Grad-CAM 余弦相似度与 SSIM 的平均值。最终结果转换为 0–100：

- `< 30`：通过；
- `30–<60`：谨慎使用；
- `>= 60`：建议人工复核。

权重和阈值位于 `config/settings.py → RiskSettings`。

> 该评分仅用于黑客松研究原型，权重和阈值尚未经过临床验证。

## 12. 黑客松演示流程

1. 启动应用，确认模型状态为“模拟演示模式”或已加载的真实检查点模式；
2. 选择三个合成演示案例之一，或上传测试图像；
3. 合成案例会自动载入推荐扰动预设，也可以在侧栏手动调整；
4. 点击“开始安全监控分析”；
5. 查看原始预测、概率和 Grad-CAM；
6. 在扰动标签页比较类别、置信度与解释相似度；
7. 查看风险组成、决策、中文报告和 JSON/CSV 导出；
8. 打开“监控仪表盘”查看历史、高风险比例和扰动失败分布。

默认演示模型下，“稳定低风险”使用三种默认扰动；
“置信度变化中风险”使用噪声强度 `0.25`、随机种子 `42` 的单项压力测试；
“预测变化高风险”使用亮度系数 `0.20` 的单项压力测试。三档决策均由当前
固定权重演示模型实时计算，案例中没有预存预测、置信度或风险分数；更换模型、
参数或评分配置后，实际决策可能变化。

## 13. 文件结构

```text
medai-watch/
├── app.py
├── agent/
│   └── monitoring_agent.py
├── config/
│   └── settings.py
├── services/
│   ├── predictor.py
│   ├── image_quality.py
│   ├── perturbations.py
│   ├── gradcam_service.py
│   ├── similarity.py
│   ├── risk_engine.py
│   ├── report_generator.py
│   ├── assets.py
│   ├── database.py
│   ├── demo_cases.py
│   ├── exports.py
│   └── container.py
├── models/
├── data/
│   ├── sample_images/
│   ├── assets/
│   └── medai_watch.db
├── pages/
│   └── 2_监控仪表盘.py
├── scripts/
│   └── verify_sample.py
├── tests/
├── requirements.txt
└── README.md
```

## 14. 测试方法

运行全部测试：

```powershell
$env:TEMP = "E:\medai-watch\.runtime\tmp"
$env:TMP = $env:TEMP
E:\medai-watch\.venv\Scripts\python.exe -m pytest -q
```

运行完整样例、数据库插入和仪表盘查询检查：

```powershell
E:\medai-watch\.venv\Scripts\python.exe E:\medai-watch\scripts\verify_sample.py
```

启动健康检查可通过访问：

```text
http://127.0.0.1:8501/_stcore/health
```

## 15. 已知限制

- 默认模型是非临床固定权重演示网络；
- 未附带真实医学数据、真实标签或训练元数据；
- 风险权重、阈值和扰动强度未经过临床验证；
- Grad-CAM 只用于解释稳定性数值比较，不验证医学关注区域；
- 当前真实模型适配器只覆盖三种 torchvision 架构；
- SQLite 适合单机演示，不适合多实例高并发；
- 未评估数据集外分布、校准、公平性、隐私合规或安全认证；
- 上传文件仅做常见图像解码，不支持 DICOM 专用元数据和窗宽窗位流程。

## 16. 医疗与临床免责声明

本项目是研究和黑客松展示软件，不是医疗器械。其输出不可用于临床诊断、治疗决策、用药建议、患者分诊或替代专业人员判断。任何真实模型接入、数据处理和展示均应由所属机构完成独立验证、隐私评估、法规审查和人工复核。

Grad-CAM、置信度和风险分数仅描述当前实现中的计算结果，不证明模型正确、安全、可靠或适用于任何患者群体。
