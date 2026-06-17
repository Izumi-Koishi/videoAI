# videoAI 系统部署文档

> **版本**: 1.0 | **最后更新**: 2026-06-17 | **目标平台**: Windows / Linux / macOS

---

## 目录

1. [系统概述](#1-系统概述)
2. [环境要求](#2-环境要求)
3. [快速部署](#3-快速部署)
4. [详细部署步骤](#4-详细部署步骤)
5. [配置说明](#5-配置说明)
6. [启动与运行](#6-启动与运行)
7. [测试与验证](#7-测试与验证)
8. [故障排查](#8-故障排查)
9. [生产环境建议](#9-生产环境建议)

---

## 1. 系统概述

videoAI 是一个**校园监控视频 AI 智能问答系统**，基于以下技术栈：

```
视频输入 → YOLO目标检测 → CLIP特征提取 → Chroma向量数据库 → LLM问答 → Web界面
```

| 模块 | 技术 | 功能 |
|------|------|------|
| 目标检测 | YOLOv8 | 视频帧中检测人物、车辆、设施等目标 |
| 特征提取 | CLIP (OpenAI/中文) | 将图像和文本映射到统一语义空间 |
| 向量检索 | ChromaDB | 存储和检索检测结果的特征向量 |
| 语言模型 | OpenAI API / Ollama | 根据检测结果和知识库生成智能回答 |
| Web界面 | Gradio | 提供可视化操作界面 |

---

## 2. 环境要求

### 硬件要求

| 资源 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | 4核 | 8核+ |
| 内存 | 8 GB | 16 GB+ |
| 磁盘 | 10 GB 空闲 | 50 GB SSD |
| GPU | 无（CPU可运行） | NVIDIA GPU 4GB+ 显存 |
| 网络 | 宽带 | 宽带（需下载模型） |

### 软件要求

| 软件 | 版本 | 说明 |
|------|------|------|
| Python | 3.10+ | 推荐 3.11 |
| pip | 最新版 | `pip install --upgrade pip` |
| Git | 2.0+ | 用于克隆仓库 |

**Windows 额外要求**:
- [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)（chromadb 编译需要）
  - 安装时勾选：MSVC v143 生成工具、Windows SDK、CMake 工具

**Linux 额外要求**:
```bash
# Ubuntu/Debian
sudo apt install build-essential python3-dev
```

---

## 3. 快速部署

### 一键启动（推荐）

**Windows**:
```cmd
双击运行 tests\start.bat
```

**Linux / macOS**:
```bash
bash tests/start.sh
```

### 手动部署（3步）

```bash
# 1. 克隆仓库并切换到 dev 分支
git clone https://github.com/DolaNoDream/videoAI.git
cd videoAI
git checkout dev

# 2. 安装依赖
pip install -r tests/requirements.txt

# 3. 运行
python tests/main.py --mode web
```

---

## 4. 详细部署步骤

### 4.1 获取代码

```bash
git clone https://github.com/DolaNoDream/videoAI.git
cd videoAI
git checkout dev
```

### 4.2 创建虚拟环境（推荐）

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 4.3 安装 Python 依赖

```bash
pip install --upgrade pip

# 完整安装
pip install -r tests/requirements.txt

# 如果下载慢，使用清华镜像
pip install -r tests/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4.4 下载模型文件（首次运行自动下载）

| 模型 | 用途 | 大小 | 说明 |
|------|------|------|------|
| `yolov8n.pt` | YOLO 目标检测 | ~6 MB | 首次运行自动下载 |
| `all-MiniLM-L6-v2` | 文本嵌入 | ~90 MB | VectorStore 自动下载 |
| `clip-vit-base-patch32` | CLIP 特征提取 | ~600 MB | 首次使用自动下载（可选） |

### 4.5 准备数据文件

将数据准备组提供的文件放入项目根目录：

```
videoAI/
└── data/
    └── data/
        ├── campus_knowledge_base.json    # 校园知识库
        ├── campus_test_qa.json           # 测试问答对
        └── *.mp4                         # 测试视频（可选）
```

### 4.6 配置 LLM API Key（可选但推荐）

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-api-key"

# Linux / macOS
export OPENAI_API_KEY="sk-your-api-key"

# 或者创建 .env 文件（推荐）
echo "OPENAI_API_KEY=sk-your-api-key" > .env
```

> **没有 API Key？** 可以使用 Ollama 本地模型（见下方配置说明）。

---

## 5. 配置说明

### 5.1 环境变量

创建 `.env` 文件或直接设置环境变量：

```bash
# ---- LLM 配置 ----
LLM_BACKEND=openai               # openai | ollama
OPENAI_API_KEY=sk-xxx            # OpenAI / 兼容 API 的密钥
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo

# ---- Ollama 本地模型 ----
# LLM_BACKEND=ollama
# OLLAMA_HOST=http://localhost:11434
# OLLAMA_MODEL=qwen2.5:7b

# ---- YOLO 检测 ----
YOLO_MODEL_NAME=yolov8n.pt      # 模型: yolov8n/s/m/l/x
YOLO_FRAME_INTERVAL=30           # 检测帧间隔
YOLO_CONFIDENCE=0.25             # 置信度阈值

# ---- 路径 ----
VECTOR_DB_PATH=./tests/output/vector_db

# ---- Web 界面 ----
GRADIO_HOST=127.0.0.1
GRADIO_PORT=7860
```

### 5.2 配置文件

主要配置文件位于 `tests/config/config.py`，可通过环境变量覆盖。

### 5.3 使用 Ollama 本地模型（无需 API Key）

```bash
# 1. 安装 Ollama
# https://ollama.com/download

# 2. 拉取模型
ollama pull qwen2.5:7b

# 3. 设置环境变量
export LLM_BACKEND=ollama
export OLLAMA_MODEL=qwen2.5:7b

# 4. 启动系统
python tests/main.py --mode web
```

---

## 6. 启动与运行

### 6.1 运行模式一览

| 模式 | 命令 | 说明 |
|------|------|------|
| 环境验证 | `python tests/main.py --mode verify` | 检查环境和依赖 |
| 视频处理 | `python tests/main.py --mode process` | YOLO检测 + 向量入库 |
| 集成测试 | `python tests/main.py --mode test` | 运行完整测试套件 |
| 完整管线 | `python tests/main.py --mode pipeline` | 处理 + 测试一站式 |
| 交互问答 | `python tests/main.py --mode interactive` | 命令行交互式问答 |
| Web界面 | `python tests/main.py --mode web` | 启动 Gradio Web 界面 |

### 6.2 Web 界面使用流程

1. 启动 Web 界面：`python tests/main.py --mode web`
2. 浏览器访问 `http://127.0.0.1:7860`
3. **配置模型** — 展开"模型配置"，填入 API Key 或切换到 Ollama
4. **上传视频** — 点击"上传视频"选择 MP4 文件
5. **开始检测** — 点击"开始检测处理"，等待 YOLO 检测完成
6. **智能问答** — 输入问题，系统检索 + LLM 生成回答
7. **查看结果** — 回答、上下文、匹配图片同步展示

### 6.3 命令行参数

```
python tests/main.py --help

可选参数:
  --mode, -m         运行模式 (verify|process|interactive|test|pipeline|web)
  --video, -v        指定视频文件路径
  --frame-interval    YOLO 检测帧间隔 (默认: 30)
  --conf-threshold    YOLO 置信度阈值 (默认: 0.25)
  --skip-llm         跳过 LLM 相关测试
  --qa-sample        测试问答采样数量 (0=全部)
```

---

## 7. 测试与验证

### 7.1 环境验证

```bash
python tests/main.py --mode verify
```

预期输出：
```
  Python 3.11.x ✓
  知识库: ... ✓
  测试问答: ... ✓
  ultralytics ✓
  chromadb ✓
  openai ✓
  ...
  总计: X/Y 项通过
```

### 7.2 集成测试

```bash
# 完整测试（需要 API Key）
python tests/integration_test.py

# 跳过 LLM 测试（仅测试模块接口和管线）
python tests/integration_test.py --skip-llm

# 快速模式（跳过性能测试）
python tests/integration_test.py --quick --skip-llm
```

测试报告输出位置：
- `tests/output/integration_test_report.md` — 可读报告
- `tests/output/integration_test_report.json` — 结构化数据

### 7.3 测试覆盖范围

| 测试类别 | 测试项 | 说明 |
|----------|--------|------|
| 数据完整性 | 4项 | 知识库、测试QA、视频文件、数据关联 |
| 模块接口 | 9项 | YOLO/VectorDB/LLM/CLIP 初始化和模板完整性 |
| 管线集成 | 5项 | 端到端视频处理、向量存储、检索、问答 |
| 知识库集成 | 4项 | 分类验证、实体统计、QA覆盖、属性完整性 |
| 性能基准 | 3项 | 导入时间、检索延迟、查询性能 |

---

## 8. 故障排查

### 8.1 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `ModuleNotFoundError: ultralytics` | 依赖未安装 | `pip install -r tests/requirements.txt` |
| `chromadb 安装失败 (Windows)` | 缺少 C++ 编译工具 | 安装 [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |
| `CUDA out of memory` | GPU 显存不足 | 设 `export CLIP_DEVICE=cpu` 使用 CPU 模式 |
| `LLM 未就绪` | 未配置 API Key | 在"模型配置"中填写 API Key，或切换到 Ollama |
| `向量数据库未就绪` | ChromaDB 未正确初始化 | 检查 `chromadb` 版本 >= 0.4.24 |
| `Connection refused (Ollama)` | Ollama 服务未启动 | 运行 `ollama serve` |
| `模型下载太慢` | 网络问题 | 设置 HuggingFace 镜像或手动下载模型 |
| `data/data 目录不存在` | 数据文件位置不正确 | 确保数据在 `videoAI/data/data/` 下 |

### 8.2 日志查看

运行日志保存在 `tests/logs/` 目录，按日期命名：
```
tests/logs/videoai_20260617.log
```

### 8.3 重置系统

```bash
# 清除向量数据库
rm -rf tests/output/vector_db/

# 清除所有输出
rm -rf tests/output/*
```

---

## 9. 生产环境建议

### 9.1 性能优化

- **GPU 加速**：使用 NVIDIA GPU 运行 YOLO 检测和 CLIP 特征提取
- **模型选择**：YOLOv8s 平衡速度与精度；yolov8n 适合低配环境
- **批量处理**：CLIP 特征提取使用较大的 batch_size（32-64）
- **持久化**：向量数据库持久化避免重复处理

### 9.2 安全建议

- API Key 使用环境变量管理，不要硬编码在代码中
- 生产环境关闭 Gradio `share=True` 公网分享
- 使用反向代理（Nginx）提供 HTTPS 访问
- 限制文件上传大小和类型

### 9.3 扩展方向

- 支持 RTSP 实时视频流
- 集成更多检测模型（YOLOv10, Grounding DINO）
- 多视频并发处理
- 添加用户认证和权限管理
- 告警推送（检测到异常行为时）

---

> 📧 如有问题，请查看项目 Issues 或联系开发团队。
>
> 🤖 由 videoAI 集成测试组维护
