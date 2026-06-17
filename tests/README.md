# tests/ — videoAI 集成测试组交付物

> **组别**: 集成测试组 | **日期**: 2026-06-17 | **分支**: dev

---

## 📦 交付物清单

```
tests/
├── README.md                       # 本文件 — 交付物总览
├── DEPLOY.md                       # 部署文档 — 从零部署到运行
├── requirements.txt                # 统一 Python 依赖清单
├── main.py                         # 主入口 — 统一启动与测试入口
├── integration_test.py             # 集成测试套件 — 自动化测试框架
├── start.bat                       # Windows 一键启动脚本
├── start.sh                        # Linux/macOS 一键启动脚本
├── test_config/
│   ├── __init__.py                 # 包标记
│   └── config.py                   # 统一配置中心
├── logs/                           # 运行日志目录
│   └── videoai_YYYYMMDD.log
└── output/                         # 测试产出目录
    ├── integration_test_report.md  # 测试报告 (Markdown)
    ├── integration_test_report.json# 测试报告 (JSON)
    ├── pipeline_test_report.json  # 管线测试报告
    └── vector_db/                  # 向量数据库持久化
```

---

## 🚀 快速开始

### 0. 前提条件
- Python 3.10+
- 数据准备组已提供 `data/data/` 下的测试数据

### 1. 一键启动
```bash
# Windows
双击 tests\start.bat

# Linux / macOS
bash tests/start.sh
```

### 2. 手动运行

```bash
# 安装依赖
pip install -r tests/requirements.txt

# 验证环境
python tests/main.py --mode verify

# 处理测试视频
python tests/main.py --mode process

# 运行集成测试
python tests/main.py --mode test

# 完整管线（处理 + 测试）
python tests/main.py --mode pipeline

# 启动 Web 界面
python tests/main.py --mode web
```

---

## 🔄 系统数据流

```
                        ┌──────────────────┐
                        │  campus_knowledge │
                        │    _base.json     │──→ LLM 知识上下文
                        └──────────────────┘
                                  │
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  视频输入  │──→│ YOLO v8  │──→│  CLIP 1  │──→│ VectorDB │──→│   LLM    │──→ 回答
│  (.mp4)   │   │  目标检测  │   │  特征提取  │   │  向量存储  │   │  问答生成  │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
                     │               │               │               │
               检测框+标签      L2归一化向量    余弦相似度检索    Prompt模板
                     │               │               │               │
                     └───────────────┴───────┬───────┘               │
                                             │                       │
                                     campus_test_qa.json ────────────┘
                                        (验收标准)
```

---

## 📊 测试策略

### 测试层级

| 层级 | 文件 | 内容 |
|------|------|------|
| **L1 环境验证** | `main.py --mode verify` | Python版本、依赖、数据文件检查 |
| **L2 模块接口** | `integration_test.py` 测试2 | YOLO/VectorDB/LLM/CLIP 导入和初始化 |
| **L3 管线集成** | `integration_test.py` 测试3 | 视频→检测→入库→检索→问答 端到端 |
| **L4 数据验证** | `integration_test.py` 测试1,4 | 知识库、QA数据完整性校验 |
| **L5 性能基准** | `integration_test.py` 测试5 | 各环节耗时基准 |

### 验收标准

- [x] 环境验证全部通过（核心项）
- [x] 所有模块可正常导入和初始化
- [x] YOLO检测能正确处理测试视频
- [x] 检测结果能存入向量数据库
- [x] 向量检索能返回相关结果
- [x] LLM能基于检索上下文生成回答
- [x] 知识库覆盖测试QA的主要实体
- [x] Web界面可访问和操作

---

## ⚙️ 配置项速查

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `OPENAI_API_KEY` | (空) | OpenAI API 密钥 |
| `LLM_BACKEND` | `openai` | LLM 后端: openai / ollama |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | 使用的模型名称 |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Ollama 本地模型 |
| `YOLO_FRAME_INTERVAL` | `30` | YOLO 检测帧间隔 |
| `YOLO_CONFIDENCE` | `0.25` | YOLO 置信度阈值 |
| `GRADIO_PORT` | `7860` | Web 界面端口 |

详见 [DEPLOY.md](DEPLOY.md) 第5节。

---

## 📝 文件说明

### main.py — 统一主入口
支持6种运行模式，是所有操作的统一入口。通过 `--mode` 切换。

### integration_test.py — 集成测试套件
独立的测试脚本，包含5类测试共25+测试项。支持 `--skip-llm` 跳过LLM测试。

### start.bat / start.sh — 一键启动脚本
交互式菜单，引导用户完成环境检查、依赖安装、模式选择。

### DEPLOY.md — 部署文档
9节完整文档，覆盖从环境准备到生产部署的全流程。

### test_config/config.py — 配置中心
所有配置项集中管理，支持环境变量覆盖。

---

> 🔗 相关文档：[DEPLOY.md](DEPLOY.md) | [llm_gradio/模块对接文档.md](../llm_gradio/模块对接文档.md) | [vectorDB/README.md](../vectorDB/README.md)
>
> 🤖 集成测试组出品
