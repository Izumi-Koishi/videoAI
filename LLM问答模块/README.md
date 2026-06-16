# YOLO-CLIP 多模态问答系统 — LLM问答 + Gradio界面模块

## 项目背景

本项目是 **2026年春 软件过程与管理** 课程小组作业的一部分，主题为「基于YOLO-CLIP的领域特定多模态问答系统开发」。系统整合目标检测、多模态语义对齐、向量检索与大语言模型四大技术模块，搭建面向特定领域的交互式多模态问答系统。

## 本模块职责

本人负责 **LLM问答 + Gradio界面** 模块，核心任务包括：

1. **大语言模型对接** — 封装统一 LLM 调用接口，支持多种后端
2. **领域专属 Prompt 模板** — 设计系统角色、上下文注入、反幻觉指令等模板
3. **完整问答主逻辑** — 实现「问题→检索→Prompt拼接→LLM生成→后处理」全链路
4. **回答质量调优** — 减少幻觉、规范输出格式、置信度评估
5. **Gradio 可视化界面** — 视频上传、检测展示、问答交互、匹配图片展示
6. **端到端交互** — 预留接口对接队友模块，Mock 模式可独立演示

## 项目结构

```
YOLO-CLIP-QA/
├── main.py                    # 启动入口
├── config.py                  # 全局配置（LLM后端、模型参数、路径等）
├── requirements.txt           # Python 依赖
├── tuning_log.md              # 回答效果调优记录
├── README.md                  # 本文件
│
├── llm/                       # LLM 调用封装层
│   ├── base.py                #   抽象基类 BaseLLM，定义统一 generate() 接口
│   ├── ollama_backend.py      #   Ollama 后端（支持 Llama3/Qwen/ChatGLM3）
│   ├── openai_backend.py      #   OpenAI 兼容 API 后端
│   └── factory.py             #   工厂函数 + MockLLM（演示/测试用）
│
├── prompts/                   # Prompt 模板库
│   ├── templates.py           #   PromptBuilder：系统角色/上下文注入/反幻觉/
│   │                          #   输出格式等模板，支持自定义领域
│   └── examples.py            #   Few-shot 标准问答示例（2个完整示例）
│
├── qa/                        # 问答引擎
│   ├── engine.py              #   QAEngine：完整问答流程编排
│   │                          #   检索→Prompt构建→LLM生成→后处理→QAResult
│   └── postprocess.py         #   AnswerPostProcessor：幻觉检测/置信度提取/
│                              #   引用追踪/质量警告
│
└── interface/                 # 交互界面
    ├── gradio_app.py          #   Gradio 完整界面：视频上传区、检测结果预览、
    │                          #   问题输入、LLM回答、匹配图片展示、配置面板
    └── mock_modules.py        #   MockYOLODetector / MockCLIPRetriever
                               #   模拟队友模块，支持独立演示
```

## 快速开始

### 环境要求

- Python 3.10+
- 依赖安装：`pip install -r requirements.txt`

### 启动应用

```bash
cd YOLO-CLIP-QA
python main.py
```

浏览器访问 `http://127.0.0.1:7860`

### 配置 LLM 后端

| 后端 | 配置方式 |
|------|----------|
| **Mock（默认）** | 无需配置，使用模拟回答演示界面和流程 |
| **Ollama** | 安装 [Ollama](https://ollama.com)，拉取模型后设置环境变量 `LLM_BACKEND=ollama` |
| **OpenAI API** | 设置 `LLM_BACKEND=openai` 和 `OPENAI_API_KEY=your-key` |

详见 `config.py` 中的全部配置项。

## 架构设计

### LLM 封装层

```
BaseLLM (抽象基类)
├── OllamaBackend              # 本地开源模型（Ollama 服务）
├── OpenAICompatibleBackend    # OpenAI 及兼容 API
└── MockLLM                    # 模拟后端（演示/测试）
```

统一接口 `generate(prompt, system_prompt, **kwargs) -> str`，通过 `create_llm(backend)` 工厂函数按需创建。

### 问答流程

```
用户问题
  → CLIP 文本特征提取（队友模块接口）
  → 向量数据库检索 Top-K（队友模块接口）
  → 相似度阈值过滤（< 0.3 丢弃）
  → Prompt 拼接（系统指令 + 检索上下文 + Few-shot + 问题 + 反幻觉指令）
  → LLM 生成回答
  → 后处理（幻觉检测 + 置信度提取 + 质量警告）
  → 返回 QAResult（回答 + 上下文 + 耗时 + 警告）
```

### Gradio 界面布局

```
┌─────────────────────────────────────────────────┐
│                  🎯 系统标题                      │
│         YOLO + CLIP + 向量检索 + LLM             │
├────────────────────┬────────────────────────────┤
│   📹 视频输入       │   💬 智能问答               │
│   [上传视频]        │   [问题输入框]              │
│   [开始检测] [清空]  │   [提交问题]               │
│   [处理状态]        │   [LLM 回答输出]            │
│                    │                            │
│   🖼 检测结果预览    │   📸 匹配目标图像           │
│   [目标 Gallery]    │   [匹配 Gallery]           │
├────────────────────┴────────────────────────────┤
│  📊 检索详情        │  🤖 系统信息                │
└─────────────────────────────────────────────────┘
```

## 反幻觉策略

| 层级 | 策略 | 实现位置 |
|------|------|----------|
| Prompt 层 | 系统角色约束 + Few-shot 规范 + 反幻觉指令 | `prompts/templates.py` |
| 参数层 | 低温度 (0.3) 减少随机生成 | `config.py` |
| 检索层 | 相似度阈值过滤低质量上下文 | `qa/engine.py` |
| 后处理层 | 幻觉关键词检测 + 置信度提取 + 质量警告 | `qa/postprocess.py` |

## 与队友模块的对接

| 队友模块 | 接口约定 | 注入方式 |
|----------|----------|----------|
| YOLO 检测器 | `detect_video(path) -> List[DetectionResult]` | `MockYOLODetector` 替换 |
| CLIP 向量检索 | `search(query_text, top_k) -> List[SearchResult]` | `qa_engine.set_retriever()` |
| 文本特征提取 | `encode(text) -> ndarray` | `qa_engine.set_text_encoder()` |

`interface/mock_modules.py` 中的 Mock 类在使用真实模块时直接替换即可。

## 交付物清单

- [x] LLM 调用封装（3 种后端 + 工厂函数）
- [x] Prompt 模板库（系统角色/上下文/Few-shot/反幻觉/格式）
- [x] 问答主逻辑脚本（完整流程 + 后处理）
- [x] 回答效果调优记录（`tuning_log.md`）
- [x] 完整 Gradio 交互界面（6 大功能区）
- [x] 可交互 Demo（`python main.py` 一键启动）
