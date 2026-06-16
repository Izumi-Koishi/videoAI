# 向量数据库子模块

基于 Chroma 的本地持久化向量数据库，为领域特定多模态问答系统提供向量检索能力。

> **注意**: 本模块专注于向量存储与检索。CLIP 特征提取由上游模块负责。

## 功能特性

- **本地持久化**: 基于 Chroma PersistentClient，数据存储在本地磁盘
- **高效入库**: 支持批量向量入库，避免循环单条插入
- **HNSW 索引**: 自动构建 HNSW 索引，使用余弦相似度
- **标准化接口**: 统一输入输出格式，便于下游 LLM 对接

## 文件结构

```
vectorDB/
├── requirements.txt              # 依赖清单（chromadb、numpy）
├── config.py                    # 全局配置
├── vector_db_manager.py         # 核心管理类（主版本）
├── vector_db_manager_simple.py  # 简化版管理类（备用版本）
├── test_vector_db.py            # 测试脚本（主版本）
├── test_simple.py               # 简化版测试脚本（备用版本）
├── README.md                    # 模块说明
└── TEST_REPORT.md               # 测试报告
```

> **版本说明**:
>
> - `vector_db_manager.py` / `test_vector_db.py` 为主版本
> - `vector_db_manager_simple.py` / `test_simple.py` 为备用版本（不依赖 numpy，直接使用列表操作）

## 环境搭建

### 安装依赖

```bash
cd vectorDB
pip install -r requirements.txt
```

### 依赖说明

```
chromadb==0.4.24    # 向量数据库
numpy>=1.24.0,<2.0.0  # 数值计算（主版本需要）
```

### 首次运行

首次运行时会自动创建目录：

- `../data/vector_db/` - 向量数据库持久化目录

## 数据格式

### 入库数据格式

每条入库数据必须包含以下字段：

```python
{
    "id": "target_00001",           # 全局唯一ID（字符串）
    "embedding": [0.1, 0.2, ...],   # 512维归一化特征向量（list[float]）
    "metadata": {
        "category": "灭火器",        # 类别名称
        "text_desc": "楼道墙壁上的红色手提式干粉灭火器",  # 文本描述
        "image_path": "./data/crops/frame_001_obj_001.jpg",  # 原图路径
        "detection_conf": 0.92,     # YOLO检测置信度（0.0-1.0）
        "frame_index": 1            # 视频帧序号
    }
}
```

**字段说明**:

| 字段                        | 类型        | 必填 | 说明                              |
| --------------------------- | ----------- | ---- | --------------------------------- |
| `id`                      | str         | 是   | 全局唯一标识，如 `target_00001` |
| `embedding`               | list[float] | 是   | 512维归一化特征向量               |
| `metadata.category`       | str         | 是   | 目标类别                          |
| `metadata.text_desc`      | str         | 是   | 目标文本描述                      |
| `metadata.image_path`     | str         | 是   | 原图相对路径                      |
| `metadata.detection_conf` | float       | 是   | YOLO置信度（0.0-1.0）             |
| `metadata.frame_index`    | int         | 是   | 视频帧序号                        |

### 返回结果格式

检索结果为字典列表，按相似度从高到低排序：

```python
[
    {
        "id": "target_00001",
        "score": 0.85,                           # 余弦相似度分数（0-1，越高越匹配）
        "category": "灭火器",
        "text_desc": "楼道墙壁上的红色手提式干粉灭火器",
        "image_path": "./data/crops/frame_001_obj_001.jpg",
        "detection_conf": 0.92
    },
    {
        "id": "target_00002",
        "score": 0.78,
        "category": "灭火器",
        "text_desc": "教室门口的红色灭火器",
        "image_path": "./data/crops/frame_002_obj_003.jpg",
        "detection_conf": 0.88
    }
]
```

## 快速开始

### 初始化向量数据库

```python
from vector_db_manager import VectorDBManager

# 初始化数据库管理器
db_manager = VectorDBManager()

# 查看集合统计信息
stats = db_manager.get_collection_stats()
print(stats)
# {'collection_name': 'campus_targets', 'vector_count': 1000}
```

### 批量入库

```python
# 准备数据（通常由上游 YOLO+CLIP 模块生成）
items = [
    {
        "id": "target_00001",
        "embedding": [0.1, 0.2, ...],  # CLIP 提取的 512 维特征向量
        "metadata": {
            "category": "灭火器",
            "text_desc": "楼道墙壁上的红色手提式干粉灭火器",
            "image_path": "./data/crops/frame_001_obj_001.jpg",
            "detection_conf": 0.92,
            "frame_index": 1
        }
    },
    # ... 更多数据
]

# 批量入库
count = db_manager.batch_upsert(items)
print(f"成功入库 {count} 条数据")
```

### 向量检索

> **重要**: 本模块不包含 CLIP 特征提取功能。查询向量需由外部 CLIP 模型生成。

```python
# 1. 上游模块（YOLO + CLIP）提取查询图像/文本的特征向量
# query_embedding = clip_model.extract_image_features(query_image_path)
# 或
# query_embedding = clip_model.extract_text_features(query_text)

# 2. 将特征向量传入本模块进行检索
results = db_manager.search(query_embedding, top_k=5)

# 3. 遍历结果
for result in results:
    print(f"ID: {result['id']}, Score: {result['score']}, Category: {result['category']}")
```

## API 接口

### VectorDBManager 类

#### `__init__(persist_dir=None, collection_name=None)`

初始化向量数据库管理器，自动创建目录和集合。

**参数**:

- `persist_dir`: 数据持久化目录（默认使用配置）
- `collection_name`: 集合名称（默认使用配置）

#### `batch_upsert(items: List[Dict]) -> int`

批量插入/更新向量数据。

**参数**:

- `items`: 待入库的数据列表，格式见上方「入库数据格式」

**返回**: 成功入库的数据条数

**异常**:

- `ValueError`: 列表为空时抛出

#### `search(query_embedding: List[float], top_k: int = None) -> List[Dict]`

向量检索接口。

**参数**:

- `query_embedding`: CLIP 提取的查询向量（512维）
- `top_k`: 返回结果数量（默认5，最大100）

**返回**: 按相似度降序排列的结果列表，格式见上方「返回结果格式」

#### `get_collection_stats() -> Dict`

获取集合统计信息。

#### `collection_exists(collection_name: str = None) -> bool`

判断集合是否存在。

#### `delete_collection(collection_name: str = None)`

删除指定集合。

#### `reset_collection()`

重置集合（删除后重新创建）。

## 配置说明

配置文件 `config.py`：

```python
# 向量数据库配置
VECTOR_DB_CONFIG = {
    "persist_directory": "./data/vector_db",
    "collection_name": "campus_targets",
    "feature_dim": 512,  # 特征向量维度
}

# HNSW 索引参数
HNSW_CONFIG = {
    "M": 16,
    "ef_construction": 200,
    "ef_search": 50,
}

# 检索配置
SEARCH_CONFIG = {
    "default_top_k": 5,
    "max_top_k": 100,
}
```

## 运行测试

```bash
cd vectorDB
python test_vector_db.py
```

测试脚本包含：

1. 数据库初始化
2. 生成1000条模拟数据
3. 批量入库测试
4. 向量检索测试（5个查询）
5. 重复ID处理测试
6. 边界条件测试
7. 性能测试（100次查询）

## 与上游模块集成

```
┌─────────────────┐    CLIP特征向量     ┌─────────────────┐
│  YOLO 目标检测   │ ─────────────────► │  本模块 (VectorDB) │
│  CLIP 特征提取   │                    │  向量存储与检索    │
└─────────────────┘                    └─────────────────┘
```

本模块作为向量数据库，接收上游模块（YOLO + CLIP）提取的特征向量，提供标准化检索接口供下游 LLM 使用。

## 常见问题

### Q: 如何清空数据库？

A: 调用 `db_manager.reset_collection()` 或删除 `data/vector_db/` 目录。

### Q: 支持哪些距离度量？

A: 默认使用余弦相似度，与 CLIP 语义对齐标准保持一致。

### Q: 特征向量维度必须是512吗？

A: 默认配置为512维，可通过修改 `config.py` 中的 `feature_dim` 调整。

### Q: 如何切换到备用版本？

A: 将 `from vector_db_manager import VectorDBManager` 改为 `from vector_db_manager_simple import VectorDBManager`。

### Q:下载依赖库时，出现Building wheel for chroma-hnswlib (pyproject.toml) ... error问题

A:该问题是在安装chroma时，依赖包 **chroma-hnswlib** 需要编译 C++ 扩展组件。建议下载Microsoft C++ Build Tools（下载地址https://visualstudio.microsoft.com/visual-cpp-build-tools/），安装

* MSVC v143 /v142 生成工具（C++ 对应 VC++14.x）
* Windows 10/11 SDK（任选其一）
* CMake、C++ CMake 工具

后可解决该问题

## 注意事项

1. 特征向量必须是归一化的512维向量
2. 入库数据的 `id` 字段必须全局唯一（重复ID会自动覆盖）
3. 建议批量入库时单次不超过10000条
4. Windows/Linux/macOS 路径兼容性已处理
5. 该模块只提供向量数据库操作，不提供特征化操作。search函数接收特征向量，可使用CLIP模块进行特征化后进行检索。
