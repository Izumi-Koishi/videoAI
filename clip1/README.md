# CLIP1 特征提取模块

本目录用于完成视频 AI 项目中的 **CLIP1：图像/文本特征提取与语义对齐验证**。它不负责 Chroma 入库；入库适配建议放到后续 CLIP2。

## 职责边界

CLIP1 已覆盖：

1. 加载 OpenAI CLIP 或中文 CLIP 预训练模型。
2. 对 YOLO 裁剪出的目标图片批量提取图像向量。
3. 对类别/属性/场景描述批量提取文本向量。
4. 对所有向量做 L2 归一化。
5. 计算图像-文本余弦相似度。
6. 生成同类/异类语义对齐验证报告。

CLIP1 暂不覆盖：

- 写入 `vectorDB.VectorDBManager.batch_upsert(items)`。
- 构造 Chroma metadata。
- LLM/Gradio 问答链路。

这些留给 CLIP2 或集成组完成。

## 建议放入仓库的位置

在仓库 `dev` 分支新建分支并复制本目录：

```bash
git checkout dev
git pull
git checkout -b clip1
cp -r clip1_deliverable/clip1 ./clip1
cp clip1_deliverable/requirements.txt ./clip1/requirements.txt
cp clip1_deliverable/README.md ./clip1/README.md
```

也可以直接把本包内容整体复制到仓库根目录，然后提交。

## 安装依赖

```bash
cd videoAI
pip install -r clip1/requirements.txt
```

默认模型是 OpenAI CLIP：

```bash
export CLIP_MODEL_NAME=openai/clip-vit-base-patch32
export CLIP_MODEL_TYPE=openai
```

如果需要中文文本效果，优先试中文 CLIP：

```bash
export CLIP_MODEL_NAME=OFA-Sys/chinese-clip-vit-base-patch16
export CLIP_MODEL_TYPE=chinese
```

Windows PowerShell 写法：

```powershell
$env:CLIP_MODEL_NAME="OFA-Sys/chinese-clip-vit-base-patch16"
$env:CLIP_MODEL_TYPE="chinese"
```

## 对外接口

```python
from clip1 import CLIPEncoder, cosine_similarity

encoder = CLIPEncoder()

image_vec = encoder.encode_image("yolo2/data/images/person/person_xxx.jpg")
text_vec = encoder.encode_text("校园道路上的行人")
score = cosine_similarity(image_vec, text_vec)

print(len(image_vec))  # 默认 512
print(score)
```

批量接口：

```python
image_vecs = encoder.encode_images([
    "yolo2/data/images/person/a.jpg",
    "yolo2/data/images/car/b.jpg",
])

text_vecs = encoder.encode_texts([
    "校园道路上的行人",
    "校园道路上的汽车",
])
```

所有 `encode_*` 返回值都是已经 L2 归一化的 `list[float]`，可以直接用于余弦相似度或交给后续向量库适配层。

## 从 YOLO 元数据批量提取特征

YOLO 组输出的元数据示例字段通常是：

```json
{
  "image_id": "target_00001",
  "image_path": "yolo2/data/images/person/person_xxx.jpg",
  "class_name": "person",
  "confidence": 0.93,
  "bbox": [10, 20, 120, 220]
}
```

运行批处理：

```bash
python -m clip1.batch_extract \
  --metadata yolo2/data/output/detection_metadata_xxx.json \
  --out data/clip_features/clip1_features.jsonl \
  --model-name openai/clip-vit-base-patch32 \
  --model-type openai \
  --batch-size 32
```

中文 CLIP：

```bash
python -m clip1.batch_extract \
  --metadata yolo2/data/output/detection_metadata_xxx.json \
  --out data/clip_features/clip1_features.jsonl \
  --model-name OFA-Sys/chinese-clip-vit-base-patch16 \
  --model-type chinese \
  --batch-size 32
```

输出 JSONL 每行包含：

```json
{
  "id": "target_00001",
  "image_path": "...",
  "class_name": "person",
  "confidence": 0.93,
  "bbox": [10, 20, 120, 220],
  "text_desc": "校园场景中的行人",
  "image_embedding": [0.01, 0.02],
  "text_embedding": [0.03, 0.04]
}
```

真实向量长度不是 2，而是模型输出维度，默认配置为 512。

## 相似度验证报告

课程指标是同类图文相似度尽量达到 `>= 0.7`，异类图文相似度尽量达到 `<= 0.3`。运行：

```bash
python -m clip1.similarity_eval \
  --metadata yolo2/data/output/detection_metadata_xxx.json \
  --out data/clip_features/clip1_similarity_report.md \
  --model-name OFA-Sys/chinese-clip-vit-base-patch16 \
  --model-type chinese \
  --sample-per-class 3 \
  --same-threshold 0.7 \
  --diff-threshold 0.3
```

报告会记录：

- 同类图像-文本 pair 数量、平均相似度、通过率。
- 异类图像-文本 pair 数量、平均相似度、通过率。
- 每个样本的图片路径、文本描述、相似度和是否通过。

注意：开箱即用的通用 CLIP 不一定能让所有校园类别直接达到课程阈值。如果分数偏低，优先优化 `clip1/descriptions.py` 里的文本模板，其次换中文 CLIP，再考虑领域微调。

## 和 vectorDB 的衔接

当前向量库要求后续传入：

```python
{
    "id": "target_00001",
    "embedding": image_embedding,
    "metadata": {
        "category": class_name,
        "text_desc": text_desc,
        "image_path": image_path,
        "detection_conf": confidence,
        "frame_index": 1
    }
}
```

CLIP1 已经提供 `image_embedding`、`text_embedding`、`text_desc` 和原始检测字段。CLIP2 只需要完成：

1. 从 `image_path` 或 YOLO 元数据解析 `frame_index`。
2. 将 `image_embedding` 改名为 `embedding`。
3. 将 `class_name/confidence` 映射到 `metadata.category/detection_conf`。
4. 调用 `VectorDBManager.batch_upsert(items)`。

## 本地测试

不下载模型的工具函数测试：

```bash
PYTHONPATH=. python -m pytest tests
```

或只做语法检查：

```bash
python -m py_compile clip1/*.py
```

## 常见问题

### 1. 模型下载太慢

提前在能联网的环境下载模型，或设置 Hugging Face 镜像/缓存。已经缓存后可设置：

```bash
export CLIP_LOCAL_FILES_ONLY=1
```

### 2. 中文查询效果不好

使用中文 CLIP，并把类别描述改成更接近图像内容的短句。例如不要只写“汽车”，可以写“校园道路上的白色汽车”。

### 3. 向量维度不是 512

vectorDB 默认校验 512 维。如果更换模型导致维度变化，需要同步修改 `vectorDB/config.py` 的 `feature_dim`，或者换回 512 维 CLIP 模型。
