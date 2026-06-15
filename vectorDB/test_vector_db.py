import logging
import time
import random
from typing import List, Dict
import numpy as np

from vector_db_manager import VectorDBManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 测试数据类别和模板
CATEGORIES = ["灭火器", "垃圾桶", "摄像头", "消防栓", "路灯", "自行车", "电动车", "汽车", "行人", "书包"]

TEXT_TEMPLATES = {
    "灭火器": ["红色手提式干粉灭火器", "楼道墙壁上的灭火器"],
    "垃圾桶": ["分类垃圾桶", "校园垃圾桶"],
    "摄像头": ["监控摄像头", "安防摄像头"],
    "消防栓": ["地上消防栓", "消防水龙头"],
    "路灯": ["校园路灯", "LED路灯"],
    "自行车": ["共享单车", "学生自行车"],
    "电动车": ["电动自行车", "电瓶车"],
    "汽车": ["轿车", "SUV"],
    "行人": ["学生", "老师"],
    "书包": ["学生书包", "双肩背包"]
}

def _generate_random_embedding(dim: int = 512) -> List[float]:
    """生成随机归一化特征向量"""
    vec = np.random.randn(dim)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()

def generate_test_data(count: int = 1000) -> List[Dict]:
    """生成模拟测试数据"""
    logger.info(f"Generating {count} test data items...")
    items = []
    for i in range(count):
        category = random.choice(CATEGORIES)
        text_descs = TEXT_TEMPLATES.get(category, ["描述"])
        item = {
            "id": f"target_{str(i+1).zfill(5)}",
            "embedding": _generate_random_embedding(512),
            "metadata": {
                "category": category,
                "text_desc": random.choice(text_descs),
                "image_path": f"./data/crops/frame_{str(i//10).zfill(3)}_obj_{str(i%10).zfill(3)}.jpg",
                "detection_conf": round(random.uniform(0.7, 0.99), 4),
                "frame_index": i // 10
            }
        }
        items.append(item)
    logger.info(f"Generated {len(items)} test data items")
    return items

def run_tests():
    """运行完整测试流程"""
    logger.info("=" * 60)
    logger.info("Starting VectorDB Test Suite")
    logger.info("=" * 60)
    
    # 1. 初始化数据库
    logger.info("\n[Step 1] Initializing VectorDBManager...")
    start_time = time.time()
    db_manager = VectorDBManager()
    init_time = time.time() - start_time
    logger.info(f"Initialization completed in {init_time:.2f} seconds")
    
    # 2. 生成测试数据
    logger.info("\n[Step 2] Generating test data...")
    test_data = generate_test_data(1000)
    
    # 3. 批量入库测试
    logger.info("\n[Step 3] Testing batch upsert...")
    start_time = time.time()
    inserted_count = db_manager.batch_upsert(test_data)
    upsert_time = time.time() - start_time
    logger.info(f"Batch upsert completed: {inserted_count} items in {upsert_time:.2f} seconds")
    logger.info(f"Throughput: {inserted_count / upsert_time:.2f} items/sec")
    
    # 4. 统计信息
    stats = db_manager.get_collection_stats()
    logger.info(f"\n[Step 4] Collection stats: {stats}")
    
    # 5. 向量检索测试
    logger.info("\n[Step 5] Testing vector search...")
    for i in range(5):
        query_embedding = _generate_random_embedding(512)
        start_time = time.time()
        results = db_manager.search(query_embedding, top_k=5)
        search_time = time.time() - start_time
        
        logger.info(f"\nQuery {i+1} (search time: {search_time:.4f}s):")
        for j, result in enumerate(results):
            logger.info(f"  [{j+1}] Score: {result['score']:.4f}, Category: {result['category']}, Desc: {result['text_desc']}")
    
    # 6. 重复ID测试
    logger.info("\n[Step 6] Testing duplicate ID handling...")
    duplicate_item = {
        "id": "target_00001",
        "embedding": _generate_random_embedding(512),
        "metadata": {
            "category": "测试类别",
            "text_desc": "重复ID测试",
            "image_path": "./data/crops/test.jpg",
            "detection_conf": 0.95,
            "frame_index": 999
        }
    }
    db_manager.batch_upsert([duplicate_item])
    stats = db_manager.get_collection_stats()
    logger.info(f"Collection count after duplicate upsert: {stats['vector_count']}")
    
    # 7. 边界测试
    logger.info("\n[Step 7] Testing edge cases...")
    
    # 7.1 空列表
    try:
        db_manager.batch_upsert([])
        logger.error("Should have raised ValueError for empty items")
    except ValueError as e:
        logger.info(f"Correctly rejected empty items: {e}")
    
    # 7.2 维度错误
    try:
        db_manager.batch_upsert([{
            "id": "test_dim_error",
            "embedding": [0.1] * 256,  # 错误维度
            "metadata": {"category": "test", "text_desc": "test", "image_path": "test.jpg", "detection_conf": 0.9, "frame_index": 0}
        }])
        logger.error("Should have raised ValueError for dimension mismatch")
    except ValueError as e:
        logger.info(f"Correctly rejected dimension mismatch: {e}")
    
    # 8. 性能测试
    logger.info("\n[Step 8] Performance test (100 queries)...")
    start_time = time.time()
    for _ in range(100):
        query_embedding = _generate_random_embedding(512)
        db_manager.search(query_embedding, top_k=10)
    total_time = time.time() - start_time
    logger.info(f"100 queries completed in {total_time:.2f} seconds")
    logger.info(f"Average query time: {total_time/100*1000:.2f} ms")
    
    logger.info("\n" + "=" * 60)
    logger.info("All tests completed successfully!")
    logger.info("=" * 60)

if __name__ == "__main__":
    run_tests()