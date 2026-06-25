import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from clip1.descriptions import build_text_desc
from clip1.encoder import cosine_similarity, l2_normalize


def test_l2_normalize_vector_norm_is_one():
    vec = l2_normalize([3.0, 4.0])
    assert np.allclose(vec, [0.6, 0.8])
    assert math.isclose(float(np.linalg.norm(vec)), 1.0, rel_tol=1e-6)


def test_l2_normalize_matrix_row_norms_are_one():
    mat = l2_normalize([[3.0, 4.0], [5.0, 12.0]])
    norms = np.linalg.norm(mat, axis=1)
    assert np.allclose(norms, [1.0, 1.0])


def test_cosine_similarity_after_normalization():
    assert math.isclose(cosine_similarity([1, 0], [10, 0]), 1.0, rel_tol=1e-6)
    assert math.isclose(cosine_similarity([1, 0], [0, 1]), 0.0, abs_tol=1e-6)


def test_build_text_desc_known_and_unknown_labels():
    assert "校园" in build_text_desc("person")
    assert build_text_desc("未知类别") == "校园场景中的未知类别目标"
