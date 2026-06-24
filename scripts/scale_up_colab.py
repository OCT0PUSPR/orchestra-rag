#!/usr/bin/env python3
"""Scale-up training on a GPU (Colab / cloud) over a large MS-MARCO slice.

This is the *same* from-scratch architecture as ``scripts/train_ml.py`` — only
bigger: a larger model, a much larger MS-MARCO slice, more epochs, and a bigger
batch (more in-batch negatives = a stronger contrastive signal). On a single
T4/A100 this trains a meaningfully stronger bi-encoder + cross-encoder.

Colab quickstart
----------------
    !git clone https://github.com/OCT0PUSPR/orchestra-rag.git
    %cd orchestra-rag
    !pip install -r requirements-min.txt -r requirements-train.txt
    !python scripts/scale_up_colab.py --msmarco-limit 50000 --epochs 10 \
        --dim 384 --depth 6 --heads 6 --batch-size 256

The produced checkpoints drop into ``orchestra/ml/checkpoints/`` and are picked
up automatically by ``get_embedder("auto")`` / ``get_reranker(enabled=True)`` /
``get_vector_store("hnsw")`` — so the full RAG stack uses the bigger models with
no code changes.

Notes for full MS-MARCO
-----------------------
* MS-MARCO passage ranking has ~8.8M passages and ~500K training queries. To go
  truly full-scale, stream the official ``collection.tsv`` + ``qrels`` (or use
  ``sentence-transformers/msmarco-bm25`` triplet-all) and raise ``--msmarco-limit``
  to the millions; on an A100 a 6-layer/384-dim encoder trains in a few hours.
* Increase ``--batch-size`` as far as memory allows — in-batch InfoNCE quality
  scales with the number of negatives per step.
* Use ``--device cuda`` (auto-selected) and consider mixed precision (left out
  here for from-scratch clarity; add ``torch.autocast`` around the forward pass).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The scale-up reuses the exact same pipeline as the laptop trainer, just with
# larger defaults. We import its ``main`` and override argv defaults.
import scripts.train_ml as trainer  # noqa: E402


def main() -> int:
    # If the user passed no model-size flags, inject larger scale-up defaults.
    argv = sys.argv[1:]
    defaults = {
        "--epochs": "10",
        "--cross-epochs": "8",
        "--batch-size": "256",
        "--dim": "384",
        "--depth": "6",
        "--heads": "6",
        "--vocab-size": "30000",
        "--msmarco-limit": "50000",
    }
    for flag, value in defaults.items():
        if flag not in argv:
            argv += [flag, value]
    sys.argv = [sys.argv[0]] + argv
    print("[scale-up] effective args:", " ".join(argv))
    return trainer.main()


if __name__ == "__main__":
    raise SystemExit(main())
