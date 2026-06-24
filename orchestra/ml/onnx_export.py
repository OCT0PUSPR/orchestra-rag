"""Export the trained bi-encoder to ONNX and verify it with onnxruntime.

The exported graph takes ``input_ids`` and ``attention_mask`` (both int64,
dynamic batch + sequence) and returns the L2-normalized embedding. We verify
parity against the PyTorch model on a sample batch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

__all__ = ["export_bi_encoder", "verify_onnx"]


def export_bi_encoder(
    bi_ckpt: str | Path,
    tokenizer_path: str | Path,
    out_path: str | Path,
    *,
    opset: int = 14,
    sample_texts: Optional[list] = None,
) -> Path:
    """Export the bi-encoder at ``bi_ckpt`` to ONNX at ``out_path``."""
    import torch

    from orchestra.ml.bi_encoder import BiEncoder
    from orchestra.ml.tokenizer import BPETokenizer

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = BiEncoder.load(bi_ckpt, map_location="cpu").to("cpu")
    model.eval()
    tokenizer = BPETokenizer.load(tokenizer_path)

    sample_texts = sample_texts or ["how fast can the robot swap its battery", "warehouse fleet coordination"]
    ids, mask = tokenizer.encode_batch(sample_texts, max_len=64)
    ids_t = torch.tensor(ids, dtype=torch.long)
    mask_t = torch.tensor(mask, dtype=torch.long)

    torch.onnx.export(
        model,
        (ids_t, mask_t),
        str(out_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "embedding": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    return out_path


def verify_onnx(
    onnx_path: str | Path,
    bi_ckpt: str | Path,
    tokenizer_path: str | Path,
    *,
    atol: float = 1e-3,
) -> float:
    """Run the ONNX model and the torch model on a sample; return max abs diff."""
    import onnxruntime as ort
    import torch

    from orchestra.ml.bi_encoder import BiEncoder
    from orchestra.ml.tokenizer import BPETokenizer

    model = BiEncoder.load(bi_ckpt, map_location="cpu").to("cpu")
    model.eval()
    tokenizer = BPETokenizer.load(tokenizer_path)
    texts = ["how long does the battery last", "approved programming languages"]
    ids, mask = tokenizer.encode_batch(texts, max_len=64)
    ids_np = np.asarray(ids, dtype=np.int64)
    mask_np = np.asarray(mask, dtype=np.int64)

    with torch.no_grad():
        torch_out = model(
            torch.tensor(ids_np, dtype=torch.long),
            torch.tensor(mask_np, dtype=torch.long),
        ).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["embedding"], {"input_ids": ids_np, "attention_mask": mask_np})[0]
    return float(np.max(np.abs(torch_out - onnx_out)))
