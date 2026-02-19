"""Vertex AI training entrypoint.

All configuration is via environment variables so the same image can be used
for different experiments without rebuilding.

Required env vars:
    GCS_BUCKET          GCS bucket name (without gs://), e.g. "my-stocks-nn"

Optional env vars (have sensible defaults):
    SP500_PATH          Full path to sp500_history.csv.
                        Accepts gs://... or a local path.
                        Default: "sp500_history.csv"
    MACRO_PATH          Full path to macro_history.csv.
                        Accepts gs://... or a local path.
                        Default: "macro_history.csv"
    CHUNK_DIR           Local directory for intermediate chunk files.
                        Default: "/tmp/data_chunks"
    GCS_OUTPUT_PREFIX   GCS key prefix for uploading checkpoints and the final
                        model weights inside GCS_BUCKET.
                        Default: "output"
    CHUNKED             Set to "1" to use memory-efficient chunked preprocessing.
                        Default: "0"
    RESUME              Set to "1" to resume from the latest checkpoint in GCS.
                        Default: "0"
    BATCH_SIZE          Training batch size. Default: 64
    EPOCHS              Maximum training epochs. Default: 100
    LR                  Initial learning rate. Default: 0.001
    PATIENCE            Early-stopping patience in epochs. Default: 10
    CHECKPOINT_DIR      Local directory for checkpoint files.
                        Default: "/tmp/checkpoints"

Example Vertex AI custom job worker-pool-spec environment variables:
    {
        "GCS_BUCKET": "my-stocks-nn",
        "SP500_PATH": "gs://my-stocks-nn/data/sp500_history.csv",
        "MACRO_PATH": "gs://my-stocks-nn/data/macro_history.csv",
        "GCS_OUTPUT_PREFIX": "runs/v1",
        "CHUNKED": "1",
        "EPOCHS": "200"
    }
"""

import os
import sys

import torch

from gcs import resolve, upload_file, upload_dir
from model import train_model, evaluate
from preprocess import prepare_data, prepare_data_chunked


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _resume_checkpoint_from_gcs(checkpoint_dir: str, gcs_bucket: str, gcs_prefix: str) -> str | None:
    """Download the latest checkpoint from GCS to *checkpoint_dir* and return its local path.

    Returns None if no checkpoint exists in GCS.
    """
    gcs_latest = f"gs://{gcs_bucket}/{gcs_prefix}/latest.pt"
    try:
        local_path = resolve(gcs_latest)
        # resolve() returns the input unchanged if the file is not found on GCS;
        # check that the file actually exists locally after the download attempt.
        if os.path.exists(local_path) and local_path != gcs_latest:
            os.makedirs(checkpoint_dir, exist_ok=True)
            dest = os.path.join(checkpoint_dir, "latest.pt")
            if local_path != dest:
                import shutil
                shutil.copy2(local_path, dest)
            print(f"[train] Resumed checkpoint from {gcs_latest}")
            return dest
    except Exception as exc:
        print(f"[train] Could not fetch checkpoint from GCS: {exc}")
    return None


def main() -> None:
    # ── Read configuration from environment ──────────────────────────────────
    gcs_bucket     = _env("GCS_BUCKET", "")
    gcs_prefix     = _env("GCS_OUTPUT_PREFIX", "output")
    chunked        = _env("CHUNKED", "0") == "1"
    resume         = _env("RESUME", "0") == "1"
    batch_size     = int(_env("BATCH_SIZE", "64"))
    epochs         = int(_env("EPOCHS", "100"))
    lr             = float(_env("LR", "0.001"))
    patience       = int(_env("PATIENCE", "10"))
    checkpoint_dir = _env("CHECKPOINT_DIR", "/tmp/checkpoints")

    print("=" * 55)
    print("  Stock Prediction Model — Vertex AI Training")
    print("=" * 55)
    print(f"  GCS bucket    : {gcs_bucket or '(local mode)'}")
    print(f"  Output prefix : {gcs_prefix}")
    print(f"  Chunked       : {chunked}")
    print(f"  Resume        : {resume}")
    print(f"  Batch size    : {batch_size}")
    print(f"  Epochs        : {epochs}")
    print(f"  LR            : {lr}")
    print(f"  Patience      : {patience}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print("-" * 55)

    # ── Optionally resume checkpoint from GCS ────────────────────────────────
    resume_from = None
    if resume and gcs_bucket:
        resume_from = _resume_checkpoint_from_gcs(checkpoint_dir, gcs_bucket, gcs_prefix)
    elif resume:
        # Local resume (checkpoint_dir/latest.pt)
        resume_from = "latest"

    # ── Prepare data ─────────────────────────────────────────────────────────
    print("\n[1/3] Preparing data...")
    if chunked:
        chunk_dir = _env("CHUNK_DIR", "/tmp/data_chunks")
        splits, scalers = prepare_data_chunked(chunk_dir=chunk_dir)
    else:
        splits, scalers = prepare_data()

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[2/3] Training model...")
    model = train_model(
        splits,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        patience=patience,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[3/3] Evaluating on test set...")
    results = evaluate(model, splits)

    # ── Save final model weights ───────────────────────────────────────────────
    weights_path = os.path.join(checkpoint_dir, "model_weights.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"\nModel saved to {weights_path}")

    if gcs_bucket:
        # Upload the final weights file
        upload_file(weights_path, f"gs://{gcs_bucket}/{gcs_prefix}/model_weights.pth")
        # Upload all checkpoints (epoch_*.pt and latest.pt)
        upload_dir(checkpoint_dir, f"gs://{gcs_bucket}/{gcs_prefix}/checkpoints")
        print(f"[train] All outputs uploaded to gs://{gcs_bucket}/{gcs_prefix}/")

    print("\nTraining complete.")
    print(f"  Close return MSE   : {results['mse_close']:.8f}")
    print(f"  Dividend yield MSE : {results['mse_div']:.10f}")
    print(f"  Direction accuracy : {results['dir_accuracy']:.2%}")


if __name__ == "__main__":
    main()
