from __future__ import annotations

import logging

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


def main():
    snapshot_download(
        repo_id="Deargen/rust-graph",
        repo_type="dataset",
        revision="b30677dd3a76a0ecf51944861adacf433e9ecc04",
        local_dir="data",
        etag_timeout=1200,  # 처음 download 시, 대략 10분 가량 소모 되어 그 2배로 설정(default: 10s)
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Exception occurred")
