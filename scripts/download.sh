#!/usr/bin/env bash
# Download Amazon Reviews'23 Industrial + Office
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="${ROOT}/data/raw"
mkdir -p "${RAW}"
cd "${RAW}"
BASE="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"

wget -c "${BASE}/review_categories/Industrial_and_Scientific.jsonl.gz"
wget -c "${BASE}/meta_categories/meta_Industrial_and_Scientific.jsonl.gz"
wget -c "${BASE}/review_categories/Office_Products.jsonl.gz"
wget -c "${BASE}/meta_categories/meta_Office_Products.jsonl.gz"
ls -lh
