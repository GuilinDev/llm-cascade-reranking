#!/usr/bin/env bash
# Download MovieLens-1M into data/raw/ml-1m/
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/raw}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

URL="https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ZIP="ml-1m.zip"

if [ -f "ml-1m/ratings.dat" ]; then
    echo "[movielens] already extracted in $DATA_DIR/ml-1m"
    exit 0
fi

if [ ! -f "$ZIP" ]; then
    echo "[movielens] downloading from $URL"
    curl -fL -o "$ZIP" "$URL"
fi

echo "[movielens] unzipping"
unzip -q -o "$ZIP"
rm -f "$ZIP"

echo "[movielens] done:"
ls -la ml-1m/
