"""Arrange HAM10000 into the train/test image folders read by scripts/train.py.

Download HAM10000 (e.g. https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 or
https://doi.org/10.7910/DVN/DBW86T) and point --source at the folder holding HAM10000_metadata.csv
and the .jpg images (subfolders are searched).

    python scripts/prepare_ham10000.py --source ~/Downloads/ham10000 --dest data/HAM10000
"""

import argparse
import glob
import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split

# Diagnosis codes in the class-index order used for the paper's experiments.
CLASSES = ["akiec", "bcc", "bkl", "df", "nv", "vasc", "mel"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", default="data/HAM10000")
    parser.add_argument("--seed", type=int, default=2024, help="seed of the stratified 80/20 train/test split")
    args = parser.parse_args()

    images = {os.path.splitext(os.path.basename(p))[0]: p
              for p in glob.glob(os.path.join(args.source, "**", "*.jpg"), recursive=True)}
    metadata = pd.read_csv(os.path.join(args.source, "HAM10000_metadata.csv"))
    missing = set(metadata["image_id"]) - set(images)
    if missing:
        raise SystemExit(f"{len(missing)} images listed in the metadata were not found under {args.source}")
    metadata["label"] = metadata["dx"].map(CLASSES.index)

    train, test = train_test_split(metadata, test_size=0.2, random_state=args.seed, stratify=metadata["label"])
    for split, frame in (("train", train), ("test", test)):
        for image_id, label in zip(frame["image_id"], frame["label"]):
            folder = os.path.join(args.dest, split, str(label))
            os.makedirs(folder, exist_ok=True)
            shutil.copy(images[image_id], folder)
        print(f"{split}: {len(frame)} images in {os.path.join(args.dest, split)}")


if __name__ == "__main__":
    main()
