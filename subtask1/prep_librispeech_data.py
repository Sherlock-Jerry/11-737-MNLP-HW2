#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
import os
import torchaudio

import pandas as pd
from data_utils import (
    create_zip,
    extract_fbank_features,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv,
)
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm


log = logging.getLogger(__name__)

# SPLITS = [
#     "train-clean-100",
#     "train-clean-360",
#     "train-other-500",
#     "dev-clean",
#     "dev-other",
#     "test-clean",
#     "test-other",
# ]

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]

SPLITS = ['train', 'dev', 'test']
# SPLITS = ['train']

def process(args):
    data_dir = Path(args.data_dir).absolute()
    out_root = Path(args.output_root).absolute()
    out_root.mkdir(exist_ok=True)
    # Extract features
    feature_root = out_root / "commonvoice"
    feature_root.mkdir(exist_ok=True)
    for split in SPLITS:
        print(f"Fetching split {split}...")
        with open(os.path.join(data_dir, split+'.tsv'), 'r', encoding='utf-8') as f:
            dataset = [i.strip() for i in f.readlines()[1:]]
            print(split, len(dataset))
        print("Extracting log mel filter bank features...")
        for line in tqdm(dataset):
            client_id, path, sentence = line.split('\t')[:3]
            sample_id = client_id
            wav, sample_rate = torchaudio.load(os.path.join(data_dir, 'clips', path))
            extract_fbank_features(
                wav, sample_rate, feature_root / f"{sample_id}.npy"
            )
    # Pack features into ZIP
    zip_path = out_root / "commonvoice.zip"
    print("ZIPing features...")
    create_zip(feature_root, zip_path)
    print("Fetching ZIP manifest...")
    audio_paths, audio_lengths = get_zip_manifest(zip_path)
    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    for split in SPLITS:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        with open(os.path.join(data_dir, split+'.tsv'), 'r', encoding='utf-8') as f:
            dataset = [i.strip() for i in f.readlines()[1:]]
        for line in tqdm(dataset):
            client_id, path, sentence = line.split('\t')[:3]
            sample_id = client_id
            utt = sentence
            spk_id = 'N/A'
            manifest["id"].append(sample_id)
            manifest["audio"].append(audio_paths[sample_id])
            manifest["n_frames"].append(audio_lengths[sample_id])
            manifest["tgt_text"].append(utt.lower())
            manifest["speaker"].append(spk_id)
        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest), out_root / f"{split}.tsv"
        )
        if split.startswith("train"):
            train_text.extend(manifest["tgt_text"])

    # Generate vocab
    print("Generating Vocab")
    print(len(train_text))
    vocab_size = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size}"
    with NamedTemporaryFile(mode="w") as f:
        for t in tqdm(train_text, desc='Writing train_text', ncols=100):
            f.write(t + "\n")
        gen_vocab(
            Path(f.name),
            out_root / spm_filename_prefix,
            args.vocab_type,
            args.vocab_size,
        )
    # Generate config YAML
    gen_config_yaml(
        out_root,
        spm_filename=spm_filename_prefix + ".model",
        specaugment_policy="ld"
    )
    # Clean up
    shutil.rmtree(feature_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", "-d", required=True, type=str)
    parser.add_argument("--output-root", "-o", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=10000, type=int)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
