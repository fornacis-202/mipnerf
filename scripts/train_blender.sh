#!/bin/bash
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script for training on the Blender dataset.

SCENE=$1  # Default to 'lego' if no argument is given
EXPERIMENT=debug
TRAIN_DIR=/kaggle/working/res/${SCENE}
DATA_DIR=/kaggle/input/nerf-dataset/nerf_synthetic/nerf_synthetic${SCENE}

# ===== Run training =====
echo "Training scene: ${SCENE}"
echo "Experiment: ${EXPERIMENT}"
echo "Train dir: ${TRAIN_DIR}"
echo "Data dir: ${DATA_DIR}"

python -m train \
  --data_dir="${DATA_DIR}" \
  --train_dir="${TRAIN_DIR}" \
  --gin_file=configs/blender.gin \
  --logtostderr
