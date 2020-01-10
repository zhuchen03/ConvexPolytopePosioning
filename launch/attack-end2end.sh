#!/usr/bin/env bash
python craft_poisons_transfer.py --gpu 2 --subs-chk-name ckpt-%s-4800-dp0.200-droplayer0.000-seed1226.t7 ckpt-%s-4800-dp0.250-droplayer0.000-seed1226.t7 ckpt-%s-4800-dp0.300-droplayer0.000.t7 --subs-dp 0.2 0.25 0.3  --substitute-nets DPN92 SENet18 ResNet50 ResNeXt29_2x64d --target-index 1 --target-label 6 --poison-label 8 --end2end True --retrain-lr 1e-4  --retrain-wd 5e-4  --target-net ResNet18

