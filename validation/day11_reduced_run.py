"""Day 11: run train_minimal.demonstrate_dilated_training() end to end in a reduced configuration.

train_minimal.py has no size/epoch CLI flags (only --quick, which trains nothing), and the default run
(200 synthetic 128x128 images, 8 epochs x 3 models + WaveNet) takes longer than 25 minutes on one CPU thread.
This driver keeps every code path but shrinks the workload:
  * synthetic images 48x48 instead of 128x128 (the generator needs size > 40)
  * 32 train / 12 test images instead of 200 / 50
  * 3 segmentation epochs instead of 8, 5 WaveNet epochs instead of 15

    OMP_NUM_THREADS=1 python validation/day11_reduced_run.py [day_dir]
(train_minimal saves PNGs into the current directory, so this driver first changes to
validation/outputs/day11_reduced_run/.)
"""
import importlib.util
import os
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'papers', '11_dilated_convolutions'))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'day11_reduced_run')
os.makedirs(OUT, exist_ok=True)
os.chdir(OUT)  # the PNGs train_minimal saves go here, not into the repo
sys.path.insert(0, DAY)
spec = importlib.util.spec_from_file_location('tm11', os.path.join(DAY, 'train_minimal.py'))
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)

import numpy as np  # noqa: E402
import torch  # noqa: E402

np.random.seed(0)
torch.manual_seed(0)

Orig = tm.SegmentationDatasetSimulator


class Small(Orig):
    def __init__(self, size=128, num_classes=5):
        super().__init__(size=48, num_classes=num_classes)

    def create_dataset(self, num_samples=100):
        # 32 = 4 full batches of 8 (a final batch of 1 would crash BatchNorm in ASPP's pooling branch)
        return super().create_dataset(32 if num_samples > 100 else 12)


tm.SegmentationDatasetSimulator = Small
_cmp = tm.DilatedSegmentationTrainer.compare_architectures
tm.DilatedSegmentationTrainer.compare_architectures = lambda self, epochs=10: _cmp(self, epochs=3)
_wn = tm.WaveNetDemo.train_wavenet_demo
tm.WaveNetDemo.train_wavenet_demo = lambda self, epochs=20: _wn(self, epochs=5)

tm.demonstrate_dilated_training()
print('\nREDUCED RUN COMPLETED END TO END')
