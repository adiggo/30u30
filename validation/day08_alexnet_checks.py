"""Day 08 (Krizhevsky, Sutskever, Hinton 2012, AlexNet) reference checks.

    OMP_NUM_THREADS=1 python validation/day08_alexnet_checks.py [day_dir]
(check 6 needs CIFAR-10 in <day_dir>/data/cifar-10-batches-py)

Checks
 1. implementation.LocalResponseNorm vs the paper's Sec. 3.3 formula (direct loop) and vs
    F.local_response_norm (PyTorch divides alpha by n, so alpha_torch = n * alpha_paper).
 2. Layer shapes for a 224x224 input (55 -> 27 -> 27 -> 13 -> 13 -> 13 -> 13 -> 6).
 3. Parameter count: implementation (no grouping) vs the paper's two-GPU layout (conv2/4/5 see only
    the 48/192/192 channels on their own GPU = groups=2).
 4. README's bias-init snippet (indices [3, 6, 8]): which modules actually receive bias = 1.
 5. Signal scale at init (N(0, 0.01) weights, zero biases as implemented) vs paper's bias = 1 scheme.
 6. AlexNetTrainer constructs on this PyTorch (verbose=True removed) and the train_minimal CIFAR path
    trains: loss over the first SGD steps on real CIFAR-10 at 224x224.
"""
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'papers', '08_alexnet')
sys.path.insert(0, DAY)
import implementation as impl  # noqa: E402

torch.manual_seed(0)
summary = []

# ------------------------------------------------------------------ 1. LRN
x = torch.randn(2, 12, 5, 5, dtype=torch.float64) * 3
lrn = impl.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
ref = torch.empty_like(x)
N, n = x.shape[1], 5
for i in range(N):                              # paper: sum over j = max(0, i-n/2) .. min(N-1, i+n/2)
    lo, hi = max(0, i - n // 2), min(N - 1, i + n // 2)
    ref[:, i] = x[:, i] / (2.0 + 1e-4 * (x[:, lo:hi + 1] ** 2).sum(1)) ** 0.75
d_paper = (lrn(x) - ref).abs().max().item()
d_torch = (lrn(x) - F.local_response_norm(x, 5, alpha=5 * 1e-4, beta=0.75, k=2.0)).abs().max().item()
d_readme = (F.local_response_norm(x, 5, alpha=1e-4, beta=0.75, k=2.0) - ref).abs().max().item()
summary.append(('LocalResponseNorm vs paper Sec. 3.3 formula (loop)', f'max|d| = {d_paper:.1e}', d_paper < 1e-12))
summary.append(('LocalResponseNorm vs F.local_response_norm(alpha = 5 * 1e-4)', f'max|d| = {d_torch:.1e}', d_torch < 1e-12))
summary.append(("README's nn.LocalResponseNorm(alpha=1e-4) vs paper formula (alpha effectively 2e-5)",
                f'max|d| = {d_readme:.2e}', True))

# ------------------------------------------------------------------ 2. shapes
model = impl.AlexNet(num_classes=1000)
h = torch.randn(1, 3, 224, 224)
sizes = []
for layer in model.features:
    h = layer(h)
    if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
        sizes.append(h.shape[-1])
ok2 = sizes == [55, 27, 27, 13, 13, 13, 13, 6]
summary.append(('Spatial size after conv1,pool1,conv2,pool2,conv3,conv4,conv5,pool5 (224 input)', str(sizes), ok2))


# ------------------------------------------------------------------ 3. parameter count, grouped vs not
def count(m):
    return sum(p.numel() for p in m.parameters())


grouped = nn.Sequential(
    nn.Conv2d(3, 96, 11, 4, 2), nn.Conv2d(96, 256, 5, padding=2, groups=2),
    nn.Conv2d(256, 384, 3, padding=1), nn.Conv2d(384, 384, 3, padding=1, groups=2),
    nn.Conv2d(384, 256, 3, padding=1, groups=2),
    nn.Linear(9216, 4096), nn.Linear(4096, 4096), nn.Linear(4096, 1000))
summary.append(('Params: implementation (ungrouped) vs paper two-GPU layout (groups=2 in conv2/4/5)',
                f'{count(model):,} vs {count(grouped):,} (paper: "60 million")', True))

# ------------------------------------------------------------------ 4. README bias-init snippet
idx = [3, 6, 8]
hits = {i: type(model.features[i]).__name__ for i in idx}
conv_idx = [i for i, m in enumerate(model.features) if isinstance(m, nn.Conv2d)]
summary.append(("README init snippet sets bias=1 on features[3,6,8]; module types there / conv indices",
                f'{hits} / convs at {conv_idx} -> only conv3 (index 8) gets bias 1; paper wants conv2, conv4, conv5 + FC hidden',
                True))

# ------------------------------------------------------------------ 5. signal scale at init
torch.manual_seed(0)
xin = torch.randn(4, 3, 224, 224)


def feature_std(m):
    m.eval()
    with torch.no_grad():
        return m(xin).std().item()


zero_bias = impl.AlexNet(10)
paper_bias = impl.AlexNet(10)
convs = [m for m in paper_bias.features if isinstance(m, nn.Conv2d)]
for c in (convs[1], convs[3], convs[4]):
    nn.init.constant_(c.bias, 1.0)
for lin in [m for m in paper_bias.classifier if isinstance(m, nn.Linear)][:2]:
    nn.init.constant_(lin.bias, 1.0)
summary.append(('Logit std at init (unit-variance input): implemented zero biases vs paper bias=1 scheme',
                f'{feature_std(zero_bias):.2e} vs {feature_std(paper_bias):.2e}', True))

# ------------------------------------------------------------------ 6. trainer + first SGD steps on real CIFAR
import torchvision.transforms as T  # noqa: E402
import torchvision.datasets as D  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

tf = T.Compose([T.Resize(224), T.RandomCrop(224, padding=4), T.RandomHorizontalFlip(0.5), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
ds = D.CIFAR10(root=os.path.join(DAY, 'data'), train=True, download=False, transform=tf)
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
torch.manual_seed(0)
net = impl.AlexNet(num_classes=10, dropout=0.5)
try:
    trainer = impl.AlexNetTrainer(net, loader, loader, torch.device('cpu'), lr=0.001, momentum=0.9, weight_decay=5e-4)
    summary.append(('AlexNetTrainer constructs (ReduceLROnPlateau without verbose)', 'ok', True))
except TypeError as e:
    summary.append(('AlexNetTrainer constructs', f'TypeError: {e}', False))
    trainer = None
if trainer is not None:
    losses, t0 = [], time.time()
    net.train()
    for step, (xb, yb) in enumerate(loader):
        trainer.optimizer.zero_grad()
        loss = trainer.criterion(net(xb), yb)
        loss.backward()
        trainer.optimizer.step()
        losses.append(round(loss.item(), 4))
        if step == 11 or time.time() - t0 > 150:
            break
    per_step = (time.time() - t0) / len(losses)
    summary.append((f'train_minimal path, first {len(losses)} SGD steps on CIFAR-10 (lr 1e-3, batch 32)',
                    f'loss {losses}; {per_step:.1f} s/step on 1 thread -> one epoch (1563 steps) ~{per_step * 1563 / 3600:.1f} h',
                    all(l == l for l in losses)))

print('\nSUMMARY')
for name, val, ok in summary:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {val}")
print('ALL PASS' if all(ok for *_, ok in summary) else 'SOME CHECKS FAILED')
