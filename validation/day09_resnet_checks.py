"""Day 09 (He et al. 2015, Deep Residual Learning) reference checks.

    OMP_NUM_THREADS=1 python validation/day09_resnet_checks.py [day_dir]

Checks
 1. implementation.create_resnet(...) vs torchvision.models.resnetXX: identical state_dict keys/shapes,
    parameter counts, and outputs (eval mode, float64) after loading torchvision's weights.
 2. BasicBlock / Bottleneck forward vs a functional re-implementation of Eq. (1)/(2) with the
    post-addition ReLU (option B projection when shapes change).
 3. Where the stride lives in the bottleneck (paper's original release: first 1x1; torchvision "V1.5": 3x3).
 4. Paper Table 6: parameter counts of the CIFAR ResNets (6n+2, widths 16/32/64, option-A zero-padding
    shortcuts) from an independent re-implementation; vs the ImageNet ResNet-18 that train_minimal trains.
 5. ResNetAnalyzer.analyze_skip_connections 'residual_norm' vs the true ||F(x)||.
 6. exercise_01.test_resnet with solution_01's BasicBlock/ResNet18 swapped in.
"""
import importlib.util
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'papers', '09_resnet')
sys.path.insert(0, DAY)
import implementation as impl  # noqa: E402

torch.manual_seed(0)
summary = []


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nparams(m):
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------------ 1. vs torchvision
for name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
    ours = impl.create_resnet(name, num_classes=1000)
    tv = getattr(torchvision.models, name)(weights=None)
    same_keys = [(k, tuple(v.shape)) for k, v in ours.state_dict().items()] == \
                [(k, tuple(v.shape)) for k, v in tv.state_dict().items()]
    detail = f'params {nparams(ours):,} vs {nparams(tv):,}; state_dict keys/shapes identical: {same_keys}'
    ok = same_keys and nparams(ours) == nparams(tv)
    if name in ('resnet18', 'resnet50'):
        for m in tv.modules():                       # make BN statistics non-trivial before comparing
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.uniform_(-0.2, 0.2); m.running_var.uniform_(0.5, 1.5)
                m.weight.data.uniform_(0.5, 1.5); m.bias.data.uniform_(-0.2, 0.2)
        ours.load_state_dict(tv.state_dict(), strict=True)
        ours, tv = ours.double().eval(), tv.double().eval()
        x = torch.randn(2, 3, 96, 96, dtype=torch.float64)
        with torch.no_grad():
            d = (ours(x) - tv(x)).abs().max().item()
        detail += f'; outputs max|d| = {d:.1e}'
        ok = ok and d < 1e-10
    summary.append((f'{name}: implementation vs torchvision', detail, ok))
    del ours, tv

# ------------------------------------------------------------------ 2. blocks vs Eq. (1)/(2)
torch.set_default_dtype(torch.float64)


def bnf(m, t):
    return F.batch_norm(t, m.running_mean, m.running_var, m.weight, m.bias, training=False, eps=m.eps)


def ref_basic(b, x):
    out = F.relu(bnf(b.bn1, F.conv2d(x, b.conv1.weight, stride=b.conv1.stride, padding=1)))
    out = bnf(b.bn2, F.conv2d(out, b.conv2.weight, padding=1))
    sc = x if b.downsample is None else bnf(b.downsample[1], F.conv2d(x, b.downsample[0].weight, stride=b.downsample[0].stride))
    return F.relu(out + sc)                                          # Eq. (1)/(2), then sigma(y) after the addition


def ref_bottleneck(b, x):
    out = F.relu(bnf(b.bn1, F.conv2d(x, b.conv1.weight)))
    out = F.relu(bnf(b.bn2, F.conv2d(out, b.conv2.weight, stride=b.conv2.stride, padding=1)))
    out = bnf(b.bn3, F.conv2d(out, b.conv3.weight))
    sc = x if b.downsample is None else bnf(b.downsample[1], F.conv2d(x, b.downsample[0].weight, stride=b.downsample[0].stride))
    return F.relu(out + sc)


net18 = impl.create_resnet('resnet18', 10).eval()
net50 = impl.create_resnet('resnet50', 10).eval()
for m in list(net18.modules()) + list(net50.modules()):
    if isinstance(m, nn.BatchNorm2d):
        m.running_mean.uniform_(-0.2, 0.2); m.running_var.uniform_(0.5, 1.5)
        m.weight.data.uniform_(0.5, 1.5); m.bias.data.uniform_(-0.2, 0.2)
worst = 0.0
with torch.no_grad():
    for blk in (net18.layer1[0], net18.layer2[0], net18.layer2[1]):
        x = torch.randn(2, blk.conv1.in_channels, 16, 16)
        worst = max(worst, (blk(x) - ref_basic(blk, x)).abs().max().item())
    for blk in (net50.layer1[0], net50.layer2[0], net50.layer2[1]):
        x = torch.randn(2, blk.conv1.in_channels, 16, 16)
        worst = max(worst, (blk(x) - ref_bottleneck(blk, x)).abs().max().item())
summary.append(('BasicBlock/Bottleneck vs functional Eq. (1)/(2) + post-addition ReLU (identity and projection)',
                f'max|d| = {worst:.1e}', worst < 1e-12))

# ------------------------------------------------------------------ 3. stride placement
b = net50.layer2[0]
summary.append(('Bottleneck downsampling stride: conv1 (1x1) / conv2 (3x3)',
                f'{b.conv1.stride} / {b.conv2.stride} -> torchvision "ResNet V1.5" placement; the original release '
                'strides in the first 1x1', True))


# ------------------------------------------------------------------ 4. CIFAR ResNets (paper Sec. 4.2, Table 6)
class OptionABlock(nn.Module):
    """Two 3x3 convs; identity shortcut, zero-padded channels + stride-2 subsampling when shapes change (option A)."""
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.stride, self.pad = stride, cout - cin

    def forward(self, x):
        sc = x[:, :, ::self.stride, ::self.stride]
        if self.pad:
            sc = F.pad(sc, (0, 0, 0, 0, 0, self.pad))                # zero-pad the extra channels
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + sc)


def cifar_resnet(n):
    layers, cin = [nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU()], 16
    for cout, s in ((16, 1), (32, 2), (64, 2)):
        for i in range(n):
            layers.append(OptionABlock(cin, cout, s if i == 0 else 1)); cin = cout
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10)]
    return nn.Sequential(*layers)


table6 = {20: 0.27, 32: 0.46, 44: 0.66, 56: 0.85, 110: 1.7, 1202: 19.4}
got = {}
for depth, ref in table6.items():
    net = cifar_resnet((depth - 2) // 6)
    with torch.no_grad():
        assert net(torch.randn(1, 3, 32, 32)).shape == (1, 10)
    got[depth] = nparams(net) / 1e6
ok4 = all(abs(got[d] - r) / r < 0.03 for d, r in table6.items())
summary.append(('Paper Table 6 CIFAR ResNet params (6n+2, option A): ours vs paper',
                ', '.join(f'{d}: {got[d]:.2f}M/{table6[d]}M' for d in table6), ok4))
summary.append(('train_minimal trains create_resnet("resnet18", 10) (ImageNet design on 224-upsampled CIFAR)',
                f'{nparams(impl.create_resnet("resnet18", 10)) / 1e6:.2f}M params vs 0.27M for the paper\'s ResNet-20', True))

# ------------------------------------------------------------------ 5. analyzer's "residual" vs true F(x)
torch.set_default_dtype(torch.float32)
net = impl.create_resnet('resnet18', 10).eval()
x = torch.randn(2, 3, 64, 64)
stats = impl.ResNetAnalyzer(net).analyze_skip_connections(x)
with torch.no_grad():
    h = net.maxpool(net.relu(net.bn1(net.conv1(x))))
    blk = net.layer1[0]
    true_F = blk.bn2(blk.conv2(blk.relu(blk.bn1(blk.conv1(h)))))
rep, tru = stats['layer1.0']['residual_norm'], true_F.norm().item()
summary.append(("ResNetAnalyzer 'residual_norm' (= ||ReLU(F+x) - x||) vs true ||F(x)|| for layer1.0",
                f'{rep:.2f} vs {tru:.2f}', True))

# ------------------------------------------------------------------ 6. exercise 1 vs solution 1
ex1 = load(os.path.join(DAY, 'exercises/exercise_01_build_resnet18.py'), 'ex1_09')
sol1 = load(os.path.join(DAY, 'exercises/solutions/solution_01_build_resnet18.py'), 'sol1_09')
ex1.BasicBlock, ex1.ResNet18 = sol1.BasicBlock, sol1.ResNet18
try:
    ex1.test_resnet()
    m = sol1.ResNet18(1000)
    ok6 = nparams(m) == nparams(torchvision.models.resnet18(weights=None))
    summary.append(('exercise_01.test_resnet with solution_01 (prints only; params == torchvision resnet18)',
                    f'{nparams(m):,} params', ok6))
except Exception as e:  # noqa: BLE001
    summary.append(('exercise_01.test_resnet with solution_01', f'{type(e).__name__}: {e}', False))

print('\nSUMMARY')
for name, val, ok in summary:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {val}")
print('ALL PASS' if all(ok for *_, ok in summary) else 'SOME CHECKS FAILED')
