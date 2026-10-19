"""Day 11 (Yu & Koltun 2015, dilated convolutions) reference checks.

    OMP_NUM_THREADS=1 python validation/day11_dilation_checks.py [day_dir]

Checks
 1. Receptive field of the paper's context module (Table 1: dilations 1,1,2,4,8,16,1 + 1x1) measured by
    back-propagating a delta from one output pixel; compare with RF = 1 + sum_l (k-1)*d_l and Table 1
    (3,5,9,17,33,65,67,67).  Also counts holes ("gridding") inside the RF square.
 2. implementation.ReceptiveFieldAnalyzer.calculate_receptive_field vs the measured RF on random
    (kernel, stride, dilation) stacks.
 3. A dilated conv equals an ordinary conv with a zero-inserted ("a trous") kernel; DilatedConv2d('same')
    preserves H x W.
 4. DilatedResNet(output_stride=8): dilation schedule vs the standard rule (layer3 d=2, layer4 d=4) and the
    receptive-field consequence.
 5. train_minimal.DilatedSegmentationTrainer.evaluate_model mIoU vs a correct confusion-matrix mIoU.
 6. WaveNetBlock causality and WaveNet receptive field (kernel 2, dilations 1..512 -> 1024).
"""
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'papers', '11_dilated_convolutions')
sys.path.insert(0, DAY)
os.environ.setdefault('MPLBACKEND', 'Agg')
import implementation as impl  # noqa: E402

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
summary = []


def measured_rf(layers, size=401):
    """RF of the centre output of a 1-channel linear conv stack with all-ones kernels (no cancellation).
    layers: list of (k, stride, dilation). Returns (rf_width, n_nonzero, n_in_square)."""
    x = torch.zeros(1, 1, size, size, requires_grad=True)
    h = x
    for k, s, d in layers:
        h = F.conv2d(h, torch.ones(1, 1, k, k), stride=s, dilation=d)
    c = h.shape[-1] // 2
    h[0, 0, c, c].backward()
    g = x.grad[0, 0]
    rows = torch.nonzero(g.abs().sum(1)).flatten()
    cols = torch.nonzero(g.abs().sum(0)).flatten()
    width = int(rows.max() - rows.min() + 1)
    assert width == int(cols.max() - cols.min() + 1)
    sq = g[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    return width, int((sq != 0).sum()), sq.numel()


def formula_rf(layers):
    rf, jump = 1, 1
    for k, s, d in layers:
        rf += (k - 1) * d * jump
        jump *= s
    return rf


# ------------------------------------------------------------------ 1. context module (paper Table 1)
ctx = [(3, 1, 1), (3, 1, 1), (3, 1, 2), (3, 1, 4), (3, 1, 8), (3, 1, 16), (3, 1, 1), (1, 1, 1)]
table1 = [3, 5, 9, 17, 33, 65, 67, 67]
meas = [measured_rf(ctx[:i + 1])[0] for i in range(len(ctx))]
summary.append(('Context module RF per layer (measured) vs paper Table 1', f'{meas} vs {table1}', meas == table1))
w, nz, tot = measured_rf(ctx)
summary.append(('Context module: holes inside the 67x67 RF', f'{tot - nz} of {tot}', tot - nz == 0))
# README/exercise version 1,1,2,4,8,16,32,64 (+1x1)
readme = [(3, 1, d) for d in (1, 1, 2, 4, 8, 16, 32, 64)] + [(1, 1, 1)]
summary.append(('README "context module" (1,1,2,4,8,16,32,64) RF', f'{formula_rf(readme)} (paper: 67)', True))
for name, ds in [('1,2,4,8', (1, 2, 4, 8)), ('2,4,8', (2, 4, 8)), ('2,2,2', (2, 2, 2)), ('1,2,5 (HDC)', (1, 2, 5))]:
    lay = [(3, 1, d) for d in ds]
    w, nz, tot = measured_rf(lay)
    summary.append((f'stack d=[{name}]: RF {w}x{w}, positions that reach the output', f'{nz}/{tot}',
                    w == formula_rf(lay)))

# ------------------------------------------------------------------ 2. analyzer formula vs measurement
rng = np.random.default_rng(0)
worst = 0
for _ in range(25):
    L = rng.integers(1, 5)
    lay = [(int(rng.choice([1, 2, 3, 5])), int(rng.choice([1, 1, 2])), int(rng.choice([1, 2, 3, 4]))) for _ in range(L)]
    cfg = [{'kernel': k, 'stride': s, 'dilation': d} for k, s, d in lay]
    pred = impl.ReceptiveFieldAnalyzer.calculate_receptive_field(cfg)[-1]
    worst = max(worst, abs(pred - measured_rf(lay)[0]))
summary.append(('ReceptiveFieldAnalyzer vs measured RF on 25 random stacks (max |diff|)', str(worst), worst == 0))

# ------------------------------------------------------------------ 3. dilation == zero-inserted kernel; same padding
x = torch.randn(2, 3, 23, 19)
wgt = torch.randn(4, 3, 3, 3)
worst3 = 0.0
for d in (1, 2, 3, 4):
    wd = torch.zeros(4, 3, 2 * d + 1, 2 * d + 1)
    wd[:, :, ::d, ::d] = wgt                                   # insert d-1 zeros between taps
    worst3 = max(worst3, (F.conv2d(x, wgt, dilation=d) - F.conv2d(x, wd)).abs().max().item())
summary.append(('F.conv2d(dilation=d) vs conv with zero-inserted kernel, d=1..4', f'{worst3:.1e}', worst3 < 1e-12))
same = all(impl.DilatedConv2d(3, 4, 3, dilation=d)(x).shape[-2:] == x.shape[-2:] for d in (1, 2, 4, 8))
summary.append(("DilatedConv2d(padding='same') keeps HxW for d=1,2,4,8", str(same), same))

# ------------------------------------------------------------------ 4. DilatedResNet dilation schedule
net = impl.DilatedResNet(impl.DilatedResidualBlock, [2, 2, 2, 2], num_classes=10, output_stride=8)
dil = {name: sorted({b.conv2.dilation for b in getattr(net, name)}) for name in ('layer2', 'layer3', 'layer4')}
feat = nn.Sequential(net.conv1, net.bn1, nn.ReLU(), net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4)
with torch.no_grad():
    fshape = tuple(feat(torch.randn(1, 3, 224, 224)).shape)
ok4 = dil['layer3'] == [2] and dil['layer4'] == [4]
summary.append(('DilatedResNet(OS=8) conv dilations (expected layer3=2, layer4=4)', f'{dil}; feature map {fshape}', ok4))


def path(net_dils):
    """Main-path (k, s, d) list: conv1 7/2, maxpool 3/2, then 2 convs per block."""
    lay = [(7, 2, 1), (3, 2, 1)]
    for li, (n_blocks, first_s, d_first, d_rest) in enumerate(net_dils):
        for b in range(n_blocks):
            lay.append((3, first_s if b == 0 else 1, d_first if b == 0 else d_rest))
            lay.append((3, 1, d_rest))
    return lay


strided = path([(2, 1, 1, 1), (2, 2, 1, 1), (2, 2, 1, 1), (2, 2, 1, 1)])
correct8 = path([(2, 1, 1, 1), (2, 2, 1, 1), (2, 1, 1, 2), (2, 1, 2, 4)])      # torchvision-style
pre_fix = path([(2, 1, 1, 1), (2, 2, 1, 1), (2, 1, 2, 2), (2, 1, 2, 2)])        # schedule before the 2026-10-18 patch
as_built = [(7, 2, 1), (3, 2, 1)]                                                # read from the constructed network
for name in ('layer1', 'layer2', 'layer3', 'layer4'):
    for b in getattr(net, name):
        as_built += [(3, b.conv1.conv.stride[0], b.conv1.dilation), (3, 1, b.conv2.dilation)]
summary.append(('Main-path RF: strided OS=32 / torchvision-style OS=8 / pre-fix OS=8 / as-built OS=8',
                f'{formula_rf(strided)} / {formula_rf(correct8)} / {formula_rf(pre_fix)} / {formula_rf(as_built)}',
                formula_rf(as_built) >= formula_rf(strided)))

# ------------------------------------------------------------------ 5. mIoU in train_minimal
spec = importlib.util.spec_from_file_location('tm11', os.path.join(DAY, 'train_minimal.py'))
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)


class Oracle(nn.Module):
    """Returns logits that reproduce a stored label map (perfect model) or a corrupted one."""
    def __init__(self, flip=0.0):
        super().__init__(); self.flip = flip
        self.g = torch.Generator().manual_seed(1)

    def forward(self, packed):
        img, mask = packed[:, :3], packed[:, 3].long()
        pred = mask.clone()
        if self.flip:
            r = torch.rand(mask.shape, generator=self.g) < self.flip
            pred[r] = torch.randint(0, 5, (int(r.sum()),), generator=self.g)
        return F.one_hot(pred, 5).permute(0, 3, 1, 2).double() * 10


def true_miou(model, loader):
    inter = np.zeros(5); union = np.zeros(5)
    with torch.no_grad():
        for xb, mb in loader:
            p = model(xb).argmax(1)
            for c in range(5):
                inter[c] += ((p == c) & (mb == c)).sum().item()
                union[c] += ((p == c) | (mb == c)).sum().item()
    return 100 * np.mean(inter[union > 0] / union[union > 0])


g = torch.Generator().manual_seed(0)
masks = torch.randint(0, 5, (50, 16, 16), generator=g)
packed = torch.cat([torch.randn(50, 3, 16, 16), masks[:, None].double()], 1)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(packed, masks), batch_size=8)
trainer = tm.DilatedSegmentationTrainer('cpu')
for flip in (0.0, 0.3):
    mdl = Oracle(flip)
    _, _, miou_script = trainer.evaluate_model(mdl, loader, nn.CrossEntropyLoss())
    mdl = Oracle(flip)
    miou_true = true_miou(mdl, loader)
    ok = abs(miou_script - miou_true) < 1e-6
    summary.append((f'evaluate_model mIoU vs confusion-matrix mIoU (corruption {flip:.0%})',
                    f'{miou_script:.1f}% vs {miou_true:.1f}%', ok))

# ------------------------------------------------------------------ 6. WaveNet causality and RF
blk = impl.WaveNetBlock(4, 4, kernel_size=2, dilation=8)
xa = torch.randn(1, 4, 64)
xb = xa.clone(); xb[..., 40:] += 100.0                       # change only the future t >= 40
ya, _ = blk(xa); yb, _ = blk(xb)
d_causal = (ya[..., :40] - yb[..., :40]).abs().max().item()
summary.append(('WaveNetBlock: outputs t<40 unchanged when inputs t>=40 change', f'{d_causal:.1e}', d_causal == 0))
wn = impl.WaveNetModel(in_channels=1, out_channels=1, residual_channels=8, skip_channels=8, num_blocks=10, num_layers=1)
xin = torch.randn(1, 1, 3000, requires_grad=True)
wn(xin)[0, 0, 2500].backward()
nzt = torch.nonzero(xin.grad[0, 0]).flatten()
rf_wn = int(nzt.max() - nzt.min() + 1)
summary.append(('WaveNet (k=2, d=1..512) measured RF (exercise_05 says 1023, WaveNet paper 1024)',
                f'{rf_wn}, last input used = t{int(nzt.max())}', rf_wn == 1024 and int(nzt.max()) == 2500))

print('\nSUMMARY')
for name, val, ok in summary:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {val}")
print('ALL PASS' if all(ok for *_, ok in summary) else 'SOME CHECKS FAILED')
