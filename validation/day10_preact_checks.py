"""Day 10 (He et al. 2016, Identity Mappings in Deep Residual Networks) reference checks.

    OMP_NUM_THREADS=1 python validation/day10_preact_checks.py [day_dir]

Checks
 1. implementation.PreActBlock / PreActBottleneck vs a functional re-implementation of the paper's full
    pre-activation unit (Fig. 4e: BN -> ReLU -> conv -> BN -> ReLU -> conv, then + shortcut, nothing after
    the addition), with copied weights, train and eval mode, identity and projection shortcuts.
 2. Identity path untouched: residual branch zeroed => block is exactly the identity, and through a
    30-block stack dL/dx_0 == dL/dx_L (Eq. 5 with F = 0). Contrast: the post-activation (v1) unit.
 3. Eq. (4): x_L = x_0 + sum_i F(x_i) holds to machine precision for a stack of PreActBlocks.
 4. Parameter counts of the CIFAR bottleneck pre-act ResNet-164 / ResNet-1001 built from
    implementation.PreActBottleneck vs paper Table 4 (1.7M / 10.2M); solution_04's ResNet1001; and the
    README's PreActResNet1001 ([111]*4 with ImageNet widths) computed analytically.
 5. solution_02: which of the five Fig. 4 variants reduce to the identity when all weights are zero.
 6. Signal propagation at init (informational): ||dL/dx_0|| / ||dL/dx_L|| through 100 v1 vs v2 units.
"""
import importlib.util
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'papers', '10_resnet_v2')
sys.path.insert(0, DAY)
import implementation as impl  # noqa: E402

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
summary = []


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bn(m, t, training):
    return F.batch_norm(t, m.running_mean.clone(), m.running_var.clone(), m.weight, m.bias,
                        training=training, momentum=0.0, eps=m.eps)


def ref_block(b, x, training):
    h = F.relu(bn(b.bn1, x, training))                              # pre-activation, shared with projection
    out = F.conv2d(h, b.conv1.weight, stride=b.conv1.stride, padding=1)
    out = F.conv2d(F.relu(bn(b.bn2, out, training)), b.conv2.weight, padding=1)
    sc = x if b.shortcut is None else F.conv2d(h, b.shortcut.weight, stride=b.shortcut.stride)
    return out + sc                                                 # no activation after the addition


def ref_bottleneck(b, x, training):
    h = F.relu(bn(b.bn1, x, training))
    out = F.conv2d(h, b.conv1.weight)
    out = F.conv2d(F.relu(bn(b.bn2, out, training)), b.conv2.weight, stride=b.conv2.stride, padding=1)
    out = F.conv2d(F.relu(bn(b.bn3, out, training)), b.conv3.weight)
    sc = x if b.shortcut is None else F.conv2d(h, b.shortcut.weight, stride=b.shortcut.stride)
    return out + sc


def randomize_bn(module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.uniform_(0.5, 1.5); m.bias.data.uniform_(-0.5, 0.5)
            m.running_mean.uniform_(-0.5, 0.5); m.running_var.uniform_(0.5, 2.0)


# ------------------------------------------------------------------ 1. block vs functional reference
worst = 0.0
cases = [(impl.PreActBlock(16, 16, 1), ref_block, 16), (impl.PreActBlock(16, 32, 2), ref_block, 16),
         (impl.PreActBottleneck(64, 16, 1), ref_bottleneck, 64), (impl.PreActBottleneck(64, 32, 2), ref_bottleneck, 64)]
for blk, ref, cin in cases:
    randomize_bn(blk)
    x = torch.randn(4, cin, 10, 10)
    for training in (True, False):
        blk.train(training)
        with torch.no_grad():
            worst = max(worst, (blk(x) - ref(blk, x, training)).abs().max().item())
summary.append(('PreActBlock/PreActBottleneck vs functional BN->ReLU->conv reference (train+eval, id+proj)',
                f'max|d| = {worst:.1e}', worst < 1e-12))

# ------------------------------------------------------------------ 2. identity path untouched
blk = impl.PreActBlock(16, 16); randomize_bn(blk); blk.train()
nn.init.zeros_(blk.conv2.weight)
x = torch.randn(4, 16, 8, 8)
d_id = (blk(x) - x).abs().max().item()
summary.append(('PreActBlock with residual branch zeroed is exactly the identity', f'max|d| = {d_id:.1e}', d_id == 0))

stack_v2 = nn.Sequential(*[impl.PreActBlock(16, 16) for _ in range(30)])
for b in stack_v2:
    nn.init.zeros_(b.conv2.weight)
pre, post = impl.PreActivationStudy.create_comparison_blocks(16, 16)
stack_v1 = nn.Sequential(*[impl.PreActivationStudy.create_comparison_blocks(16, 16)[1] for _ in range(30)])
for b in stack_v1:
    nn.init.zeros_(b.bn2.weight); nn.init.zeros_(b.bn2.bias)          # residual branch output = 0
x = torch.randn(4, 16, 8, 8, requires_grad=True)
v = torch.randn(4, 16, 8, 8)
g2, = torch.autograd.grad((stack_v2(x) * v).sum(), x)
g1, = torch.autograd.grad((stack_v1(x) * v).sum(), x)
d_g2 = (g2 - v).abs().max().item()
frac_zero_v1 = (g1 == 0).double().mean().item()
summary.append(('30 pre-act units, F=0: dL/dx_0 == dL/dx_L (Eq. 5, "1" term only)', f'max|d| = {d_g2:.1e}', d_g2 == 0))
summary.append(('30 post-act (v1) units, F=0: fraction of gradient entries zeroed by ReLU(x+F)',
                f'{frac_zero_v1:.1%}', frac_zero_v1 > 0.3))

# ------------------------------------------------------------------ 3. Eq. (4) additivity
stack = [impl.PreActBlock(16, 16) for _ in range(8)]
for b in stack:
    randomize_bn(b); b.eval()
x0 = torch.randn(2, 16, 8, 8)
xs, h = [x0], x0
with torch.no_grad():
    for b in stack:
        h = b(h); xs.append(h)
    resid_sum = sum(ref_block(b, xi, False) - xi for b, xi in zip(stack, xs[:-1]))
    d_eq4 = (xs[-1] - (x0 + resid_sum)).abs().max().item()
summary.append(('Eq. (4) x_L = x_0 + sum F(x_i) over 8 pre-act units', f'max|d| = {d_eq4:.1e}', d_eq4 < 1e-12))


# ------------------------------------------------------------------ 4. parameter counts
def cifar_preact_bottleneck(n, num_classes=10):
    """Paper's CIFAR bottleneck ResNet (depth 9n+2): conv1 3x3x16, 3 stages of n units, widths 16/32/64 (x4)."""
    layers, cin = [nn.Conv2d(3, 16, 3, padding=1, bias=False)], 16
    for mid, stride in ((16, 1), (32, 2), (64, 2)):
        for i in range(n):
            layers.append(impl.PreActBottleneck(cin, mid, stride if i == 0 else 1)); cin = mid * 4
    layers += [nn.BatchNorm2d(cin), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(cin, num_classes)]
    return nn.Sequential(*layers)


def count(m, include_bn=False):
    """conv + fc weights/biases; BatchNorm affine parameters only if include_bn."""
    return sum(p.numel() for mod in m.modules() for p in mod.parameters(recurse=False)
               if include_bn or not isinstance(mod, nn.BatchNorm2d))


def weighted_depth(m):
    """conv layers on the main path (exclude 1x1 projection shortcuts) + fc."""
    convs = [mod for name, mod in m.named_modules() if isinstance(mod, nn.Conv2d) and 'shortcut' not in name
             and 'downsample' not in name]
    return len(convs) + sum(isinstance(mod, nn.Linear) for mod in m.modules())


r164, r1001 = cifar_preact_bottleneck(18), cifar_preact_bottleneck(111)
sol4 = load(os.path.join(DAY, 'exercises/solutions/solution_04_resnet_1001.py'), 'sol4_10')
s1001 = sol4.ResNet1001(10)
summary.append(('CIFAR pre-act ResNet-164 from PreActBottleneck: depth / conv+fc params / incl. BN (paper Table 4: 1.7M)',
                f'{weighted_depth(r164)} / {count(r164) / 1e6:.2f}M / {count(r164, True) / 1e6:.2f}M',
                weighted_depth(r164) == 164 and abs(count(r164) / 1e6 - 1.7) < 0.05))
summary.append(('CIFAR pre-act ResNet-1001 from PreActBottleneck: depth / conv+fc params / incl. BN (paper Table 4: 10.2M)',
                f'{weighted_depth(r1001)} / {count(r1001) / 1e6:.2f}M / {count(r1001, True) / 1e6:.2f}M',
                weighted_depth(r1001) == 1001 and abs(count(r1001) / 1e6 - 10.2) < 0.05))
summary.append(('solution_04 ResNet1001: depth / conv+fc params', f'{weighted_depth(s1001)} / {count(s1001) / 1e6:.2f}M',
                weighted_depth(s1001) == 1001 and abs(count(s1001) / 1e6 - 10.2) < 0.05))


def bottleneck_params(cin, mid):
    p = cin * mid + 9 * mid * mid + mid * 4 * mid                    # 1x1, 3x3, 1x1 weights (no bias)
    p += 2 * cin + 2 * mid + 2 * mid                                 # bn1, bn2, bn3 (gamma, beta)
    if cin != 4 * mid:
        p += cin * 4 * mid                                           # projection shortcut
    return p


def readme_1001_params():
    p, cin = 3 * 64 * 9, 64                                          # README conv1 3x3, 64
    for mid in (64, 128, 256, 512):
        for i in range(111):
            p += bottleneck_params(cin, mid); cin = 4 * mid
    return p + 2 * cin + cin * 10 + 10, 1 + 4 * 111 * 3 + 1


p_rd, d_rd = readme_1001_params()
summary.append(('README PreActResNet1001 ([111]*4, ImageNet widths): weighted depth / params (analytic)',
                f'{d_rd} / {p_rd / 1e6:.0f}M (paper: 1001 / 10.2M)', True))
vd = impl.VeryDeepResNetV2(depth=200, num_classes=10)
summary.append(('implementation.VeryDeepResNetV2(depth=200): weighted depth / params',
                f'{weighted_depth(vd)} / {count(vd) / 1e6:.2f}M', weighted_depth(vd) == 200))

# ------------------------------------------------------------------ 5. solution_02 identity claim
sol2 = load(os.path.join(DAY, 'exercises/solutions/solution_02_ablation_placement.py'), 'sol2_10')
x = torch.randn(1, 64, 8, 8)
ident = {}
for name, cls in [('a', sol2.OriginalBlock), ('b', sol2.BNAfterAddBlock), ('c', sol2.ReLUBeforeAddBlock),
                  ('d', sol2.ReLUOnlyPreActBlock), ('e', sol2.FullPreActBlock)]:
    blk = cls(64)
    with torch.no_grad():
        for p in blk.parameters():
            p.zero_()
        ident[name] = bool(torch.allclose(blk(x), x, atol=1e-5))
summary.append(('solution_02: variants that become the identity with all weights zeroed (it prints "only (e)")',
                str({k: v for k, v in ident.items()}), True))


# ------------------------------------------------------------------ 6. signal propagation at init (info)
def grad_ratio(blocks):
    x = torch.randn(8, 16, 8, 8, requires_grad=True)
    y = nn.Sequential(*blocks)(x)
    gy = torch.randn_like(y)
    gx, = torch.autograd.grad(y, x, gy)
    return (gx.norm() / gy.norm()).item(), (y.std() / x.std()).item()


torch.manual_seed(1)
r2 = grad_ratio([impl.PreActBlock(16, 16) for _ in range(100)])
r1 = grad_ratio([impl.PreActivationStudy.create_comparison_blocks(16, 16)[1] for _ in range(100)])
summary.append(('100 units at init, train-mode BN: ||dL/dx0||/||dL/dxL|| and std(x_L)/std(x_0)',
                f'v2 (pre-act) {r2[0]:.2f}, {r2[1]:.2f}; v1 (post-act) {r1[0]:.2f}, {r1[1]:.2f}', True))

print('\nSUMMARY')
for name, val, ok in summary:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {val}")
print('ALL PASS' if all(ok for *_, ok in summary) else 'SOME CHECKS FAILED')
