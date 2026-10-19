"""
Solution 05: Complete VLAE

Final integration of the AF prior and the restricted PixelCNN decoder, trained with free bits.

[PATCH 2026-10-18] Was "Successfully cures posterior collapse", but with the plain ELBO this model
collapses (KL to the prior -> ~1 nat after 100 batches). It now trains with the paper's free bits
(App. A: 0.01 nats per pixel, implementation.free_bits_loss), prints the true KL next to the
objective, and checks on test images that the decoder uses z. CPU-sized default: MAX_BATCHES
batches (3 full epochs take over an hour on one core); set MAX_BATCHES = 0 for full epochs.
"""

import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementation import VLAE, free_bits_loss, latent_usage

MAX_BATCHES = 150   # [PATCH 2026-10-18] total training batches; 0 = run all epochs in full
FREE_BITS = 0.01    # nats per data dimension (paper App. A); 0.0 trains the plain ELBO


def train_vlae_solution():
    batch_size = 64
    epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: (x > 0.5).float()
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Full Model
    model = VLAE(input_dim=1, latent_dim=32, n_layers=3, use_flow=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"Training Full VLAE on {device} (free bits {FREE_BITS} nats/pixel)...")

    step = 0
    for epoch in range(epochs):
        model.train()
        total_loss, total_kl, n_seen = 0, 0, 0
        for batch_idx, (x, _) in enumerate(loader):
            if MAX_BATCHES and step >= MAX_BATCHES:
                break
            x = x.to(device)

            # Forward returns 5 items
            logits, mu, logvar, z, log_det = model(x)

            # [PATCH 2026-10-18] Free-bits objective; parts['kl'] is the true KL (no max)
            loss, parts = free_bits_loss(model, logits, x, mu, logvar, z, free_bits=FREE_BITS)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += parts['recon'] + parts['kl']
            total_kl += parts['kl']
            n_seen += x.size(0)
            step += 1

            if batch_idx % 25 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(loader)}] objective {loss.item()/x.size(0):.2f} | "
                      f"-ELBO {(parts['recon'] + parts['kl'])/x.size(0):.2f} | KL {parts['kl']/x.size(0):.2f} nats/image")

        if n_seen:
            print(f"====> Epoch {epoch} ({n_seen} images) Average -ELBO: {total_loss / n_seen:.2f} | "
                  f"KL: {total_kl / n_seen:.2f} nats/image")
        if MAX_BATCHES and step >= MAX_BATCHES:
            break

    # [PATCH 2026-10-18] Is z used? Compare -log p(x|z) with each image's own z and with another's.
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    x_test = (test_set.data[:1000].float() / 255 > 0.5).float().unsqueeze(1).to(device)
    u = latent_usage(model, x_test)
    print(f"\nTest images: KL to the prior {u['kl']:.2f} nats (budget {FREE_BITS * 784:.2f}); "
          f"-log p(x|z) own z {u['nll_own']:.2f} vs another image's z {u['nll_shuffled']:.2f} "
          f"(gap {u['gap']:.2f}); I(x; z) estimate {u['mi']:.2f} nats")

    # Final Visualization (teacher-forced one-step predictions, not decompressions)
    model.eval()
    with torch.no_grad():
        x_sample, _ = next(iter(loader))
        x_sample = x_sample[:8].to(device)
        logits, _, _, _, _ = model(x_sample)
        recons = torch.sigmoid(logits)
        grid = torch.cat([x_sample, recons], dim=0)

        if not os.path.exists('results'):
            os.makedirs('results')
        save_image(grid, "results/final_vlae_comp.png", nrow=8)
        print("Reconstruction saved to results/final_vlae_comp.png")

if __name__ == "__main__":
    train_vlae_solution()
