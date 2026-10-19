"""
Solution 03: Reproducing Posterior Collapse

Experiment showing how a high-capacity decoder without architectural 
constraints chooses the 'easy' path of ignoring the latent code z.

[PATCH 2026-10-18] What this run shows is the plain ELBO's collapse: the encoder starts near
q(z|x) = N(0, I) (KL ~0) and the KL never rises. The 3-layer decoder collapses the same way
without free bits, so depth is not what causes it here; free bits (solution 05,
implementation.free_bits_loss) is the paper's fix. CPU-sized default: MAX_BATCHES batches
(one full epoch of the 8-layer model takes over half an hour on one core); 0 = full epoch.
"""

import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementation import VLAE, loss_function

MAX_BATCHES = 60   # [PATCH 2026-10-18] CPU-sized default; 0 = the full epoch

def train_collapsing_vae():
    batch_size = 64
    epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: (x > 0.5).float()
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # To ensure collapse, use many decoder layers (e.g. 8) and NO flow prior
    model = VLAE(input_dim=1, latent_dim=32, n_layers=8, use_flow=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"Training High-Capacity VAE (Check for Collapse) on {device}...")
    
    kl_history = []
    
    for epoch in range(epochs):
        for batch_idx, (x, _) in enumerate(loader):
            if MAX_BATCHES and batch_idx >= MAX_BATCHES:
                break
            x = x.to(device)
            
            # VLAE forward returns 5 items
            logits, mu, logvar, z, log_det = model(x)
            
            # VLAE loss_function expects 6 items
            loss = loss_function(logits, x, mu, logvar, z, log_det)
            
            # Monitor KL component
            with torch.no_grad():
                # Analytical KL for Gaussian
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_history.append(kld.item() / x.size(0))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:   # [PATCH 2026-10-18] was every 100 batches
                print(f"Batch {batch_idx} Loss: {loss.item()/x.size(0):.4f} KL: {kl_history[-1]:.4f}")

    print(f"Training complete. KL over the last 10 batches: {sum(kl_history[-10:]) / 10:.4f} nats/image")
    
    plt.plot(kl_history)
    plt.title("KL Divergence (Low/Descending = Collapse)")
    plt.xlabel("Step")
    plt.ylabel("KL (nats)")
    plt.show()

if __name__ == "__main__":
    train_collapsing_vae()
