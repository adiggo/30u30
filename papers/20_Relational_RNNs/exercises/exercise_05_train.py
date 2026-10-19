import torch
import torch.nn.functional as F
import os, sys  # [PATCH 2026-10-18] make the day's implementation.py importable (ModuleNotFoundError before)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from implementation import RelationalRNN
from exercise_04_nth_farthest import generate_nth_farthest_batch

# [PATCH 2026-10-18] trains on the labelled Nth Farthest task with cross-entropy (was: MSE regression)
# and reports accuracy against the right baselines, split by question type.
#
# What to expect (3 vectors of 2 dims, chance 1/3): when n = K-1 the answer is m, which is in the
# input, so copying m gives 2/K = 0.67 overall with no reasoning. With n uniform from the first step
# (the paper's protocol) both the RMC and an LSTM sit at that level for a very long time: averaged over
# n, every label other than m is equally likely whatever the vectors are. A short curriculum - first ask
# only for the farthest vector - lets the RMC learn the distances; then all n take a few hundred steps.


def train(steps=2000, curriculum_steps=1500, batch_size=128, num_objects=3, num_features=2):
    K, D = num_objects, num_features
    torch.manual_seed(0)

    # TODO: Initialize the RelationalRNN: input_size = D + 3K, 4 memory slots of size 64, 4 heads,
    # output_size = K (one logit per label), and the paper's output MLP: head_hidden=(256, 256, 256, 256)
    model = None  # YOUR CODE HERE
    if model is None:  # [PATCH 2026-10-18] placeholders were `x = # ...` (SyntaxError); delete this guard once done
        raise NotImplementedError("Exercise 5: build the model, then fill in the batch, forward and loss TODOs")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"Starting Training ({K} vectors of {D} dims; chance {1 / K:.2f})...")

    for step in range(steps):
        # TODO: Generate a batch (inputs (B, K, D + 3K), targets (B,) labels). For the first
        # curriculum_steps steps ask only for the farthest vector (fixed_n=0), then every n (fixed_n=None)
        inputs, targets = None, None  # YOUR CODE HERE

        # TODO: Forward pass -> logits (B, K)
        logits = None  # YOUR CODE HERE

        # TODO: Cross-entropy loss, then backward and an optimizer step
        loss = None  # YOUR CODE HERE
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 250 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

    print("\nTraining Complete!")

    # Evaluation on fresh sequences with every n (the paper's question mix), split by question type
    model.eval()
    with torch.no_grad():
        inputs, targets = generate_nth_farthest_batch(2000, K, D)
        correct = model(inputs).argmax(-1) == targets
        n = inputs[:, 0, D + K:D + 2 * K].argmax(-1)             # the question's n, read from the input
        copy_m = n == K - 1
    print(f"Accuracy: {correct.float().mean():.3f} overall (chance {1 / K:.3f}, copying m when n = K-1: {2 / K:.3f})")
    print(f"  n = K-1 questions (answer = m): {correct[copy_m].float().mean():.3f}")
    print(f"  relational questions (n < K-1): {correct[~copy_m].float().mean():.3f} (chance {1 / (K - 1):.3f})")


if __name__ == "__main__":
    train()
