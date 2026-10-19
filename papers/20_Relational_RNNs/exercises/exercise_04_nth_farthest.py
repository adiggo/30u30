import torch

# [PATCH 2026-10-18] rewritten for the paper's labelled task (Santoro et al. 2018, Sec. 4.1, App. A.1).
# The old version regressed "the vector farthest from vector 0"; now n and m are sampled per sequence,
# the vectors carry shuffled labels, and the answer is a label (a K-way classification).


def generate_nth_farthest_batch(batch_size, num_objects=8, num_features=16, fixed_n=None):
    """
    Generates a batch of the Nth Farthest task: "What is the n-th farthest vector from vector m?"

    Each sequence shows K = num_objects random vectors x ~ U(-1, 1)^D, one per time step. Vector k
    carries the label k (one-hot); the objects are shown in a random order, so a label does not tell
    you when the vector arrives. Every step also carries the question (n, m) as two one-hots.
    Per step: (x; label; n; m) -> D + 3K numbers (40 for the paper's K = 8, D = 16).

    Counting is 0-based: n = 0 asks for the farthest vector and n = K-1 for the nearest - which is
    vector m itself (distance 0). fixed_n: ask every sequence for this rank (the curriculum of
    exercise 5); None samples n uniformly, as in the paper.

    Returns:
        inputs: (batch_size, K, D + 3K) float tensor
        targets: (batch_size,) long tensor - the label of the n-th farthest vector from vector m
    """
    K, D = num_objects, num_features
    eye = torch.eye(K)                                    # row k = one-hot of k
    inputs = torch.zeros(batch_size, K, D + 3 * K)
    targets = torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        vectors = torch.rand(K, D) * 2 - 1                # (K, D); vector k has label k
        n = torch.randint(0, K, ()).item()                # rank asked for (0 = farthest)
        if fixed_n is not None:
            n = fixed_n
        m = torch.randint(0, K, ()).item()                # label of the reference vector

        # TODO: Euclidean distance from every vector to vector m -> shape (K,)
        dists = None  # YOUR CODE HERE

        # TODO: labels sorted from farthest to nearest (torch.argsort(..., descending=True));
        # the target is the label at position n
        ranking = None  # YOUR CODE HERE
        target = None  # YOUR CODE HERE
        if target is None:  # [PATCH 2026-10-18] delete this guard once done
            raise NotImplementedError("Exercise 4: fill in dists, ranking, target, seq and order")

        # TODO: the K x (D + 3K) sequence: row k = (vectors[k], eye[k], eye[n], eye[m]);
        # hint: torch.cat([vectors, eye, eye[n].expand(K, K), eye[m].expand(K, K)], dim=1)
        seq = None  # YOUR CODE HERE

        # TODO: show the rows in a random order (torch.randperm(K))
        order = None  # YOUR CODE HERE

        inputs[b] = seq[order]
        targets[b] = target

    return inputs, targets


def test_generator():
    print("Testing Data Generator...")
    B, K, D = 200, 5, 3
    inputs, targets = generate_nth_farthest_batch(B, K, D)
    assert inputs.shape == (B, K, D + 3 * K)
    assert targets.shape == (B,) and targets.dtype == torch.long

    for b in range(B):
        seq = inputs[b]
        labels = seq[:, D:D + K].argmax(1)
        n = seq[0, D + K:D + 2 * K].argmax().item()
        m = seq[0, D + 2 * K:].argmax().item()
        assert sorted(labels.tolist()) == list(range(K)), "labels must be a permutation of 0..K-1"
        assert (seq[:, D + K:] == seq[0, D + K:]).all(), "n and m must be repeated at every step"
        # Recompute the answer from the inputs alone
        ref = seq[labels.tolist().index(m), :D]
        dist = {labels[t].item(): (seq[t, :D] - ref).norm().item() for t in range(K)}
        ranked = sorted(dist, key=lambda label: -dist[label])        # labels, farthest first
        assert targets[b].item() == ranked[n], f"sequence {b}: expected label {ranked[n]}, got {targets[b].item()}"
        if n == K - 1:
            assert targets[b].item() == m, "the K-th farthest (n = K-1) is vector m itself"

    order_varies = len({tuple(inputs[b, :, D:D + K].argmax(1).tolist()) for b in range(B)}) > 1
    assert order_varies, "the presentation order must be random"
    print("Test Passed!")


if __name__ == "__main__":
    test_generator()
