"""
Exercise 1: Build a Character-Level RNN from Scratch
=====================================================

Your task: Fill in the TODOs to create a working RNN.

This will help you understand:
- How data flows through an RNN
- How gradients are computed via BPTT
- Why gradient clipping matters
"""

import numpy as np

class CharRNN:
    """
    A minimal character-level RNN for learning.
    
    TODO: Implement the missing methods below.
    """
    
    def __init__(self, vocab_size, hidden_size, seq_length, learning_rate=0.01):
        """
        Initialize the RNN.
        
        Args:
            vocab_size: Number of unique characters
            hidden_size: Size of hidden state vector
            seq_length: Number of characters to process at once
            learning_rate: Step size for gradient descent
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        
        # TODO: Initialize weight matrices
        # Hint: Use small random values, e.g., np.random.randn(...) * 0.01
        #
        # You need:
        # - Wxh: Input to hidden (vocab_size x hidden_size)
        # - Whh: Hidden to hidden (hidden_size x hidden_size)
        # - Why: Hidden to output (hidden_size x vocab_size)
        # - bh: Hidden bias (hidden_size x 1)
        # - by: Output bias (vocab_size x 1)
        
        self.Wxh = None  # TODO
        self.Whh = None  # TODO
        self.Why = None  # TODO
        self.bh = None   # TODO
        self.by = None   # TODO
        
    def forward(self, inputs, targets, h_prev):
        """
        Forward pass through the RNN.
        
        Args:
            inputs: List of character indices (length = seq_length)
            targets: List of target character indices, the next character after each input
                     (the loss needs them)
            h_prev: Previous hidden state (hidden_size x 1)
            
        Returns:
            loss: Cross-entropy loss, summed over the sequence
            h_last: Final hidden state
            cache: Values needed for backward pass
        """
        # [PATCH 2026-10-18] targets added: the loss -log p_t[target_t] cannot be computed without
        # them (same signature as solutions/exercise_01_solution.py). The repo's
        # implementation.CharRNN splits this into forward(inputs, hprev) and loss(ps, targets);
        # here, as in Karpathy's min-char-rnn, forward() returns the loss as well.
        
        # Storage for forward pass
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        loss = 0
        
        # TODO: Implement forward pass through time
        # For each time step t:
        #   1. Create one-hot encoding for inputs[t]
        #   2. Compute hidden state: h_t = tanh(Wxh @ x_t + Whh @ h_{t-1} + bh)
        #   3. Compute output: y_t = Why @ h_t + by
        #   4. Compute probabilities: p_t = softmax(y_t)
        #   5. Accumulate loss: -log(p_t[target_t])
        #
        # Hint: Store xs, hs, ys, ps in dictionaries for backward pass
        
        for t in range(len(inputs)):
            # TODO: Your code here
            pass
        
        return loss, hs[len(inputs)-1], (xs, hs, ys, ps)
    
    def backward(self, inputs, targets, cache):
        """
        Backward pass: compute gradients via BPTT.
        
        Args:
            inputs: List of character indices
            targets: List of target character indices
            cache: Values from forward pass (xs, hs, ys, ps)
            
        Returns:
            grads: Dictionary of gradients for all parameters
        """
        
        xs, hs, ys, ps = cache
        
        # Initialize gradients to zero
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        dh_next = np.zeros_like(hs[0])
        
        # TODO: Implement backward pass through time
        # For each time step t (in reverse order):
        #   1. Compute gradient of loss w.r.t. output: dy = p_t (copy), dy[target_t] -= 1
        #   2. Accumulate output layer gradients: dWhy, dby
        #   3. Backpropagate to hidden layer: dh = Why.T @ dy + dh_next
        #   4. Backpropagate through tanh: dh_raw = (1 - h_t^2) * dh
        #   5. Accumulate hidden layer gradients: dWxh, dWhh, dbh
        #   6. Propagate gradient to previous time step: dh_next = Whh.T @ dh_raw
        #   7. Clip gradients to prevent explosion
        #
        # Hint: Use np.clip(grad, -5, 5) for gradient clipping
        
        for t in reversed(range(len(inputs))):
            # TODO: Your code here
            pass
        
        # TODO: Clip gradients
        # for gradient in [dWxh, dWhh, dWhy, dbh, dby]:
        #     np.clip(gradient, -5, 5, out=gradient)
        
        return {'Wxh': dWxh, 'Whh': dWhh, 'Why': dWhy, 'bh': dbh, 'by': dby}
    
    def update_weights(self, grads):
        """
        Update weights using gradient descent.
        
        Args:
            grads: Dictionary of gradients
        """
        
        # TODO: Update each parameter
        # param = param - learning_rate * grad
        # (plain SGD passes train_rnn() with learning_rate=0.1; the solution uses Adagrad, like
        # min-char-rnn: keep a running sum of squared gradients and divide by its square root)
        
        pass
    
    def sample(self, h, seed_idx, n):
        """
        Generate text by sampling from the model.
        
        Args:
            h: Initial hidden state
            seed_idx: Starting character index
            n: Number of characters to generate
            
        Returns:
            indices: List of generated character indices
        """
        
        x = np.zeros((self.vocab_size, 1))
        x[seed_idx] = 1
        indices = []
        
        # TODO: Implement sampling
        # For each step:
        #   1. Compute h = tanh(Wxh @ x + Whh @ h + bh)
        #   2. Compute y = Why @ h + by
        #   3. Compute p = softmax(y)
        #   4. Sample next character from p
        #   5. Use sampled character as input for next step
        
        for t in range(n):
            # TODO: Your code here
            pass
        
        return indices


def train_rnn(num_iterations=300, seed=0):
    """
    Test harness: train your RNN on "abcabc..." and check that it learns the pattern.

    Returns (mean loss per character over the first 20 updates, over the last 20 updates).
    """
    # [PATCH 2026-10-18] the harness was all TODOs (it trained nothing and checked nothing);
    # it now runs your forward/backward/update_weights/sample and checks the loss
    np.random.seed(seed)

    # Simple training data
    data = "abc" * 40
    chars = sorted(set(data))          # sorted, so the indices are the same on every run
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    data_indices = [char_to_idx[ch] for ch in data]

    seq_length, hidden_size = 3, 10
    rnn = CharRNN(vocab_size=vocab_size, hidden_size=hidden_size, seq_length=seq_length,
                  learning_rate=0.1)

    h = np.zeros((hidden_size, 1))
    p = 0
    losses = []
    for it in range(num_iterations):
        if p + seq_length + 1 >= len(data_indices):    # wrap around and reset the memory
            p = 0
            h = np.zeros((hidden_size, 1))
        inputs = data_indices[p:p + seq_length]
        targets = data_indices[p + 1:p + seq_length + 1]   # the next character at each position

        loss, h, cache = rnn.forward(inputs, targets, h)
        grads = rnn.backward(inputs, targets, cache)
        rnn.update_weights(grads)

        losses.append(loss / seq_length)
        p += seq_length
        if it % 100 == 0:
            print(f"Iteration {it}: loss {losses[-1]:.4f} per character")

    first, last = np.mean(losses[:20]), np.mean(losses[-20:])
    sample = rnn.sample(np.zeros((hidden_size, 1)), seed_idx=char_to_idx['a'], n=12)
    print(f"Mean loss per character: first 20 updates {first:.3f}, last 20 updates {last:.3f} "
          f"(uniform guess {np.log(vocab_size):.3f})")
    print(f"Generated after 'a': {''.join(idx_to_char[i] for i in sample)!r} (expect 'bcabca...')")
    assert last < 0.1, f"loss per character {last:.3f} did not fall below 0.1: check backward/update"
    print("Training complete! Test passed.")
    return first, last


if __name__ == "__main__":
    print("Exercise 1: Build a Character-Level RNN")
    print("=" * 50)
    print("\nYour task: Fill in the TODOs above.")
    print("\nSteps:")
    print("1. Initialize weight matrices")
    print("2. Implement forward pass")
    print("3. Implement backward pass")
    print("4. Implement weight updates")
    print("5. Implement sampling")
    print("\nGood luck!")
    print("\n" + "=" * 50)
    
    # Uncomment when you're ready to test:
    # train_rnn()
