"""
Minimal Training Loop: the paper's regularized LSTM language model on Penn Treebank
==================================================================================

[PATCH 2026-10-18] Rewritten. The old loop ran a forward-only model on 100 random integer
sequences, so nothing could be learned (the loss sat at chance, 19 ln 26 = 61.9). This script
trains LSTMLanguageModel from implementation.py -- word embeddings -> 2 LSTM layers -> softmax,
dropout on the non-recurrent connections only (Zaremba et al. 2014, Sec. 3.2) -- with truncated
backpropagation through time, gradient-norm clipping and plain SGD on the real Penn Treebank,
and reports perplexity, the paper's metric (lower is better; exp of the per-word cross-entropy).

What the paper trained (Sec. 4.1; full PTB: 929k training words, 10k-word vocabulary, batch 20):

    model              layers x units  dropout  init (uniform)  unroll  epochs  learning rate           clip  test ppl
    non-regularized    2 x 200         0%       +-0.1           20      13      1; /2 each epoch after 4   5*   114.5
    medium             2 x 650         50%      +-0.05          35      39      1; /1.2 after 6            5     82.7
    large              2 x 1500        65%      +-0.04          35      55      1; /1.15 after 14         10     78.4
    (* clip value from the paper's code, github.com/wojzaremba/lstm; the text gives none for this model)

What we run (pure NumPy on one CPU core, so much smaller; same method, same 10k-word softmax):

    default (python train_minimal.py, about 3 min of CPU):
        2 x 200 LSTM, 50% dropout (keep_prob 0.5), init +-0.1, unroll 35, batch 20, clip 5,
        lr 1 for 4 epochs then /2 per epoch, 6 epochs on the first 100k training words
        (about 1/9 of PTB); validation and test perplexity on the full validation/test sets.
    --compare (~10 min of CPU): the same 2 x 200 model with and without dropout, everything else
        identical, on the same 100k training words for 10 epochs (lr 1 for 6 epochs, then /2 per
        epoch). On this small slice a 4.65M-parameter model is "large" relative to its data, the
        regime the paper is about (Sec. 1: "large RNNs tend to overfit"). Without dropout it
        overfits: its training perplexity falls far below the validation perplexity, which starts
        rising. With dropout the gap stays small and the best validation and test perplexities are
        better. Recorded numbers are in the README.
    --paper-config {small,medium,large}: the paper's exact settings on all of PTB. Correct but slow
        in NumPy on one CPU core: roughly 3 min per epoch for small (13 epochs); medium and large cost
        several and tens of times more per epoch over 39 and 55 epochs (hours to days).

Data: Mikolov's PTB split as shipped in the paper's repo (ptb.{train,valid,test}.txt). Like the
paper's data.lua, every line end becomes an <eos> token. Files are read from --data-dir, else
$PTB_DIR, else ./data/ptb, and downloaded from raw.githubusercontent.com if missing.

Run: python train_minimal.py --help
"""

import argparse
import os
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import our implementation
from implementation import LSTMLanguageModel, EarlyStoppingMonitor, clip_grad_norm

HERE = os.path.dirname(os.path.abspath(__file__))
PTB_URL = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.{}.txt'

# The paper's settings (Sec. 4.1) -- used by --paper-config and printed for comparison.
PAPER_CONFIGS = {
    'small': dict(hidden=200, keep_prob=1.0, init_scale=0.1, bptt=20, epochs=13, decay_after=4,
                  lr_decay=2.0, max_grad_norm=5.0, paper_valid=120.7, paper_test=114.5),
    'medium': dict(hidden=650, keep_prob=0.5, init_scale=0.05, bptt=35, epochs=39, decay_after=6,
                   lr_decay=1.2, max_grad_norm=5.0, paper_valid=86.2, paper_test=82.7),
    'large': dict(hidden=1500, keep_prob=0.35, init_scale=0.04, bptt=35, epochs=55, decay_after=14,
                  lr_decay=1.15, max_grad_norm=10.0, paper_valid=82.2, paper_test=78.4),
}

# Our CPU-sized settings. Only the size of the data slice and the number of epochs differ between
# the default run and the comparison; everything else follows the paper's recipe.
CPU_DEFAULT = dict(hidden=200, layers=2, keep_prob=0.5, init_scale=0.1, bptt=35, batch_size=20,
                   lr=1.0, decay_after=4, lr_decay=2.0, max_grad_norm=5.0, epochs=6,
                   train_tokens=100_000, valid_tokens=0, train_eval_tokens=20_000)
CPU_COMPARE = dict(CPU_DEFAULT, epochs=10, decay_after=6)


# ============================================================================
# DATA
# ============================================================================

def default_data_dir() -> str:
    """--data-dir default: $PTB_DIR if set, else data/ptb next to this file."""
    return os.environ.get('PTB_DIR') or os.path.join(HERE, 'data', 'ptb')


def ensure_ptb(data_dir: str) -> str:
    """Download any missing ptb.{train,valid,test}.txt from the paper's repository."""
    os.makedirs(data_dir, exist_ok=True)
    for split in ('train', 'valid', 'test'):
        path = os.path.join(data_dir, 'ptb.%s.txt' % split)
        if not os.path.exists(path):
            url = PTB_URL.format(split)
            print('Downloading %s -> %s' % (url, path))
            urllib.request.urlretrieve(url, path + '.part')
            os.replace(path + '.part', path)          # rename only once complete
    return data_dir


def read_words(path: str) -> List[str]:
    """Words of one PTB file, with <eos> at every line end (as the paper's data.lua does)."""
    with open(path, encoding='utf-8') as f:
        return f.read().replace('\n', ' <eos> ').split()


def load_ptb(data_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Return (train_ids, valid_ids, test_ids, vocab) as 1-D int arrays of word ids.

    The vocabulary is built from the training file in order of first appearance (data.lua does
    the same); Mikolov's split is closed, so valid/test contain no new words (rare words are
    already <unk>). It has 10,000 entries including <eos> and <unk>.
    """
    data_dir = ensure_ptb(data_dir or default_data_dir())
    words = {s: read_words(os.path.join(data_dir, 'ptb.%s.txt' % s)) for s in ('train', 'valid', 'test')}
    vocab: Dict[str, int] = {}
    for w in words['train']:
        if w not in vocab:
            vocab[w] = len(vocab)
    # np.array(list of ints, dtype=np.int64) -> one flat id array per split
    ids = [np.array([vocab[w] for w in words[s]], dtype=np.int64) for s in ('train', 'valid', 'test')]
    return ids[0], ids[1], ids[2], vocab


def batchify(ids: np.ndarray, batch_size: int) -> np.ndarray:
    """Cut a token stream into batch_size contiguous streams, one per column -> (rows, batch_size).

    Column b holds ids[b*rows:(b+1)*rows] (the tail that does not fit is dropped), like the
    paper's data.lua `replicate`. Successive windows of rows then continue each stream, which is
    what lets the hidden state be carried from one minibatch to the next (Sec. 4.1).
    """
    rows = len(ids) // batch_size
    # reshape(B, rows): stream b is row b; .T makes it column b; ascontiguousarray makes slicing a
    # window data[i:i+T] a cheap contiguous block
    return np.ascontiguousarray(ids[:rows * batch_size].reshape(batch_size, rows).T)


def windows(data: np.ndarray, bptt: int):
    """Yield (x, y) windows of up to bptt time steps; y is x shifted by one word."""
    for i in range(0, data.shape[0] - 1, bptt):
        T = min(bptt, data.shape[0] - 1 - i)          # the last window may be shorter
        yield data[i:i + T], data[i + 1:i + 1 + T]   # (T, B) each


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_epoch(model: LSTMLanguageModel, data: np.ndarray, bptt: int, lr: float,
                max_grad_norm: float, weight_decay: float = 0.0) -> Tuple[float, float]:
    """One pass of truncated BPTT over `data` (rows, B). Returns (perplexity, mean grad norm).

    The perplexity is measured during training, i.e. WITH dropout on (what the paper's code
    prints as "train perp."); see evaluate() for the dropout-free number.
    """
    state = model.init_state(data.shape[1])           # zero state at the start of each pass
    nll_sum, n_words, norms = 0.0, 0, []
    for x, y in windows(data, bptt):
        loss, state, cache = model.forward(x, y, state, training=True)
        # Truncated BPTT: backward() also returns dL/d(carried-in state); dropping it is exactly
        # what "truncated" means -- no gradient flows into the previous window.
        grads, _ = model.backward(cache)
        if weight_decay > 0:                           # our addition (not in the paper): L2 on weights
            for k, g in grads.items():
                if not k.startswith('b'):              # weights and embeddings only, no biases
                    g += weight_decay * model.params[k]
        norms.append(clip_grad_norm(grads, max_grad_norm))   # Sec. 4.1: clip the global norm
        model.sgd_update(grads, lr)
        nll_sum += loss * x.shape[1]                  # loss is (1/B) * sum -> times B = summed NLL
        n_words += x.size
    return float(np.exp(nll_sum / n_words)), float(np.mean(norms))


def evaluate(model: LSTMLanguageModel, ids: np.ndarray, batch_size: int, bptt: int) -> float:
    """Perplexity of a token stream with dropout off. batch_size 1 = one continuous stream (the
    paper's test evaluation); batch_size 20 = 20 parallel streams (its validation)."""
    data = batchify(ids, batch_size)
    state = model.init_state(batch_size)
    nll_sum, n_words = 0.0, 0
    for x, y in windows(data, bptt):
        loss, state, _ = model.forward(x, y, state, training=False)
        nll_sum += loss * x.shape[1]
        n_words += x.size
    return float(np.exp(nll_sum / n_words))


def learning_rate(epoch: int, lr: float, decay_after: int, lr_decay: float) -> float:
    """Sec. 4.1 schedule: lr for the first `decay_after` epochs, then divided by lr_decay each epoch
    (epochs counted from 1). E.g. medium: 1.0 for epochs 1-6, then 1/1.2, 1/1.2^2, ..."""
    return lr / lr_decay ** max(0, epoch - decay_after)


def train_lm(cfg: Dict, train_ids: np.ndarray, valid_ids: np.ndarray, test_ids: Optional[np.ndarray] = None,
             vocab_size: int = 10000, seed: int = 0, weight_decay: float = 0.0, patience: int = 0,
             verbose: bool = True, label: str = '') -> Dict:
    """Train one LSTMLanguageModel with config `cfg` (keys as in CPU_DEFAULT) and return its history.

    Tracks the best validation perplexity and keeps those parameters; at the end the best model is
    restored and evaluated on the test stream (if given). patience > 0 adds early stopping (our
    addition; the paper trains for a fixed number of epochs).
    """
    model = LSTMLanguageModel(vocab_size, cfg['hidden'], num_layers=cfg['layers'],
                              keep_prob=cfg['keep_prob'], init_scale=cfg['init_scale'], seed=seed)
    n_train = cfg['train_tokens'] or len(train_ids)
    n_valid = cfg['valid_tokens'] or len(valid_ids)
    train_data = batchify(train_ids[:n_train], cfg['batch_size'])
    train_eval_ids = train_ids[:min(n_train, cfg['train_eval_tokens'] or n_train)]
    valid_part = valid_ids[:n_valid]
    monitor = EarlyStoppingMonitor(patience=patience if patience > 0 else 10 ** 9, verbose=False)
    hist = {'label': label, 'config': dict(cfg), 'params': model.parameter_count(), 'epoch': [], 'lr': [],
            'train_ppl': [], 'train_eval_ppl': [], 'valid_ppl': [], 'grad_norm': [], 'seconds': []}
    if verbose:
        print('%s%d-layer LSTM, %d units, keep_prob %.2f, %s parameters; %s training words, %s validation words'
              % (label + ': ' if label else '', cfg['layers'], cfg['hidden'], cfg['keep_prob'],
                 format(model.parameter_count(), ','), format(n_train, ','), format(len(valid_part), ',')))
        print('  untrained: validation perplexity %.0f (a uniform guess over %d words is %d)'
              % (evaluate(model, valid_part, cfg['batch_size'], cfg['bptt']), vocab_size, vocab_size))
        print('  %5s %8s %12s %12s %10s %9s %6s' % ('epoch', 'lr', 'train ppl', 'train ppl', 'valid ppl',
                                                  '|grad|', 'sec'))
        print('  %5s %8s %12s %12s %10s %9s %6s' % ('', '', '(dropout on)', '(eval mode)', '', '(mean)', ''))
    t_start, cpu_start = time.time(), time.process_time()
    for epoch in range(1, cfg['epochs'] + 1):
        t0 = time.time()
        lr = learning_rate(epoch, cfg['lr'], cfg['decay_after'], cfg['lr_decay'])
        train_ppl, gnorm = train_epoch(model, train_data, cfg['bptt'], lr, cfg['max_grad_norm'], weight_decay)
        train_eval_ppl = evaluate(model, train_eval_ids, cfg['batch_size'], cfg['bptt'])
        valid_ppl = evaluate(model, valid_part, cfg['batch_size'], cfg['bptt'])
        for k, v in zip(('epoch', 'lr', 'train_ppl', 'train_eval_ppl', 'valid_ppl', 'grad_norm', 'seconds'),
                        (epoch, lr, train_ppl, train_eval_ppl, valid_ppl, gnorm, time.time() - t0)):
            hist[k].append(v)
        improved = valid_ppl < monitor.best_loss
        keep_going = monitor.check(valid_ppl, epoch, model.get_state() if improved else None)
        if verbose:
            print('  %5d %8.4f %12.1f %12.1f %10.1f %9.2f %6.0f%s'
                  % (epoch, lr, train_ppl, train_eval_ppl, valid_ppl, gnorm, time.time() - t0,
                     '  *best' if improved else ''))
        if not keep_going:
            if verbose:
                print('  early stopping: no improvement for %d epochs' % patience)
            break
    hist['best_valid_ppl'] = monitor.best_loss
    hist['best_epoch'] = monitor.best_epoch
    model.set_state(monitor.best_model_state)          # keep the best-validation parameters
    if test_ids is not None:
        hist['test_ppl'] = evaluate(model, test_ids, 1, cfg['bptt'])
    hist['total_seconds'] = time.time() - t_start
    hist['cpu_seconds'] = time.process_time() - cpu_start
    hist['model'] = model
    if verbose:
        print('  best validation perplexity %.1f at epoch %d%s; %.0f s (%.0f s CPU)'
              % (hist['best_valid_ppl'], hist['best_epoch'],
                 ('; test perplexity of that model %.1f' % hist['test_ppl']) if test_ids is not None else '',
                 hist['total_seconds'], hist['cpu_seconds']))
    return hist


# ============================================================================
# COMMAND LINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train the paper\'s regularized LSTM language model on Penn Treebank (NumPy). '
                    'Hyperparameters left unset take the preset: the default run, --compare, or --paper-config.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--compare', action='store_true',
                      help='dropout vs no dropout, same everything else (about 10 min of CPU)')
    mode.add_argument('--paper-config', choices=sorted(PAPER_CONFIGS),
                      help='the paper\'s exact settings on all of PTB (slow in NumPy: hours)')
    parser.add_argument('--data-dir', default=None, help='folder with ptb.{train,valid,test}.txt '
                        '(default: $PTB_DIR, else ./data/ptb; downloaded if missing)')
    # (flag, cfg key, type, help); every default is None = "use the preset's value"
    knobs = [('--hidden', 'hidden', int, 'LSTM units per layer (preset 200)'),
             ('--layers', 'layers', int, 'number of LSTM layers (preset 2)'),
             ('--keep-prob', 'keep_prob', float, 'dropout keep probability on the non-recurrent connections; '
              'the paper\'s "50%% dropout" is 0.5, "65%%" is 0.35, 1.0 = none (preset 0.5; with --compare '
              'this is the dropout model, the other one uses 1.0)'),
             ('--init-scale', 'init_scale', float, 'parameters start uniform in [-s, s] (preset 0.1)'),
             ('--epochs', 'epochs', int, 'epochs (preset 6; --compare 10)'),
             ('--lr', 'lr', float, 'SGD learning rate (preset 1.0)'),
             ('--decay-after', 'decay_after', int, 'epochs at the initial learning rate (preset 4; --compare 6)'),
             ('--lr-decay', 'lr_decay', float, 'divide the learning rate by this after each later epoch (preset 2)'),
             ('--max-grad-norm', 'max_grad_norm', float, 'clip the global gradient norm at this (preset 5)'),
             ('--bptt', 'bptt', int, 'unrolled time steps per window (preset 35)'),
             ('--batch-size', 'batch_size', int, 'parallel streams per minibatch (preset 20)'),
             ('--train-tokens', 'train_tokens', int, 'train on the first N training words; 0 = all 929,589 '
              '(preset 100000)'),
             ('--valid-tokens', 'valid_tokens', int, 'validate on the first N words; 0 = all 73,760 (preset 0)')]
    for flag, _, typ, hlp in knobs:
        parser.add_argument(flag, type=typ, default=None, help=hlp)
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='L2 weight decay on weights and embeddings (our addition; the paper uses none)')
    parser.add_argument('--patience', type=int, default=0,
                        help='early-stopping patience in epochs (our addition; 0 = off, as in the paper)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (initialization and dropout masks)')
    args = parser.parse_args()

    cfg = dict(CPU_COMPARE if args.compare else CPU_DEFAULT)
    if args.paper_config:
        p = PAPER_CONFIGS[args.paper_config]
        cfg.update({k: p[k] for k in ('hidden', 'keep_prob', 'init_scale', 'bptt', 'epochs', 'decay_after',
                                      'lr_decay', 'max_grad_norm')})
        cfg.update(layers=2, batch_size=20, lr=1.0, train_tokens=0, valid_tokens=0)
        print('Paper config "%s": the paper reports validation %.1f / test %.1f perplexity.'
              % (args.paper_config, p['paper_valid'], p['paper_test']))
        print('This is the full-size run; in NumPy on one CPU core it takes hours (see the docstring).')
    for _, key, _, _ in knobs:                       # explicit flags override the preset
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value

    print('\n' + '=' * 78)
    print('REGULARIZED LSTM LANGUAGE MODEL ON PENN TREEBANK (Zaremba et al. 2014)')
    print('=' * 78)
    train_ids, valid_ids, test_ids, vocab = load_ptb(args.data_dir)
    print('PTB: %s train / %s valid / %s test words, vocabulary %d (incl. <eos>, <unk>)'
          % (format(len(train_ids), ','), format(len(valid_ids), ','), format(len(test_ids), ','), len(vocab)))
    print('Recipe: dropout on non-recurrent connections only, truncated BPTT over %d steps with the state'
          % cfg['bptt'])
    print('carried across windows, batch %d, SGD lr %.1f for %d epochs then /%.3g per epoch, clip %.0f.'
          % (cfg['batch_size'], cfg['lr'], cfg['decay_after'], cfg['lr_decay'], cfg['max_grad_norm']))
    print('Paper (full PTB): non-regularized 2x200 114.5, medium 2x650 82.7, large 2x1500 78.4 test ppl.\n')

    runs = []
    if args.compare:
        for keep, label in ((1.0, 'NO dropout'), (cfg['keep_prob'], 'WITH dropout (keep %.2g)' % cfg['keep_prob'])):
            runs.append(train_lm(dict(cfg, keep_prob=keep), train_ids, valid_ids, test_ids, len(vocab),
                                 seed=args.seed, weight_decay=args.weight_decay, patience=args.patience,
                                 label=label))
            print()
        print('=' * 78)
        print('SUMMARY (same model, data, initialization and schedule; only dropout differs)')
        print('  %-22s %10s %9s %12s %10s %9s' % ('', 'best valid', '(epoch)', 'final valid', 'train ppl',
                                                 'test ppl'))
        for h in runs:
            print('  %-22s %10.1f %9d %12.1f %10.1f %9.1f'
                  % (h['label'], h['best_valid_ppl'], h['best_epoch'], h['valid_ppl'][-1],
                     h['train_eval_ppl'][-1], h['test_ppl']))
        print('  train ppl: eval mode (no dropout), final epoch, first %s training words.'
              % format(cfg['train_eval_tokens'], ','))
        print('  test ppl: the best-validation parameters on the full test set, one continuous stream.')
    else:
        runs.append(train_lm(cfg, train_ids, valid_ids, test_ids, len(vocab), seed=args.seed,
                             weight_decay=args.weight_decay, patience=args.patience))
    return runs


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(0)
