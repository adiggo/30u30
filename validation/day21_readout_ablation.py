"""Day 21: does the set2set readout (an attention-weighted MEAN of node states plus a bounded LSTM state) struggle
with the synthetic target, which is extensive (a sum over atoms and bonds)? Runs train_minimal.train_model with the
script's defaults (500 molecules, seed 42, hidden 64, T=3, 50 epochs, Adam 1e-3, L1 loss) for readout = set2set and sum.
Reference points: predict-the-mean MAE 2.98; least squares on [atom counts, #bonds, 1] MAE 0.089.
Usage: OMP_NUM_THREADS=1 python validation/day21_readout_ablation.py [day_dir]"""
import argparse, os, sys, tempfile
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "papers", "21_Neural_Message_Passing")
sys.path.insert(0, DAY)
import train_minimal as T
args = argparse.Namespace(hidden_dim=64, n_messages=3, message_type="edge_network", readout="set2set", set2set_steps=6,
                          epochs=50, lr=1e-3, batch_size=32, n_molecules=500, n_targets=3, compare_messages=False,
                          seed=42, output_dir=tempfile.gettempdir())  # train_model never writes there
for ro in ["set2set", "sum"]:
    _, losses, metrics = T.train_model(args, readout_type=ro, verbose=False)
    maes = [m["mae"] for m in metrics]
    print(f"readout={ro:8s} final test MAE {maes[-1]:.3f}  best {min(maes):.3f}  final train L1 {losses[-1]:.3f}")
