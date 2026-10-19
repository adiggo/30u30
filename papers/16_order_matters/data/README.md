# Data Directory

This directory contains datasets and data generation utilities for Pointer Networks training.

---

## Available Datasets

### 1. Sorting Dataset

**Purpose:** Train models to sort numbers

**Generation:**
```python
from torch.utils.data import Dataset
import torch

class SortingDataset(Dataset):
    def __init__(self, num_samples=10000, set_size=10):
        self.num_samples = num_samples
        self.set_size = set_size
    
    def __getitem__(self, idx):
        values = torch.rand(self.set_size)
        _, sorted_indices = torch.sort(values)
        return values.unsqueeze(1), sorted_indices
```

**Usage:**
```bash
python train.py --task sort --set-size 10 --num-samples 10000
```

**Dataset stats:**
- **Default size:** 5,000 fresh training sets per epoch, 1,000 validation
- **Element range:** `train.SortingDataset` draws uniformly from [0, 10) (the paper used [0, 1]; the snippet above shows [0, 1))
- **Set sizes:** 5-50 elements (configurable)

---

### 2. Convex Hull Dataset

**Purpose:** Learn geometric reasoning - find boundary points

**Generation:**
```python
from scipy.spatial import ConvexHull
import torch

# Generate random 2D points
points = torch.rand(num_points, 2)

# Compute ground truth hull
hull_indices = ConvexHull(points.numpy()).vertices  # counter-clockwise, arbitrary start
```

**Targets** (`train.ConvexHullDataset`, 2026-10-18): the vertices rotated to start at the leftmost
point (one canonical order per hull), then `STOP` = `num_points` (an extra learned memory slot of
`ReadProcessWrite(stop_token=True)` that ends the output), then `IGNORE_INDEX` (-1) up to length
`num_points + 1`; the loss skips the IGNORE steps. The earlier labels padded with a repeated vertex,
which the decoder had already masked (infinite loss). Convex hull is a Pointer Networks task (Day 18).

**Usage:**
```bash
python train.py --task hull --set-size 10
```

**Dataset stats:**
- **Default size:** 5,000 fresh training sets per epoch, 1,000 validation
- **Point distribution:** Uniform in [0, 1]^2
- **Hull size:** Variable (mean 6.0 vertices for 10 points, 7.8 for 20; measured on 5,000 sets)

---

### 3. TSP Dataset

**Purpose:** Traveling Salesman Problem - find good tours

**Generation:**
```python
# Generate city coordinates
cities = torch.rand(num_cities, 2)

# Compute greedy nearest-neighbor tour (teacher signal)
def greedy_tour(cities):
    # start at the leftmost city: an order-invariant model cannot see which city is index 0
    current = int(torch.argmin(cities[:, 0]))
    unvisited = set(range(len(cities))) - {current}
    tour = [current]
    
    while unvisited:
        nearest = min(unvisited, 
                     key=lambda c: torch.norm(cities[current] - cities[c]))
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    return tour
```

**Usage:**
```bash
python train.py --task tsp --set-size 15 --num-samples 10000
```

**Dataset stats:**
- **Default size:** 5,000 fresh training sets per epoch, 500 validation
- **City distribution:** Uniform in [0, 1]^2
- **Tour quality:** Greedy heuristic (~1.25x optimal)

---

## Dynamic Data Generation

All datasets are generated **on-the-fly** during training:

**Advantages:**
- No storage required (generates as needed)  
- Infinite dataset - never repeats  
- Easy to adjust difficulty (change set size)  
- No download needed

**How it works:**
```python
# In the dataloader
for epoch in range(num_epochs):
    for batch in dataloader:
        # Generates new random instances each iteration
        inputs, targets = batch
```

---

## Pre-generated Datasets (Optional)

For reproducibility, you can generate fixed datasets:

### Generate and Save

```python
import torch
from train import SortingDataset

# Generate dataset (each item is drawn at random when it is accessed)
dataset = SortingDataset(num_samples=10000, set_size=10)

# Save to disk
torch.save({
    'data': [dataset[i] for i in range(len(dataset))],
    'config': {
        'task': 'sort',
        'set_size': 10,
        'num_samples': 10000
    }
}, 'sorting_dataset_10k.pt')
```

### Load Pre-generated

```python
import torch

# Load dataset
data = torch.load('sorting_dataset_10k.pt')
samples = data['data']
config = data['config']

print(f"Loaded {len(samples)} samples for {config['task']}")
```

---

## Dataset Difficulty Levels

### Easy (Good for Initial Training)
- **Sorting:** 5-10 elements
- **Convex Hull:** 10-15 points
- **TSP:** 5-10 cities

### Medium (Standard Training)
- **Sorting:** 15-30 elements
- **Convex Hull:** 20-30 points  
- **TSP:** 15-20 cities

### Hard (Generalization Test)
- **Sorting:** 50-100 elements
- **Convex Hull:** 50+ points
- **TSP:** 30-50 cities

---

## Dataset Statistics

Measured results (sorting, convex hull) are in `../README.md` under "Results in this folder".
The tables that used to be here (e.g. "100% sorting accuracy at 10 elements", "98% hulls at
10 points") were not measurements and have been removed (2026-10-18).

---

## Custom Datasets

Want to try your own set-to-sequence problem?

### Template

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """Your custom set-to-sequence problem."""
    
    def __init__(self, num_samples, set_size):
        self.num_samples = num_samples
        self.set_size = set_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Step 1: Generate input set (order doesn't matter)
        inputs = self.generate_input_set(self.set_size)
        
        # Step 2: Compute ground truth output sequence (order matters!)
        targets = self.compute_target_sequence(inputs)
        
        return inputs, targets
    
    def generate_input_set(self, size):
        # Your input generation logic
        raise NotImplementedError
    
    def compute_target_sequence(self, inputs):
        # Your target computation logic
        raise NotImplementedError
```

### Example: Sorting by Absolute Value

```python
class AbsValueSortDataset(Dataset):
    def __getitem__(self, idx):
        # Generate numbers (can be negative!)
        values = torch.randn(self.set_size)
        
        # Sort by absolute value
        abs_values = torch.abs(values)
        _, sorted_indices = torch.sort(abs_values)
        
        return values.unsqueeze(1), sorted_indices
```

---

## Data Augmentation

### For Better Generalization

```python
# Random scaling
values = values * random.uniform(0.5, 2.0)

# Random offset
values = values + random.uniform(-1.0, 1.0)

# Random subset (for variable-size training)
actual_size = random.randint(5, max_size)
values = values[:actual_size]
```

---

## Quick Start

```bash
# Train on sorting (easiest)
python train.py --task sort --set-size 10 --epochs 50

# Train on convex hull (medium)
python train.py --task hull --set-size 20 --epochs 100

# Train on TSP (hardest)
python train.py --task tsp --set-size 15 --epochs 200
```

---

## Resources

- **Training script:** `../train.py`
- **Dataset generators:** `../train.py` (SortingDataset, ConvexHullDataset, TSPDataset)
- **Paper:** https://arxiv.org/abs/1511.06391

---

**Tip:** Start with small sets (5-10 elements) and gradually increase.
