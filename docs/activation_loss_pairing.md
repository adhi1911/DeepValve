### Regression

| Output Type              | Activation  | Loss                  | When / Insight                              |
|--------------------------|-------------|-----------------------|---------------------------------------------|
| Unbounded real           | Linear      | MSE                   | Standard least-squares regression           |
| Unbounded real           | Linear      | MAE                   | Robust to outliers                          |
| Unbounded real           | Linear      | Huber                 | Smooth MAE–MSE tradeoff                     |
| Positive real            | ReLU        | MSE / MAE             | Enforce non-negativity                      |
| Positive real            | Softplus    | MSE / MAE             | Smooth ReLU alternative                     |
| Probabilistic (Gaussian) | Linear      | NLL (Gaussian)        | Predict mean (and variance if modeled)      |
| Count data               | Exponential | Poisson NLL           | Poisson regression                          |
| Rate / intensity         | Softplus    | Poisson / Gamma NLL   | Stable gradients                            |

---

### Binary Classification

| Label Space     | Activation             | Loss                        | When / Insight                  |
|------------------|------------------------|-----------------------------|----------------------------------|
| {0,1}           | Sigmoid               | Binary Cross-Entropy (BCE)  | Standard binary classification  |
| {−1,+1}         | Bipolar Sigmoid / Tanh | Squared loss / Hinge        | Older NN / SVM-style            |
| {0,1}           | Sigmoid               | Focal Loss                  | Severe class imbalance          |
| {0,1}           | Linear                | Hinge Loss                  | Linear SVM                      |
| {0,1}           | Sigmoid               | BCE + class weights         | Imbalanced datasets             |
| {0,1}           | Sigmoid               | KL Divergence               | Probabilistic targets           |
| Multi-label binary | Sigmoid (per class) | BCE (per class)             | Independent labels              |

 **Key Rule**: Binary → Sigmoid outputs independent probabilities

---

### Multiclass Classification (Single-Label)

| Class Relation         | Activation  | Loss                  | When / Insight                  |
|-------------------------|-------------|-----------------------|----------------------------------|
| Mutually exclusive      | Softmax     | Categorical Cross-Entropy | Standard multiclass            |
| Mutually exclusive      | Softmax     | Sparse CE             | Integer labels                  |
| Mutually exclusive      | Linear      | Multiclass Hinge      | SVM                             |
| Mutually exclusive      | Softmax     | Label-smoothed CE     | Regularization                  |
| Mutually exclusive      | Softmax     | Focal Loss            | Imbalanced multiclass           |
| Hierarchical classes    | Softmax (tree) | Hierarchical CE     | Taxonomy labels                 |

 **Key Rule**: Softmax enforces sum-to-1 constraint

---

### Multilabel Classification (Important Distinction)

| Task                     | Activation          | Loss                        | Insight                        |
|---------------------------|---------------------|-----------------------------|--------------------------------|
| Multi-label (independent) | Sigmoid (per class) | Binary Cross-Entropy        | Each label independent         |
| Multi-label (imbalanced)  | Sigmoid             | Focal Loss                  | Rare labels                    |
| Multi-label probabilistic | Sigmoid             | BCE + weights               | Class imbalance                |

 **Never use Softmax for multilabel**

---

### Special / Advanced Cases

| Scenario                | Activation          | Loss                  | Why                              |
|--------------------------|---------------------|-----------------------|----------------------------------|
| Ordinal regression      | Sigmoid chain       | Ordinal CE            | Ordered labels                  |
| Survival analysis       | Linear / Softplus   | Cox partial loss      | Time-to-event                   |
| Energy-based models     | Linear              | Contrastive loss      | Unnormalized outputs            |
| Metric learning         | Linear              | Triplet / Contrastive | Distance learning               |
| Ranking                 | Linear              | Pairwise ranking loss | IR systems                      |