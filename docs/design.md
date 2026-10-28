# Design

## Private set intersection

`psi.py`:
- Each customer ID is hashed onto Ed25519 (a SHA-512 try-and-increment candidate multiplied by the cofactor 8, so it lies in the prime-order subgroup).
- Each party blinds its points with a fresh random scalar, and the bank compares the double-blinded values. A match means the same ID.
- The tests check that hashed points really are in the prime-order subgroup.

Both sides learn the size of the overlap, and the bank learns which of its rows are in it. In the three-party flow the bank runs two pairwise intersections, so it learns its overlap with each party separately. The protocol is only secure if every party follows it (semi-honest).

## Split network

`splitnn.py`:
- Each party's bottom network maps its 4 features through 16 hidden units (tanh) to an 8-dimensional embedding. The bank's top network has one hidden layer plus a linear skip.
- Training is plain mini-batch SGD in NumPy. Only embeddings and their gradients cross between parties.
- The UPI and telecom parties apply a defence before sending: a rank-1 bottleneck (in effect one number spread over 8 dimensions), L2 clipping to norm 0.95, a cross-covariance penalty between features and embedding, and Gaussian noise with σ = 0.45. The noise is a fixed setting, not calibrated to a differential-privacy budget.

## Inversion attack

`inversion.py` assumes the bank knows the raw features of the training customers (a generous assumption) and fits two decoders from embedding back to features: ridge regression on a quadratic basis and a 64-64 MLP. Leakage is the better decoder's R² on held-out customers, averaged over the UPI and telecom features. A low R² means these two decoders fail; it does not prove nothing leaks, and the gradients are not attacked.

## Results on synthetic data

`python -m splitbureau bench --ablation` (900 shared customers, 160 epochs, seed 42). Defaults in the synthetic data are a fixed function of the features plus a little noise, so the AUCs are far higher than a real bureau would see.

| Model | ROC-AUC | PR-AUC | Attacker R² |
|---|---|---|---|
| Bank's own 4 features | 0.703 | 0.617 | – |
| Split network, no defence | 0.976 | 0.969 | 0.954 |
| Split network, full defence | 0.867 | 0.841 | 0.134 |
| All 12 features pooled | 0.986 | 0.981 | – |

| Defence | ROC-AUC | Attacker R² |
|---|---|---|
| none | 0.976 | 0.954 |
| rank-1 bottleneck only | 0.932 | 0.233 |
| bottleneck + clipping + penalty | 0.933 | 0.229 |
| bottleneck + noise | 0.865 | 0.137 |
| full defence | 0.867 | 0.134 |

PSI found exactly the 900 shared customers among 1,080, 1,140 and 1,110 records in about 5 seconds. An undefended embedding is almost fully invertible; most of the protection comes from sending one number per party, while the noise lowers R² further at a real cost in accuracy.

## Results on real data

`python -m splitbureau realdata --data "default of credit card clients.xls"` uses UCI's Default of Credit Card Clients data (CC BY 4.0): 30,000 card holders at a Taiwanese bank in 2005, 22.1% of whom defaulted the following month. Results go to `results/real_data.json`.

The columns all come from one bank, so they are split three ways: the bank keeps the credit limit, education, marital status and age plus the label; a "bureau" holds six months of repayment status; a "payments" party holds six monthly bill and payment amounts. Sex is left out, and the rows are already aligned, so PSI is skipped. The split is 75/25 (7,500 test rows), and every model setting is the synthetic benchmark's, unchanged.

| Model | ROC-AUC | PR-AUC | Attacker R² |
|---|---|---|---|
| Bank's own 4 features | 0.623 | 0.300 | – |
| Split network, no defence | 0.769 | 0.543 | 0.947 |
| Split network, full defence | 0.735 | 0.491 | 0.111 |
| All 22 features pooled | 0.766 | 0.540 | – |

The other parties' data is what makes the model useful: the bank alone gets 0.62, and the split network matches pooling everything at 0.77. The defence costs 0.035 ROC-AUC here and keeps three quarters of what the other parties add. What leaks is what predicts: with the defence, overall R² falls to 0.11, but the latest repayment status is still recovered at R² 0.41, while bill amounts leak almost nothing.
