# Project Rules & Agent Directives: SleepyBrain Dynamic Connectome (GAT+GRU)

This file defines the mandatory coding, execution, checkpointing, and architectural rules for the **Stockholm SleepyBrain (`ds000201`) Dynamic Connectome Analysis & GAT+GRU Forecaster** project. All agents and pair-programming sessions operating on this workspace MUST adhere strictly to these rules without exception.

---

## 1. Mandatory Checkpointing & Persistence ("Zero Redundant Re-Training" Rule)

Every data transformation, baseline classifier, deep learning model, and cross-validation loop MUST implement persistent disk checkpointing. Running code in Google Colab or remote environments is susceptible to timeouts and disconnects; the user must never be forced to re-train a model from scratch across sessions.

### Concrete Implementation Protocol:
1. **Target Directory:**
   All persistent artifacts MUST be saved under `output_root`:
   ```python
   output_root = Path('/content/drive/MyDrive/sem 4 project (1)/preprocessed')
   ```
2. **Pre-Check Guard & Retrain Toggle:**
   Every training cell MUST define a boolean toggle (default `False`, NEVER `True`) and verify whether the checkpoint already exists on disk before doing any heavy computation:
   ```python
   checkpoint_path = output_root / '<model>_best.pt'
   history_path = output_root / '<model>_history.json'
   FORCE_RETRAIN_<MODEL> = False

   if checkpoint_path.exists() and not FORCE_RETRAIN_<MODEL>:
       print(f"[FOUND CHECKPOINT] Existing trained checkpoint at: {checkpoint_path}")
       print("Loading saved model weights (skipping re-training)...")
       # load state_dict and history
   else:
       # Run training loop, track best validation metric, save weights and history
   ```
3. **Immediate & Incremental Disk Checkpointing:**
   - Model weights and history MUST be saved to disk **immediately inside the epoch/fold loop** whenever the validation metric improves, NOT deferred until after all epochs/folds finish.
   - Long-running loops (Step 33 5-fold CV, Step 34 30-epoch training, Step 35 permutation test) MUST save progress after every single fold/epoch/permutation.
   - Interrupted runs MUST detect partial progress and resume from the last completed epoch or fold, rather than restarting from scratch.
4. **Artifact Standards Per Step:**
   - **Step 27 (Linear SVM):** Cache extracted feature vectors in `output_root / 'static_fc_features.npz'` to avoid re-reading 165 files from Drive. Save trained scaler and model via `joblib.dump(..., output_root / 'svm_baseline.joblib')`.
   - **Step 28 (Plain GCN):** Save best model state to `output_root / 'gcn_baseline_best.pt'` immediately upon improvement and metrics to `output_root / 'gcn_baseline_history.json'`.
   - **Step 30 (GAT+GRU Baseline):** `FORCE_RETRAIN_V1 = False`. Save best model state to `output_root / 'gat_gru_baseline_best.pt'` and metrics to `output_root / 'gat_gru_baseline_history.json'`.
   - **Step 31 (GAT+GRU V2 Regularized):** Save best model state to `output_root / 'gat_gru_v2_best.pt'` and metrics to `output_root / 'gat_gru_v2_history.json'`.
   - **Step 32 (PCA + RBF-SVM):** Verify `pca_svm_baseline.joblib` exists before fitting. Save scaler, PCA, and SVM to `output_root / 'pca_svm_baseline.joblib'`.
   - **Step 33 (5-Fold Cross-Validation):** Check if individual fold checkpoints `gat_gru_v2_fold_{fold+1}_best.pt` exist and skip completed folds. Checkpoint aggregate metrics incrementally after each fold to `output_root / 'cv_results.json'`.
   - **Step 34 (GAT+GRU V3 Stabilized Accumulation):** Save best model state to `output_root / 'gat_gru_v3_best.pt'` immediately inside the loop, history to `output_root / 'gat_gru_v3_history.json'`, and epoch resume state to `output_root / 'gat_gru_v3_resume.pt'` after every epoch.
   - **Step 35 (Permutation Test Null Baseline):** Reuse Step 33's real score from `cv_results.json` (saving 75 epochs). Progress MUST be checkpointed incrementally after each permutation to `output_root / 'permutation_test_results.json'`.

---

## 2. Publication-Grade Visualizations for Critical Stages

Whenever an analytical, diagnostic, or benchmarking step is reached, generate a publication-quality Matplotlib figure and save it at 300 DPI to `output_root`:

- **Figure Standards:**
  - Save using `plt.savefig(save_path, dpi=300, bbox_inches='tight')`.
  - Always render inline with `plt.show()`.
  - Use high-contrast, accessible palettes (`#2b5c8f`, `#de2d26`, `#2ca02c`, `#e6550d`, `#6baed6`).
  - Annotate exact numerical values directly on bar charts, boxplots, and milestone markers.
  - Never use deprecated syntax (e.g. use `tick_labels` instead of `labels` for `plt.boxplot`).
  - Implement fallback handling so plotting cells run successfully even if previous variables were cleared from memory.

- **Mandatory Figure Registry:**
  - `qc_motion_distribution.png` (Step 16c: Mean & Max FD across cohort).
  - `behavioral_kss_manipulation_check.png` (Step 21b: Karolinska Sleepiness Scale paired crossover distribution).
  - `dynamic_fc_window_progression.png` (Step 22b: Sliding-window dFC matrix evolution).
  - `graph_sparsification_topology.png` (Step 23b: Sparsified adjacency matrix & node degree distribution).
  - `model_performance_benchmarks.png` (Step 30b: Cross-model AUROC comparison & learning curves).
  - `cv_5fold_generalization_benchmark.png` (Step 33b: 5-Fold LSO CV performance & model evolution).
  - `stabilized_training_dynamics.png` (Step 34b: Loss progression vs. $\ln(2)$ chance entropy floor and AUROC progression).
  - `permutation_test_null_distribution.png` (Step 35b: Empirical null distribution histogram, 95th percentile threshold, and real model significance).

---

## 3. Strict Leave-Subject-Out (LSO) Partitioning

Data leakage between training and validation partitions will cause catastrophic over-optimistic bias:
- **No Random Session Splits:** Because subjects underwent a 2-arm crossover design (both rested and sleep-deprived sessions exist per person), random session-level splits allow the network to memorize individual subject anatomies and resting states.
- **Partition by Subject ID:** All splits (`train_test_split`, `KFold`) MUST operate on unique subject IDs (`df['subject'].unique()`). A subject's sessions must reside exclusively in either the training set or the validation set.

---

## 4. Objective Ground Truth Integrity (`Sl_cond`)

- **The Gold Standard:** Always derive the binary sleep deprivation label (`sleep_deprived`) from `Sl_cond` in `participants.tsv`:
  - `Sl_cond == 1`: Sleep-deprived before Session 1, Rested before Session 2.
  - `Sl_cond == 2`: Rested before Session 1, Sleep-deprived before Session 2.
- **Do NOT Use Blinding Variables:** Never use `SleepDeprivedSession1` or `SleepDeprivedSession2` as ground truth labels. These represent the experimenter's subjective guessing under blinded conditions and exhibit ~18% error compared to the objective protocol.

---

## 5. Tensor Shape & PyTorch Discipline

- **Shape Alignment:** Always reshape logits and targets explicitly before passing to loss functions to prevent PyTorch dimension broadcasting bugs:
  ```python
  loss = loss_fn(logit.view(-1), label.view(-1))
  prob = torch.sigmoid(logit.view(-1)).item()
  ```
- **Graph Sequence Loading:** In PyTorch 2.4+, loading serialized PyTorch Geometric `Data` objects requires `weights_only=False`:
  ```python
  graphs = torch.load(path, weights_only=False)
  ```
- **Graph Convolution Stability:** For GCN degree matrix normalization, avoid zero or negative edge weights by taking absolute Pearson correlation values with numerical stability:
  ```python
  edge_weight = torch.abs(graph.edge_attr.squeeze(-1)) + 1e-5
  ```
- **Overfitting Mitigation (Node Identity Embeddings):** Do NOT feed raw 400-dimensional FC rows as static node features. This acts as an exact fingerprint of the graph, causing the GAT to memorize sessions (train AUROC 0.99, val AUROC 0.54). Instead, use learnable ROI embeddings (`nn.Embedding(400, 32)`), edge attributes for attention, and dropout ($p=0.3$).

---

## 6. Notebook and Code Integrity

- Preserve all markdown explanations, step numbering, and mathematical formulas throughout [`sleepybrain_dynamic_connectome_roadmap.ipynb`](file:///d:/connectome%20gat/sleepybrain_dynamic_connectome_roadmap.ipynb).
- Keep cells self-contained with explicit imports and device declarations (`device = 'cuda' if torch.cuda.is_available() else 'cpu'`).
- Ensure all Python files and scripts handle path operations using `pathlib.Path`.

