# Consultation Briefing — DNN Experiments (PyTorch)

Reading time: ~15 minutes — concise, conversation-ready briefing covering what we did, how we did it, the results we obtained, caveats, and suggested next steps.

---

**Executive summary**

- Goal: reimplement and extend the previous sklearn MLP experiments in PyTorch, evaluate stronger MLP variants (residual blocks, advanced optimizers/schedulers), and compare DNNs to the existing tree-based stacking baselines across three representative tasks (Heart classification, Temperature regression, Wind forecasting).
- Outcome: DNNs (ResidualMLP family) improved over the sklearn MLP baseline in most cases but did not consistently beat the best tree-based stacking ensembles. The gap is largest on the small clinical dataset (heart). For temperature and wind, DNNs closed much of the gap; wind forecasting reached parity with a simple ridge model.
- Key numbers (short): Heart best PyTorch AUC = 0.9517 vs stacking 0.9784; Temperature best PyTorch R² = 0.7520 vs stacking 0.7766; Wind best PyTorch R² = 0.9710 vs Ridge 0.9714.
- Artifacts: trained `PyTorchMLPWrapper` objects saved under `pytorch_results/models/` and integrated into the Streamlit dashboard (`prediction_dashboard.py`). Small FT‑Transformer and TabNet proof-of-concept runs were executed and saved to `pytorch_results/ft_tabnet_results.csv`.

---

1) Project context and objectives

We targeted three representative dashboard tasks to evaluate whether a deeper, PyTorch-based MLP workflow could outperform the existing classical ML (tree-based) stacking approach used in the repo. The core objectives were:

- Reimplement the MLP experiments in PyTorch to gain precise control over training recipes (optimizers, weight decay, schedulers, warmup, residual connections).
- Add a Residual MLP variant (ResNet-style MLP) and run controlled sweeps to find robust PyTorch models.
- Integrate the best DNN wrappers into the Streamlit dashboard for interactive comparison and inspection.
- Try two modern tabular deep models (FT‑Transformer and TabNet) as a quick next-step comparison.

2) Datasets and tasks (what we ran)

- Heart classification (small clinical dataset)
    - Rows: ~952 (the dataset used in the dashboard experiments).
    - Task: binary classification (disease vs no disease). Metric used for selection: AUC (ROC).
    - Note: small sample size — variance is higher and ensembles often benefit more than single DNNs in this regime.

- Temperature regression (large, continuous meteorological dataset)
    - Rows: ~62k.
    - Task: single-output regression (predict next-step temperature using available features).
    - Note: larger sample size; DNNs typically scale well here but feature engineering and scaling matter.

- Wind forecasting (sliding-window univariate forecasting)
    - Rows: ~50k (time-series derived samples using sliding windows).
    - Task: univariate forecasting with recursive multi-step predictions (we used a 24-step lookback window in the experiments saved to the repo).
    - Special: the MinMax scaler for wind was fit on a single 'power' column during training; this required careful per-element scaling at inference time in the dashboard.

For all three tasks we used the same data splits and preprocessing logic as the existing pipeline so comparisons are apples-to-apples (models were selected on the validation set and final metrics reported on test data used in the project's evaluation scripts).

3) Preprocessing pipeline highlights

- Imputation: the wrapper supports imputation (KNN / configured imputer) when needed; the temperature wrapper saved the imputer and label encoders so dashboard inference reuses identical preprocessing.
- Scaling:
    - Heart / Temperature: StandardScaler in most saved wrappers.
    - Wind: MinMaxScaler was used but it was fit on a single 'power' column — this required a special per-element scale/unscale helper in the dashboard to reconstruct the network input correctly for forecasting.
- Categorical variables: label encoders are saved in the wrapper and reapplied at inference time.

4) Models implemented and wrappers

- FlexMLP: a flexible feed-forward MLP implementation used as a baseline. Configurable depth, hidden sizes, dropout, and optional batch normalization.
- ResidualBlock + ResidualMLP: residual (skip) connections inside dense layers to improve gradient flow and allow deeper MLPs. The best-performing DNNs were `ResidualMLP` variants.
- `PyTorchMLPWrapper`: a thin sklearn-like wrapper that stores the trained `nn.Module` plus preprocessing objects (scaler, imputer, label encoders), metadata (feature list, task type), and prediction helpers (`predict`, `predict_proba`, `predict_regression`). These wrappers are pickled with `joblib.dump` and require `pytorch_model_utils.py` to unpickle in another session.

Quick conceptual snippets (for the meeting, you can paste these):

Residual block idea (conceptual):

```python
class ResidualBlock(nn.Module):
        def __init__(self, dim, dropout=0.0):
                super().__init__()
                self.fc1 = nn.Linear(dim, dim)
                self.bn1 = nn.BatchNorm1d(dim)
                self.act = nn.ReLU()
                self.drop = nn.Dropout(dropout)
                self.fc2 = nn.Linear(dim, dim)
                self.bn2 = nn.BatchNorm1d(dim)

        def forward(self, x):
                out = self.act(self.bn1(self.fc1(x)))
                out = self.drop(out)
                out = self.bn2(self.fc2(out))
                return self.act(out + x)
```

Wrapper usage example (load & predict):

```python
from joblib import load
wrapper = load('pytorch_results/models/heart_classification_pytorch_model.pkl')
X = pd.DataFrame([patient_row], columns=wrapper.feature_columns)
prob = wrapper.predict_proba(X)[0,1]
```

5) Training recipes & practical choices (what we actually tried)

We reimplemented the training loops to allow more flexible recipes than sklearn's MLP. The main levers we used:

- Optimizers: AdamW and SGD (with momentum / Nesterov where applicable).
- Regularization: explicit weight decay and dropout in the network architecture.
- LR schedulers: CosineAnnealingLR and ReduceLROnPlateau were tested to improve convergence.
- Early stopping: models were selected using validation metrics to avoid overfitting.
- Losses: standard losses per task (BCE for binary classification, MSE/Huber for regression).

Rather than listing every hyperparameter sweep, the experiment scripts in the repo contain the exact configurations; the notable architecture that performed consistently well was the `ResidualMLP` with a wide hidden projection (the saved wrapper uses a hidden projection of 512 with stacked residual blocks — noted as `ResidualMLP (512 × 4)` in the results).

6) Experiment orchestration (which scripts to inspect)

- Baseline DNN experiments (PyTorch): [pytorch_dnn_experiments.py](pytorch_dnn_experiments.py)
- Advanced sweeps (residuals, more optimizers/schedulers): [pytorch_advanced_experiments.py](pytorch_advanced_experiments.py)
- FT‑Transformer / TabNet quick runs: [ft_tabnet_experiments.py](ft_tabnet_experiments.py)
- Dashboard integration: [prediction_dashboard.py](prediction_dashboard.py)
- Model + utilities: [pytorch_model_utils.py](pytorch_model_utils.py)

7) Results — what we obtained (task-by-task)

- Heart classification (small dataset)
    - Best PyTorch: ResidualMLP (512 × 4) — AUC ≈ 0.9517 (saved wrapper: `heart_classification_pytorch_model.pkl`).
    - ML stacking baseline: AUC ≈ 0.9784.
    - Interpretation: DNNs improved over sklearn MLP baselines but did not match the stacking ensemble here. With under 1k samples, tree ensembles with careful features and stacking tend to generalize better than single dense nets unless heavy regularization/ensembling is used.

- Temperature regression (large dataset)
    - Best PyTorch: ResidualMLP (512 × 4) — R² ≈ 0.7520 (saved wrapper: `temperature_regression_pytorch_model.pkl`).
    - ML stacking baseline: R² ≈ 0.7766.
    - Interpretation: DNNs closed a sizable portion of the gap but the ensemble still edges out by a few percentage points. With more tuning (FT‑Transformer or TabNet at scale, or ensembling DNNs + trees) this gap can narrow further.

- Wind forecasting (sliding-window)
    - Best PyTorch: FlexMLP + Cosine LR scheduler — R² ≈ 0.9710 (saved wrapper: `wind_forecasting_pytorch_model.pkl`).
    - Benchmark: Ridge regression R² ≈ 0.9714.
    - Interpretation: near parity. The forecasting setup (window=24, recursive multi-step) meant shallow networks plus correct scaling were already very competitive.

All saved model wrappers are under `pytorch_results/models/`. The dashboard loads them for interactive demonstration.

8) FT‑Transformer and TabNet quick experiments (what we ran and results)

We ran short proof-of-concept FT‑Transformer and TabNet runs to see whether modern tabular deep models could beat the DNN family we trained.

- Heart (Dataset2):
    - FT‑Transformer (small config) — validation AUC ≈ 0.9092.
    - TabNet (small run) — validation AUC ≈ 0.9343 (early stopped at epoch 18; best val 0.93431).
    - Comment: TabNet beat the FT‑Transformer in the short run and came closer to the PyTorch ResidualMLP, but still fell short of the tree-based stacking leader.

- Temperature (Dataset1 sample):
    - FT‑Transformer — validation R² ≈ 0.4817 (small quick run).
    - TabNet — initial run failed for regression because the TabNet wrapper expected a 2D target; the fix is to reshape `y` to `(-1, 1)` for regression targets. After fixing that, TabNet should be re-run (we marked this as a follow-up).

All FT/TabNet results and logs were saved to `pytorch_results/ft_tabnet_results.csv` and the simple experiment script is `ft_tabnet_experiments.py`.

9) Dashboard integration and inference details

- File: [prediction_dashboard.py](prediction_dashboard.py)
    - The dashboard now loads `PyTorchMLPWrapper` objects (cached) and offers a DNN prediction button in each task section alongside the existing ML models.
    - We handled a few production issues discovered during integration:
        - Humidity scale mismatch for temperature: dashboard inputs are 0–100% while the DNN expects 0–1; we convert humidity to 0–1 for DNN inference.
        - Wind MinMax scaler: scaler was fit on a single 'power' column — inference requires per-element scaling and reconstruction; we added `dnn_wind_predict()` helper to do this safely.
        - Classification scalar shapes: some saved wrappers returned scalar `numpy.int64` for `predict` on single rows while `predict_proba` returned arrays — dashboard code now handles both shapes robustly.

10) Short reproducibility checklist (commands)

Activate the same environment used for development, install dependencies, then run experiments or the dashboard.

Windows PowerShell example:

```powershell
& .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
pip install pytorch-tabnet
# ensure torch is installed in the venv (we used torch 2.10.0+cpu during development)
# run an experiment (example)
python pytorch_advanced_experiments.py --task heart
# launch the dashboard
streamlit run prediction_dashboard.py
```

11) What to say in the consultation (short bullets / copy-ready)

- "We reimplemented MLPs in PyTorch to enable richer training recipes — optimizers like AdamW, explicit weight decay, schedulers, and residual connections."
- "Residual MLPs (wide projection + stacked residual blocks) gave the best DNN results we found."
- "Tree-based stacking still outperforms single DNNs on the small clinical task (heart). For temperature and wind the DNNs did much better and reached near parity for wind forecasting."
- "We’ve integrated the saved DNN wrappers into the Streamlit dashboard so you can run live comparisons and example predictions during the meeting."
- "Quick FT‑Transformer and TabNet runs show promise (TabNet performed well on heart), but they need more tuning and GPU time to be definitive."

12) Limitations, caveats, and why stacking still wins in some cases

- Small-sample regimes: with <1k rows (heart) DNNs tend to overfit or require heavy regularization and ensembling. Tree ensembles with engineered features and stacking can exploit structure more robustly.
- Feature engineering / categorical handling: some tree models implicitly handle interactions and missing values better than a plain MLP.
- Computational cost: DNN tuning (FT‑Transformer, larger TabNet) benefits from GPU time and larger hyperparameter sweeps; we ran short experiments on CPU for speed.

13) Concrete suggested next steps (pick 1–3 to discuss during consultation)

- Quick follow-up: re-run TabNet for temperature after reshaping targets to `(-1, 1)` (trivial fix) and add the result to the summary.
- Mid-term: benchmark FT‑Transformer and TabNet on wind forecasting (requires adapting sliding-window input) and run multi-seed experiments for stability.
- Longer-term: try stack-of-models mixing DNN predictions with tree-based features (stacked or blended ensemble) — often yields the best of both worlds.

14) Files & artifacts to point the professor to (open in the editor)

- Model utilities and wrappers: [pytorch_model_utils.py](pytorch_model_utils.py)
- Experiments drivers: [pytorch_dnn_experiments.py](pytorch_dnn_experiments.py), [pytorch_advanced_experiments.py](pytorch_advanced_experiments.py)
- FT/TabNet script: [ft_tabnet_experiments.py](ft_tabnet_experiments.py)
- Dashboard (DNN integrated): [prediction_dashboard.py](prediction_dashboard.py)
- Saved DNN wrappers: `pytorch_results/models/` (open the folder in the project explorer)
- FT/TabNet quick results: `pytorch_results/ft_tabnet_results.csv`

---

If you want, I can run the small TabNet re-run for temperature (reshape target → (-1,1)) now and append the updated row to `pytorch_results/ft_tabnet_results.csv`, or run a short FT/TabNet job on wind (will require changing the input shape to match sliding windows). Which follow-up should I prioritize?

*File updated automatically by the project assistant.*
