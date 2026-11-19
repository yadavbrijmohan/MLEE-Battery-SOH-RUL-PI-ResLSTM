# Physics-Informed Attention Networks for Li-Ion Battery State-of-Health (SOH) and RUL Estimation

**Course:** MLEE (Machine Learning for Electrical Engineering) — Department of Electrical and Electronics Engineering, BITS Pilani, Pilani Campus  
**Team Members:** Brijmohan Yadav (`\[2023B3A80948P]`), 
**Instructor:** `\\\[Ritesh Vyas]` | **Semester:** First Semester, 2026–2027

\---

## Project Overview

This repository contains the code, figures, project report, and A3 poster for our MLEE course project on **Lithium-Ion Battery State-of-Health (SOH) and Remaining Useful Life (RUL) Estimation**.

Built on the quantile-cleaned NASA Ames PCoE `B05\\\_discharge\\\_soh.csv` dataset from [`standing-o/SoH\\\_estimation\\\_of\\\_Lithium-ion\\\_battery`](https://github.com/standing-o/SoH_estimation_of_Lithium-ion_battery) and the [NASA Ames Prognostics Data Repository](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets), all models are strictly aligned with the **BITS Pilani MLEE Course Handout**:

1. **Baseline — Linear Regression (MLEE Unit 1.2.1):** Leakage-free linear baseline trained on measurable BMS sensor signals (excluding raw `capacity`).
2. **Method A — Standard LSTM (MLEE Unit 4.3):** Sequence model capturing cycle-to-cycle temporal dependencies over a sliding window of $W = 10$ cycles.
3. **Proposed — Physics-Informed Residual LSTM with MLP Temporal Attention (`PI-ResLSTM + Attn`, MLEE Units 4.3.3, 2.1, \& 1.1):** Combines a 2-layer Residual LSTM (mitigating vanishing gradients, Unit 4.2.3), a linear trend shortcut, an MLP temporal attention gate (Unit 2.1), Multitask Learning heads (Unit 1.1), input/target normalization \& denormalization (Unit 2.1.2), and a thermodynamic monotonicity loss penalty ([Wang et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11109204/)).

\---

## Repository Structure

```text
├── README.md                              # Project documentation \\\& setup guide
├── requirements.txt                       # Python dependencies
├── pi\\\_reslstm\\\_b05\\\_soh\\\_rul.py              # Standalone Python script (reproduces all metrics \\\& plots)
├── MLEE\\\_PI\\\_ResLSTM\\\_B05\\\_Colab.ipynb        # Google Colab notebook (click Run All)
├── figures/
│   ├── fig\\\_method\\\_b05.png                 # System pipeline \\\& architecture block diagram
│   ├── fig\\\_result1\\\_b05.png                # SOH \\\& RUL test trajectories on NASA B0005
│   └── fig\\\_result2\\\_b05.png                # Quantitative SOH RMSE, MAE, and R² comparison
├── report/
│   └── MLEE\\\_Project\\\_Report\\\_B05\\\_Final.docx # Full MLEE course project report
└── poster/
    └── MLEE\\\_Poster\\\_B05\\\_Final.pdf          # Print-ready A3 project poster
```

\---

## Dataset \& Leakage-Free Preprocessing

* **Source:** NASA Ames PCoE 18650 $\\text{LiCoO}\_2$ Battery `B0005` (`B05\\\_discharge\\\_soh.csv`, 168 discharge cycles at 2.0 A, 24 °C, 70% End-of-Life at **cycle 125**).
* **Automatic Retrieval:** You do **not** need to download large raw `.mat` files manually. Running the Colab notebook or Python script automatically clones `standing-o/SoH\\\_estimation\\\_of\\\_Lithium-ion\\\_battery` and loads `B05\\\_discharge\\\_soh.csv`.
* **Leakage Prevention:** Raw `capacity` is explicitly removed from the inputs so models predict SOH and RUL strictly from 6 measurable BMS signals (`terminal\\\_voltage`, `terminal\\\_current`, `temperature`, `charge\\\_current`, `charge\\\_voltage`, `time`) plus `cycle`.
* **Split \& Scaling (Unit 2.1.2):** 50% Train (cycles 1–84, `74` sequences) / 50% Test (cycles 85–168, `84` sequences) with `StandardScaler` on inputs and `MinMaxScaler` on targets.

\---

## Quantitative Results (NASA B0005 | 70% EOL = Cycle 125 | 50/50 Split)

|Method (MLEE Syllabus Unit)|SOH RMSE (%)|SOH MAE (%)|RUL RMSE (cycles)|SOH $R^2$|Phys. Viol. (%)|
|-|:-:|:-:|:-:|:-:|:-:|
|**Baseline: Linear Reg (Unit 1.2.1)**|**1.51**|**1.25**|**0.00**|**0.8578**|**0.00**|
|**Method A: Standard LSTM (Unit 4.3)**|5.78|4.94|25.97|-1.0722|76.19|
|**Proposed: PI-ResLSTM + Attn (Units 4.3.3 \& 2.1)**|**2.88**|**2.35**|30.81|**0.4870**|76.19|

### Key Takeaways

* **50.2% Error Reduction Over Standard LSTM:** `PI-ResLSTM + Attn` cuts SOH RMSE from `5.78%` to `2.88%` and lifts $R^2$ from `-1.0722` to `+0.4870`, tracking rest-induced capacity regeneration spikes (cycles 90, 104, 120, 151) where Standard LSTM saturates near `\\\~74.5%` SOH.
* **Honest Failure Analysis:** On a single-cell 50% split (`74` training sequences), Linear Regression achieves `1.51%` SOH RMSE because `B0005` ages nearly linearly under constant-current discharge, whereas deep recurrent models face small-sample out-of-distribution extrapolation on late-life RUL. Multi-cell full-lifecycle training (`B0005`, `B0006`, `B0007` $\\rightarrow$ `B0018`) is the primary next step.

\---

## How to Run

### Option 1: Google Colab (Recommended)

1. Upload `MLEE\\\_PI\\\_ResLSTM\\\_B05\\\_Colab.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Click **Runtime → Run all**. The notebook automatically clones the dataset, trains all three models, prints the results table, and saves `fig\\\_result1.png`.

### Option 2: Local Python Environment

```bash
pip install -r requirements.txt
git clone https://github.com/standing-o/SoH\\\_estimation\\\_of\\\_Lithium-ion\\\_battery.git
python pi\\\_reslstm\\\_b05\\\_soh\\\_rul.py
```

\---

## References

1. F. Wang *et al.*, "Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis," *Nature Communications*, vol. 15, no. 1, art. 4332, 2024.
2. K. A. Severson *et al.*, "Data-driven prediction of battery cycle life before capacity degradation," *Nature Energy*, vol. 4, pp. 383–391, 2019.
3. A. Vaswani *et al.*, "Attention is all you need," in *Proc. NeurIPS*, 2017, pp. 5998–6008.
4. B. Saha and K. Goebel, "Battery data set," *NASA Ames Prognostics Data Repository*, 2007.
5. S. Oh *et al.*, [`SoH\\\_estimation\\\_of\\\_Lithium-ion\\\_battery`](https://github.com/standing-o/SoH_estimation_of_Lithium-ion_battery), GitHub Repository, 2020.

