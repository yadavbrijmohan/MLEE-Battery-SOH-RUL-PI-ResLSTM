"""
MLEE Course Project — BITS Pilani, Pilani Campus
Physics-Informed Attention Networks for Li-Ion Battery SOH & RUL Estimation
Reproduces the exact NASA B0005 benchmark (70% EOL = Cycle 125, 50/50 split)
using strictly MLEE syllabus models:
  - Baseline: Linear Regression (Unit 1.2.1)
  - Method A: Standard LSTM (Unit 4.3)
  - Proposed: PI-ResLSTM + MLP Temporal Attention (Units 4.3.3 & 2.1)
"""

import os
import glob
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

torch.manual_seed(42)
np.random.seed(42)

# 1. Auto-clone standing-o repo if not already present
repo_dir = "SoH_estimation_of_Lithium-ion_battery"
if not os.path.exists(repo_dir):
    subprocess.run(["git", "clone", "https://github.com/standing-o/SoH_estimation_of_Lithium-ion_battery.git"], check=True)

csv_files = sorted(glob.glob(f"{repo_dir}/**/refined_dataset/*.csv", recursive=True))
target_csv = [f for f in csv_files if "B05_discharge_soh.csv" in f][0]
df = pd.read_csv(target_csv)

# Drop raw 'capacity' to prevent target leakage (SOH = capacity / 2.0 * 100)
sensor_cols = ["terminal_voltage", "terminal_current", "temperature", "charge_current", "charge_voltage", "time"]
df_cyc = df.groupby("cycle", as_index=False).agg(
    SOH=("SOH", "mean"),
    **{c: (c, "mean") for c in sensor_cols}
).sort_values("cycle").reset_index(drop=True)

if df_cyc["SOH"].max() <= 1.5:
    df_cyc["SOH"] = df_cyc["SOH"] * 100.0

EOL_THRESHOLD = 70.0
below_eol = df_cyc.index[df_cyc["SOH"] <= EOL_THRESHOLD]
eol_idx = int(below_eol[0]) if len(below_eol) > 0 else len(df_cyc) - 1
df_cyc["RUL"] = np.maximum(0, eol_idx - np.arange(len(df_cyc)))
print(f"Battery: B05 | Total Cycles: {len(df_cyc)} | 70% EOL Cycle: {eol_idx + 1}")

# 2. Split 50% Train / 50% Test and Normalize Inputs & Targets (MLEE Unit 2.1.2)
SPLIT_RATIO = 0.50
WINDOW = 10
split_idx = int(len(df_cyc) * SPLIT_RATIO)

X_raw = df_cyc[sensor_cols + ["cycle"]].values
y_soh_raw = df_cyc["SOH"].values.reshape(-1, 1)
y_rul_raw = df_cyc["RUL"].values.reshape(-1, 1)

x_scaler = StandardScaler().fit(X_raw[:split_idx])
soh_scaler = MinMaxScaler().fit(y_soh_raw[:split_idx])
rul_scaler = MinMaxScaler().fit(y_rul_raw[:split_idx])

X_scaled = x_scaler.transform(X_raw)
y_soh_s = soh_scaler.transform(y_soh_raw).ravel()
y_rul_s = rul_scaler.transform(y_rul_raw).ravel()

X_seq, ys_seq, yp_seq, yr_seq, cyc_seq = [], [], [], [], []
for i in range(WINDOW, len(X_scaled)):
    X_seq.append(X_scaled[i-WINDOW:i])
    ys_seq.append(y_soh_s[i])
    yp_seq.append(y_soh_s[i-1])
    yr_seq.append(y_rul_s[i])
    cyc_seq.append(i + 1)

X_seq, ys_seq, yp_seq, yr_seq, cyc_seq = map(np.array, [X_seq, ys_seq, yp_seq, yr_seq, cyc_seq])
tr_end = split_idx - WINDOW

X_tr, X_te = X_seq[:tr_end], X_seq[tr_end:]
ys_tr, ys_te_s = ys_seq[:tr_end], ys_seq[tr_end:]
yp_tr, yp_te_s = yp_seq[:tr_end], yp_seq[tr_end:]
yr_tr, yr_te_s = yr_seq[:tr_end], yr_seq[tr_end:]
cyc_te = cyc_seq[tr_end:]

ys_te = soh_scaler.inverse_transform(ys_te_s.reshape(-1, 1)).ravel()
yp_te = soh_scaler.inverse_transform(yp_te_s.reshape(-1, 1)).ravel()
yr_te = rul_scaler.inverse_transform(yr_te_s.reshape(-1, 1)).ravel()

# 3. Baseline: Linear Regression (MLEE Unit 1.2.1)
lr_soh = LinearRegression().fit(X_tr[:, -1, :], ys_tr)
lr_rul = LinearRegression().fit(X_tr[:, -1, :], yr_tr)
pred_soh_lr = soh_scaler.inverse_transform(lr_soh.predict(X_te[:, -1, :]).reshape(-1, 1)).ravel()
pred_rul_lr = np.clip(rul_scaler.inverse_transform(lr_rul.predict(X_te[:, -1, :]).reshape(-1, 1)).ravel(), 0, None)

# 4. Method A: Standard LSTM (MLEE Unit 4.3)
class StandardLSTM(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.fc_soh = nn.Linear(hidden, 1)
        self.fc_rul = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return self.fc_soh(h).squeeze(-1), self.fc_rul(h).squeeze(-1)

# 5. Proposed: Physics-Informed Residual LSTM + MLP Attention (MLEE Units 4.3.3 & 2.1)
class PIResLSTMAttn(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.lstm1 = nn.LSTM(hidden, hidden, batch_first=True)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.mlp_attn = nn.Sequential(nn.Linear(hidden, 16), nn.Tanh(), nn.Linear(16, 1))
        self.skip_lin = nn.Linear(in_dim, 2)
        self.fc_soh = nn.Linear(hidden, 1)
        self.fc_rul = nn.Linear(hidden, 1)
    def forward(self, x):
        h0 = torch.relu(self.proj(x))
        h1 = h0 + self.lstm1(h0)[0]
        h2 = h1 + self.lstm2(h1)[0]
        w = torch.softmax(self.mlp_attn(h2).squeeze(-1), dim=-1)
        ctx = torch.sum(h2 * w.unsqueeze(-1), dim=1)
        trend = self.skip_lin(x[:, -1, :])
        soh_out = self.fc_soh(ctx).squeeze(-1) + trend[:, 0]
        rul_out = self.fc_rul(ctx).squeeze(-1) + trend[:, 1]
        return soh_out, rul_out, w

def train_and_eval(model, use_physics=False, epochs=350, lr=5e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    mse = nn.MSELoss()
    X_t = torch.tensor(X_tr, dtype=torch.float32)
    ys_t = torch.tensor(ys_tr, dtype=torch.float32)
    yp_t = torch.tensor(yp_tr, dtype=torch.float32)
    yr_t = torch.tensor(yr_tr, dtype=torch.float32)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(X_t)
        ps, pr = out[0], out[1]
        loss = mse(ps, ys_t) + mse(pr, yr_t)
        if use_physics:
            loss_mono = torch.mean(torch.relu(ps - yp_t - 0.04) ** 2)
            loss = loss + 0.5 * loss_mono
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        out_te = model(torch.tensor(X_te, dtype=torch.float32))
    ps_denorm = soh_scaler.inverse_transform(out_te[0].numpy().reshape(-1, 1)).ravel()
    pr_denorm = np.clip(rul_scaler.inverse_transform(out_te[1].numpy().reshape(-1, 1)).ravel(), 0, None)
    return ps_denorm, pr_denorm

pred_soh_lstm, pred_rul_lstm = train_and_eval(StandardLSTM(X_tr.shape[2]), use_physics=False)
pred_soh_pi, pred_rul_pi = train_and_eval(PIResLSTMAttn(X_tr.shape[2]), use_physics=True)

def report_metrics(name, ps, pr):
    soh_rmse = np.sqrt(mean_squared_error(ys_te, ps))
    soh_mae  = mean_absolute_error(ys_te, ps)
    rul_rmse = np.sqrt(mean_squared_error(yr_te, pr))
    r2       = r2_score(ys_te, ps)
    viol     = np.mean(((ps - yp_te) > 0.15) & ((ys_te - yp_te) <= 0.05)) * 100.0
    print(f"{name:34s} | SOH RMSE: {soh_rmse:5.2f}% | SOH MAE: {soh_mae:5.2f}% | RUL RMSE: {rul_rmse:5.2f} cyc | R2: {r2:6.4f} | Phys Viol: {viol:5.2f}%")

print("\n=== TEST RESULTS (B05_discharge_soh.csv | 70% EOL | Train 50% / Test 50%) ===")
report_metrics("Baseline: Linear Reg (Unit 1.2.1)", pred_soh_lr, pred_rul_lr)
report_metrics("Method A: LSTM (Unit 4.3)", pred_soh_lstm, pred_rul_lstm)
report_metrics("Proposed: PI-ResLSTM + Attn", pred_soh_pi, pred_rul_pi)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)
ax1.plot(cyc_te, ys_te, "k-", lw=2.2, label="True SOH")
ax1.plot(cyc_te, pred_soh_lr, "--", label="Baseline: Linear Reg")
ax1.plot(cyc_te, pred_soh_lstm, "-.", label="Method A: LSTM")
ax1.plot(cyc_te, pred_soh_pi, "-", lw=2.0, label="Proposed: PI-ResLSTM + Attn")
ax1.axhline(EOL_THRESHOLD, color="r", ls=":", label="70% EOL")
ax1.set_title("SOH Estimation on Test Split (B05)"); ax1.set_xlabel("Cycle"); ax1.set_ylabel("SOH (%)"); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(cyc_te, yr_te, "k-", lw=2.2, label="True RUL")
ax2.plot(cyc_te, pred_rul_lr, "--", label="Baseline: Linear Reg")
ax2.plot(cyc_te, pred_rul_lstm, "-.", label="Method A: LSTM")
ax2.plot(cyc_te, pred_rul_pi, "-", lw=2.0, label="Proposed: PI-ResLSTM + Attn")
ax2.set_title("RUL Prediction on Test Split (B05)"); ax2.set_xlabel("Cycle"); ax2.set_ylabel("RUL (Cycles)"); ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_result1.png", dpi=300)
plt.show()
