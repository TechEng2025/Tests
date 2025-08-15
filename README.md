# Tests
# ============================================================
# Colab: Hotel Reviews Sentiment (50k) — Keras + PyTorch RNNs
# Benchmarks: Acc / Prec / Rec / F1 + BLEU / ROUGE-L / WER
# Auto-retrain detection + Historical logs + Public Drive links
# ============================================================

# 0) Install deps
!pip -q install textblob scikit-learn openpyxl rouge-score jiwer pydrive torch torchvision torchaudio --upgrade
import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')

# 1) Imports & config
import os, io, zipfile, random, json, time, re
from datetime import datetime, timedelta, timezone
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from rouge_score import rouge_scorer
from jiwer import wer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smooth = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Drive auth
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED); torch.manual_seed(SEED)

# Dataset/model settings
N = 50000
VOCAB_SIZE = 12000
MAX_LEN = 60
EMBED_DIM = 128
LSTM_UNITS = 128
KERAS_EPOCHS = 4
KERAS_BATCH = 256
TORCH_EPOCHS = 3
TORCH_BATCH = 256
LR = 1e-3

# Filenames
RAW_XLSX  = "hotel_reviews_50000_with_timestamp.xlsx"
PRED_XLSX = "hotel_reviews_50000_predictions.xlsx"
KERAS_H5  = "keras_bilstm_hotel.h5"
TORCH_PT  = "torch_lstm_hotel.pt"
LOG_CSV   = "benchmark_log.csv"
BUNDLE_ZIP= "hotel_sentiment_bundle.zip"

# 2) Helpers
def generate_dataset_excel(path=RAW_XLSX, rows=N):
    positive = [
        "Amazing stay! The room was spotless and the staff were incredibly friendly.",
        "Excellent service and a beautiful view from the balcony.",
        "The breakfast buffet was outstanding and the location was perfect.",
        "Highly recommend! Perfect location and comfortable rooms.",
        "Very clean, modern design, and helpful staff."
    ]
    neutral = [
        "The stay was okay, nothing special.",
        "Average experience; room was fine but not memorable.",
        "Location was convenient but facilities were standard.",
        "Neither particularly good nor bad; just an average hotel.",
        "Fine for a short stay; met basic needs."
    ]
    negative = [
        "Terrible service; the staff were rude and unhelpful.",
        "Room was dirty and smelled bad.",
        "The air conditioning did not work and it was very noisy.",
        "Horrible experience; I will never come back.",
        "The food was cold and tasteless."
    ]
    per = rows//3
    ist_offset = timedelta(hours=5, minutes=30)
    ts = (datetime.utcnow().replace(tzinfo=timezone.utc)+ist_offset).isoformat(timespec='seconds')
    data=[]
    for _ in range(per):           data.append([random.randint(4,5), random.choice(positive), 2, "", ts])
    for _ in range(per):           data.append([3,               random.choice(neutral), 1, "", ts])
    for _ in range(rows-2*per):    data.append([random.randint(1,2), random.choice(negative), 0, "", ts])
    random.shuffle(data)
    df_full = pd.DataFrame(data, columns=["rating","reviewText","Label","Predicted","Timestamp"])
    df_mis  = pd.DataFrame(columns=["rating","reviewText","Label","Keras_Pred","Keras_Prob","Torch_Pred","Torch_Prob","Timestamp","Last_Misclassified"])
    df_sum  = pd.DataFrame({
        "Total Rows":[len(df_full)],
        "Negative (0)":[int((df_full["Label"]==0).sum())],
        "Neutral (1)":[int((df_full["Label"]==1).sum())],
        "Positive (2)":[int((df_full["Label"]==2).sum())],
        "Generated At (Asia/Kolkata)":[ts]
    })
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df_full.to_excel(w, sheet_name="FullData", index=False)
        df_mis.to_excel(w, sheet_name="Misclassified", index=False)
        df_sum.to_excel(w, sheet_name="Summary", index=False)
    return path

def textblob_label(t, pos_th=0.12, neg_th=-0.12):
    p = TextBlob(str(t)).sentiment.polarity
    return 2 if p > pos_th else (0 if p < neg_th else 1)

def aux_metrics(y_true, y_pred):
    # Compute BLEU/ROUGE-L/WER over label words ("Negative/Neutral/Positive")
    map_lbl = {0:"Negative",1:"Neutral",2:"Positive"}
    refs = [map_lbl[int(a)] for a in y_true]
    hyps = [map_lbl[int(p)] for p in y_pred]
    bleu = np.mean([sentence_bleu([r.split()], h.split(), smoothing_function=smooth) for r,h in zip(refs,hyps)])
    rougeL = np.mean([rouge.score(r,h)["rougeL"].fmeasure for r,h in zip(refs,hyps)])
    wers = np.mean([wer(r,h) for r,h in zip(refs,hyps)])
    return float(bleu), float(rougeL), float(wers)

def acc_prec_rec_f1(y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return float(acc), float(prec), float(rec), float(f1)

# 3) Data: create or load; detect retraining need (hash by size & modtime)
def file_signature(path):
    st = os.stat(path)
    return f"{st.st_size}_{int(st.st_mtime)}"

if not os.path.exists(RAW_XLSX):
    print("Generating 50k dataset…")
    generate_dataset_excel(RAW_XLSX)
else:
    print("Dataset found:", RAW_XLSX)

sig_before = file_signature(RAW_XLSX)

df = pd.read_excel(RAW_XLSX, sheet_name="FullData")
X = df["reviewText"].astype(str).values
y = df["Label"].astype(int).values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)

# 4) Keras BiLSTM
tokenizer_keras = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer_keras.fit_on_texts(X_train)
X_train_pad = pad_sequences(tokenizer_keras.texts_to_sequences(X_train), maxlen=MAX_LEN, padding="post", truncating="post")
X_test_pad  = pad_sequences(tokenizer_keras.texts_to_sequences(X_test),  maxlen=MAX_LEN, padding="post", truncating="post")

keras_model = Sequential([
    Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.1)),
    Dense(128, activation="relu"),
    Dropout(0.4),
    Dense(3, activation="softmax")
])
keras_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
es = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
hist_keras = keras_model.fit(X_train_pad, y_train, validation_split=0.1, epochs=KERAS_EPOCHS, batch_size=KERAS_BATCH, verbose=2, callbacks=[es])

k_probs = keras_model.predict(X_test_pad, verbose=0)
k_pred  = np.argmax(k_probs, axis=-1)
k_acc, k_prec, k_rec, k_f1 = acc_prec_rec_f1(y_test, k_pred)
k_bleu, k_rouge, k_wer = aux_metrics(y_test, k_pred)
print(f"Keras BiLSTM -> Acc:{k_acc:.4f} Prec:{k_prec:.4f} Rec:{k_rec:.4f} F1:{k_f1:.4f} | BLEU:{k_bleu:.4f} ROUGE-L:{k_rouge:.4f} WER:{k_wer:.4f}")

# 5) PyTorch LSTM (use same tokenizer vocab)
class TextDataset(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        seqs = tok.texts_to_sequences(texts)
        self.x = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")
        self.y = np.array(labels, dtype=np.int64)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.tensor(self.x[i], dtype=torch.long), torch.tensor(self.y[i], dtype=torch.long)

train_ds = TextDataset(X_train, y_train, tokenizer_keras, MAX_LEN)
test_ds  = TextDataset(X_test,  y_test,  tokenizer_keras, MAX_LEN)
train_dl = DataLoader(train_ds, batch_size=TORCH_BATCH, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=TORCH_BATCH, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TorchLSTM(nn.Module):
    def __init__(self, vocab, emb, hidden, num_classes=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden*2, 128)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.4)
        self.out = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.emb(x)
        o, (h, c) = self.lstm(x)
        h_cat = torch.cat((h[-2], h[-1]), dim=1)  # last fwd/bwd
        x = self.fc1(h_cat)
        x = self.relu(x); x = self.drop(x)
        return self.out(x)

model_t = TorchLSTM(VOCAB_SIZE, EMBED_DIM, LSTM_UNITS).to(device)
opt = torch.optim.Adam(model_t.parameters(), lr=LR)
crit = nn.CrossEntropyLoss()

def run_epoch(dl, train=True):
    model_t.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    all_pred, all_true = [], []
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        if train: opt.zero_grad()
        logits = model_t(xb)
        loss = crit(logits, yb)
        if train:
            loss.backward(); opt.step()
        loss_sum += loss.item()*yb.size(0)
        pred = logits.argmax(1)
        correct += (pred==yb).sum().item()
        total += yb.size(0)
        all_pred.append(pred.detach().cpu().numpy()); all_true.append(yb.detach().cpu().numpy())
    return loss_sum/total, correct/total, np.concatenate(all_pred), np.concatenate(all_true)

for ep in range(1, TORCH_EPOCHS+1):
    tr_loss, tr_acc, _, _ = run_epoch(train_dl, True)
    te_loss, te_acc, t_pred, y_true_ep = run_epoch(test_dl, False)
    print(f"Epoch {ep}/{TORCH_EPOCHS} | train_acc {tr_acc:.4f} | val_acc {te_acc:.4f}")

t_acc, t_prec, t_rec, t_f1 = acc_prec_rec_f1(y_test, t_pred)
t_bleu, t_rouge, t_wer = aux_metrics(y_test, t_pred)
print(f"Torch LSTM   -> Acc:{t_acc:.4f} Prec:{t_prec:.4f} Rec:{t_rec:.4f} F1:{t_f1:.4f} | BLEU:{t_bleu:.4f} ROUGE-L:{t_rouge:.4f} WER:{t_wer:.4f}")

# 6) TextBlob baseline on test (fast reference)
tb_pred = np.array([textblob_label(t) for t in X_test])
tb_acc, tb_prec, tb_rec, tb_f1 = acc_prec_rec_f1(y_test, tb_pred)
tb_bleu, tb_rouge, tb_wer = aux_metrics(y_test, tb_pred)
print(f"TextBlob     -> Acc:{tb_acc:.4f} Prec:{tb_prec:.4f} Rec:{tb_rec:.4f} F1:{tb_f1:.4f} | BLEU:{tb_bleu:.4f} ROUGE-L:{tb_rouge:.4f} WER:{tb_wer:.4f}")

# 7) Save predictions to Excel (full dataset with confidences + misclassified)
imap = {0:"Negative",1:"Neutral",2:"Positive"}
def predict_full_keras(texts):
    seq = pad_sequences(tokenizer_keras.texts_to_sequences(texts), maxlen=MAX_LEN, padding="post", truncating="post")
    probs = keras_model.predict(seq, verbose=0)
    return probs.argmax(axis=1), probs.max(axis=1)

def predict_full_torch(texts):
    seq = pad_sequences(tokenizer_keras.texts_to_sequences(texts), maxlen=MAX_LEN, padding="post", truncating="post")
    dl = DataLoader(torch.tensor(seq, dtype=torch.long), batch_size=TORCH_BATCH, shuffle=False)
    model_t.eval(); preds, confs = [], []
    with torch.no_grad():
        for xb in dl:
            xb = xb.to(device)
            logits = model_t(xb)
            p = logits.argmax(1); pr = torch.softmax(logits, dim=1).max(1).values
            preds.extend(p.cpu().numpy()); confs.extend(pr.cpu().numpy())
    return np.array(preds), np.array(confs)

all_texts = df["reviewText"].astype(str).values
k_all_pred, k_all_conf = predict_full_keras(all_texts)
t_all_pred, t_all_conf = predict_full_torch(all_texts)

df_out = df.copy()
df_out["Keras_Pred"] = [imap[int(p)] for p in k_all_pred]
df_out["Keras_Prob"] = k_all_conf
df_out["Torch_Pred"] = [imap[int(p)] for p in t_all_pred]
df_out["Torch_Prob"] = t_all_conf

mask_mis = (df_out["Keras_Pred"] != df_out["Label"].map(imap)) | (df_out["Torch_Pred"] != df_out["Label"].map(imap))
df_mis = df_out.loc[mask_mis, ["rating","reviewText","Label","Keras_Pred","Keras_Prob","Torch_Pred","Torch_Prob","Timestamp"]].copy()
ist_offset = timedelta(hours=5, minutes=30)
now_ist = (datetime.utcnow().replace(tzinfo=timezone.utc)+ist_offset).isoformat(timespec="seconds")
df_mis["Last_Misclassified"] = now_ist

df_summary = pd.DataFrame({
    "Metric":[
        "TB_Acc","TB_Prec","TB_Rec","TB_F1","TB_BLEU","TB_ROUGE_L","TB_WER",
        "K_Acc","K_Prec","K_Rec","K_F1","K_BLEU","K_ROUGE_L","K_WER",
        "T_Acc","T_Prec","T_Rec","T_F1","T_BLEU","T_ROUGE_L","T_WER",
        "Rows","Misclassified"
    ],
    "Value":[
        tb_acc,tb_prec,tb_rec,tb_f1,tb_bleu,tb_rouge,tb_wer,
        k_acc,k_prec,k_rec,k_f1,k_bleu,k_rouge,k_wer,
        t_acc,t_prec,t_rec,t_f1,t_bleu,t_rouge,t_wer,
        int(len(df_out)), int(len(df_mis))
    ]
})

with pd.ExcelWriter(PRED_XLSX, engine="openpyxl") as w:
    df_out.to_excel(w, sheet_name="FullData", index=False)
    df_mis.to_excel(w, sheet_name="Misclassified", index=False)
    df_summary.to_excel(w, sheet_name="Summary", index=False)
print("Saved:", PRED_XLSX)

# 8) Save models
keras_model.save(KERAS_H5)
torch.save(model_t.state_dict(), TORCH_PT)

# 9) Historical benchmark logging (append row per run)
run_stamp = now_ist  # IST timestamp
log_row = {
    "run_time_ist": run_stamp,
    # TextBlob
    "tb_acc": tb_acc, "tb_prec": tb_prec, "tb_rec": tb_rec, "tb_f1": tb_f1, "tb_bleu": tb_bleu, "tb_rougeL": tb_rouge, "tb_wer": tb_wer,
    # Keras
    "k_acc": k_acc, "k_prec": k_prec, "k_rec": k_rec, "k_f1": k_f1, "k_bleu": k_bleu, "k_rougeL": k_rouge, "k_wer": k_wer,
    # Torch
    "t_acc": t_acc, "t_prec": t_prec, "t_rec": t_rec, "t_f1": t_f1, "t_bleu": t_bleu, "t_rougeL": t_rouge, "t_wer": t_wer,
    # retrain flag (if source dataset changed since last run)
    "dataset_sig": file_signature(RAW_XLSX),
}
# Local log (we'll merge with Drive version)
if os.path.exists(LOG_CSV):
    log_df = pd.read_csv(LOG_CSV)
else:
    log_df = pd.DataFrame(columns=list(log_row.keys()))
log_df = pd.concat([log_df, pd.DataFrame([log_row])], ignore_index=True)
log_df.to_csv(LOG_CSV, index=False)

# 10) Plots of historical metrics
def plot_history(df_hist, metric_keys, title, outfile):
    plt.figure(figsize=(8,4))
    for k in metric_keys:
        if k in df_hist.columns:
            plt.plot(df_hist[k].astype(float).values, label=k)
    plt.title(title); plt.xlabel("Run #"); plt.ylabel("Score")
    plt.legend(); plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

plot_history(log_df, ["tb_f1","k_f1","t_f1"], "F1 over runs", "hist_f1.png")
plot_history(log_df, ["tb_acc","k_acc","t_acc"], "Accuracy over runs", "hist_acc.png")
plot_history(log_df, ["k_bleu","t_bleu"], "BLEU over runs", "hist_bleu.png")
plot_history(log_df, ["k_rougeL","t_rougeL"], "ROUGE-L over runs", "hist_rouge.png")
plot_history(log_df, ["k_wer","t_wer"], "WER over runs (lower is better)", "hist_wer.png")

# 11) Auto-retrain detection note
sig_after = file_signature(RAW_XLSX)
auto_retrained = (sig_before != sig_after)
print("Auto-retrain check (dataset changed during run):", auto_retrained)

# 12) Upload to Google Drive + make PUBLIC links
auth.authenticate_user()
gauth = GoogleAuth(); gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# Find or create folder
FOLDER_TITLE = "Hotel_Sentiment_Analysis"
# Try to find an existing folder to append logs across sessions
file_list = drive.ListFile({'q': f"mimeType='application/vnd.google-apps.folder' and title='{FOLDER_TITLE}' and trashed=false"}).GetList()
if file_list:
    folder = file_list[0]
else:
    folder = drive.CreateFile({'title': FOLDER_TITLE, 'mimeType': 'application/vnd.google-apps.folder'}); folder.Upload()
folder_id = folder['id']
print("Drive folder:", FOLDER_TITLE, folder_id)

# If there is an old log in Drive, merge it
drive_logs = drive.ListFile({'q': f"'{folder_id}' in parents and title='{LOG_CSV}' and trashed=false"}).GetList()
if drive_logs:
    # Download and merge
    tmp = drive_logs[0]
    tmp.GetContentFile("drive_" + LOG_CSV)
    try:
        old_df = pd.read_csv("drive_" + LOG_CSV)
        merged = pd.concat([old_df, log_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["run_time_ist","dataset_sig"], keep='last')
        merged.to_csv(LOG_CSV, index=False)
        log_df = merged
    except Exception as e:
        print("Log merge warning:", e)

# Bundle zip
with zipfile.ZipFile(BUNDLE_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for p in [RAW_XLSX, PRED_XLSX, KERAS_H5, TORCH_PT, LOG_CSV, "hist_f1.png","hist_acc.png","hist_bleu.png","hist_rouge.png","hist_wer.png"]:
        z.write(p)

def upload_public(local_path, parent_id, title=None):
    title = title or os.path.basename(local_path)
    f = drive.CreateFile({'title': title, 'parents':[{'id': parent_id}]})
    f.SetContentFile(local_path); f.Upload()
    # Public read
    f.InsertPermission({'type':'anyone','value':'anyone','role':'reader'})
    # Prefer direct download link if available; else alternateLink
    link = f.get('webContentLink') or f.get('alternateLink')
    return link

artifacts = [RAW_XLSX, PRED_XLSX, KERAS_H5, TORCH_PT, LOG_CSV, "hist_f1.png","hist_acc.png","hist_bleu.png","hist_rouge.png","hist_wer.png", BUNDLE_ZIP]
links = {p: upload_public(p, folder_id) for p in artifacts}

print("\n✅ Public download links (tap to open/download):")
for k,v in links.items():
    print(f"{k}: {v}")

print("\nDone. Re-run later after adding new reviews to the Excel to append new benchmarks and update the history plots.")
