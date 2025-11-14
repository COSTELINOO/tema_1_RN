import pickle
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import time

EPOCHS = 450
BATCH_SIZE = 2048
LR_INIT = 0.16
WARMUP_EPOCHS = 5
CLIP_NORM = 10.0
MOMENTUM = 0.9
SCHED_PATIENCE = 2
SCHED_DECAY = 0.5
MIN_LR = 1e-3
SEED = 42
TTA_MAX_SHIFT = 1
EMA_DECAY = 0.999

train_file = "/kaggle/input/fii-nn-2025-homework-3/extended_mnist_train.pkl"
test_file  = "/kaggle/input/fii-nn-2025-homework-3/extended_mnist_test.pkl"

with open(train_file, "rb") as fp:
    train = pickle.load(fp)
with open(test_file, "rb") as fp:
    test = pickle.load(fp)

train_images, train_labels = [], []
for img, label in train:
    train_images.append(img.astype(np.float32) / 255.0)
    train_labels.append(label)
test_images = [img.astype(np.float32) / 255.0 for img, _ in test]

X = np.stack([im.flatten() for im in train_images]).astype(np.float32)
y = np.array(train_labels, dtype=np.int64)
X_test = np.stack([im.flatten() for im in test_images]).astype(np.float32)

def stratified_split(X, y, val_ratio=0.1, seed=SEED):
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    train_idx, val_idx = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = int(len(idx) * val_ratio)
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    train_idx = np.concatenate(train_idx)
    val_idx   = np.concatenate(val_idx)
    rng.shuffle(train_idx); rng.shuffle(val_idx)
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

X_train, y_train, X_val, y_val = stratified_split(X, y, 0.1, SEED)

mean = X_train.mean(axis=0, keepdims=True)
std  = X_train.std(axis=0, keepdims=True) + 1e-6
X_train = (X_train - mean) / std
X_val   = (X_val   - mean) / std
X_test  = (X_test  - mean) / std

def he_uniform(in_dim, out_dim, rng):
    limit = np.sqrt(6.0 / in_dim)
    return rng.uniform(-limit, limit, size=(in_dim, out_dim)).astype(np.float32)
def xavier_uniform(in_dim, out_dim, rng):
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    return rng.uniform(-limit, limit, size=(in_dim, out_dim)).astype(np.float32)
def relu(x): return np.maximum(0.0, x)
def relu_grad(x):
    g = np.zeros_like(x, dtype=np.float32); g[x>0.0] = 1.0; return g
def softmax_logits(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True))

class MLP:
    def __init__(self, in_dim=784, hidden=100, out_dim=10, clip_norm=CLIP_NORM, seed=SEED):
        rng = np.random.RandomState(seed)
        self.W1 = he_uniform(in_dim, hidden, rng)
        self.b1 = np.zeros((1, hidden), dtype=np.float32)
        self.W2 = xavier_uniform(hidden, out_dim, rng)
        self.b2 = np.zeros((1, out_dim), dtype=np.float32)
        self.clip = clip_norm
        self.vW1 = np.zeros_like(self.W1); self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2); self.vb2 = np.zeros_like(self.b2)
    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        h1 = relu(z1)
        z2 = h1 @ self.W2 + self.b2
        return z2, (X, z1, h1, z2)
    def backward(self, cache, y_true):
        X, z1, h1, z2 = cache
        B = X.shape[0]
        p = softmax_logits(z2)
        p[np.arange(B), y_true] -= 1.0
        p /= B
        dW2 = h1.T @ p
        db2 = p.sum(axis=0, keepdims=True)
        dh1 = p @ self.W2.T
        dz1 = dh1 * relu_grad(z1)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)
        if self.clip is not None and self.clip > 0:
            tot = np.sqrt(np.sum(dW1*dW1)+np.sum(db1*db1)+np.sum(dW2*dW2)+np.sum(db2*db2)) + 1e-12
            if tot > self.clip:
                s = self.clip / tot
                dW1*=s; db1*=s; dW2*=s; db2*=s
        return dW1, db1, dW2, db2
    def sgd_momentum(self, grads, lr, momentum=MOMENTUM):
        dW1, db1, dW2, db2 = grads
        self.vW1 = momentum * self.vW1 - lr * dW1
        self.vb1 = momentum * self.vb1 - lr * db1
        self.vW2 = momentum * self.vW2 - lr * dW2
        self.vb2 = momentum * self.vb2 - lr * db2
        self.W1 += self.vW1; self.b1 += self.vb1
        self.W2 += self.vW2; self.b2 += self.vb2
    def get_weights(self):
        return (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())
    def set_weights(self, w):
        self.W1[:], self.b1[:], self.W2[:], self.b2[:] = w
    def predict(self, X, bs=4096):
        out = []
        for i in range(0, X.shape[0], bs):
            z2, _ = self.forward(X[i:i+bs])
            out.append(np.argmax(z2, axis=1))
        return np.concatenate(out)

def random_shift_batch(xb, max_shift=2, rng=None):
    if rng is None:
        rng = np.random.RandomState(int(time.time()) % 10_000_000)
    B = xb.shape[0]
    xb_img = xb.reshape(B, 28, 28)
    pad = np.pad(xb_img, ((0,0),(max_shift,max_shift),(max_shift,max_shift)), mode="constant")
    out = np.empty_like(xb_img)
    for i in range(B):
        dy = rng.randint(-max_shift, max_shift+1)
        dx = rng.randint(-max_shift, max_shift+1)
        y0 = max_shift + dy
        x0 = max_shift + dx
        out[i] = pad[i, y0:y0+28, x0:x0+28]
    return out.reshape(B, -1)

def tta_logits(model, X, max_shift=TTA_MAX_SHIFT, bs=4096):
    shifts = [-max_shift, 0, max_shift]
    acc_logits = None
    for dy in shifts:
        for dx in shifts:
            B = X.shape[0]
            xb_img = X.reshape(B, 28, 28)
            pad = np.pad(xb_img, ((0,0),(max_shift,max_shift),(max_shift,max_shift)), mode="constant")
            y0 = max_shift + dy
            x0 = max_shift + dx
            shifted = pad[:, y0:y0+28, x0:x0+28].reshape(B, -1)
            chunk_logits = []
            for i in range(0, shifted.shape[0], bs):
                z2, _ = model.forward(shifted[i:i+bs])
                chunk_logits.append(z2)
            z = np.vstack(chunk_logits)
            acc_logits = z if acc_logits is None else acc_logits + z
    acc_logits /= (len(shifts) ** 2)
    return acc_logits

def batches(X, y, bs, shuffle=True, seed=SEED, augment=False):
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.RandomState(seed + int(time.time()) % 100000)
        rng.shuffle(idx)
    for i in range(0, n, bs):
        sel = idx[i:i+bs]
        xb, yb = X[sel], y[sel]
        if augment:
            xb = random_shift_batch(xb, max_shift=2)
        yield xb, yb

def ce_loss_torch(logits_np, y_np, reduction="mean"):
    logits_t = torch.from_numpy(logits_np.copy())
    y_t = torch.from_numpy(y_np.copy())
    return F.cross_entropy(logits_t, y_t, reduction=reduction).item()

def evaluate(model, X, y, bs=2048, use_ema=False, ema=None, use_tta=False):
    if use_ema and ema is not None:
        bak = model.get_weights()
        model.set_weights(ema)
    if use_tta:
        logits = tta_logits(model, X, max_shift=TTA_MAX_SHIFT, bs=bs)
        loss = ce_loss_torch(logits, y, reduction="mean")
        pred = np.argmax(logits, axis=1)
        acc = (pred == y).mean()
        if use_ema and ema is not None:
            model.set_weights(bak)
        return loss, acc
    totL, totC, totN = 0.0, 0, 0
    for xb, yb in batches(X, y, bs, shuffle=False, augment=False):
        logits, _ = model.forward(xb)
        loss = ce_loss_torch(logits, yb, reduction="sum")
        pred = np.argmax(logits, axis=1)
        totL += loss; totC += (pred == yb).sum(); totN += yb.size
    if use_ema and ema is not None:
        model.set_weights(bak)
    return totL / totN, totC / totN

def train(model, Xtr, ytr, Xva, yva, lr0=LR_INIT, epochs=EPOCHS, bs=BATCH_SIZE):
    lr = lr0
    best_val = float("inf")
    no_imp = 0
    ema = model.get_weights()
    for ep in range(1, epochs+1):
        if ep <= WARMUP_EPOCHS:
            warm_frac = ep / WARMUP_EPOCHS
            lr = max(MIN_LR, lr0 * (0.25 + 0.75 * warm_frac))
        seen, runL, runC = 0, 0.0, 0
        for xb, yb in batches(Xtr, ytr, bs, shuffle=True, augment=True):
            logits, cache = model.forward(xb)
            loss = ce_loss_torch(logits, yb, reduction="sum") / yb.shape[0]
            grads = model.backward(cache, yb)
            model.sgd_momentum(grads, lr, momentum=MOMENTUM)
            pred = np.argmax(logits, axis=1)
            runL += loss * yb.shape[0]
            runC += (pred == yb).sum()
            seen += yb.shape[0]
            w = model.get_weights()
            ema = tuple(EMA_DECAY * e + (1.0 - EMA_DECAY) * w_i for e, w_i in zip(ema, w))
        tr_loss = runL / seen
        tr_acc  = runC / seen
        va_loss, va_acc = evaluate(model, Xva, yva, bs=4096, use_ema=True, ema=ema, use_tta=False)
        if va_loss + 1e-6 < best_val:
            best_val = va_loss
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= SCHED_PATIENCE and lr > MIN_LR and ep > WARMUP_EPOCHS:
                lr = max(MIN_LR, lr * SCHED_DECAY)
                no_imp = 0
        print(f"Ep {ep:03d}/{epochs} | lr={lr:.4f} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} acc={va_acc:.4f}")
    return model, ema

model = MLP(in_dim=784, hidden=100, out_dim=10, clip_norm=CLIP_NORM, seed=SEED)
model, ema = train(model, X_train, y_train, X_val, y_val, lr0=LR_INIT, epochs=EPOCHS, bs=BATCH_SIZE)

tr_loss, tr_acc = evaluate(model, X_train, y_train, bs=4096, use_ema=True, ema=ema, use_tta=False)
va_loss, va_acc = evaluate(model, X_val, y_val, bs=4096, use_ema=True, ema=ema, use_tta=True)
print(f"FINAL -> TRAIN: loss={tr_loss:.4f}, acc={tr_acc:.4f} | VAL(TTA+EMA): loss={va_loss:.4f}, acc={va_acc:.4f}")

bak = model.get_weights()
model.set_weights(ema)
logits_test = tta_logits(model, X_test, max_shift=TTA_MAX_SHIFT, bs=4096)
model.set_weights(bak)
preds_test = np.argmax(logits_test, axis=1)
pd.DataFrame({"ID": np.arange(len(preds_test)), "target": preds_test}).to_csv("submission.csv", index=False)
print("Saved submission.csv")