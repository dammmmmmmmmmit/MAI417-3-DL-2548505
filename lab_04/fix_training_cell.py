# ── FIX: redefine loss + optimizer + train (paste in a new cell) ──
import time, math
import numpy as np

def _sig(x):
    return 1.0 / (1.0 + xp.exp(-xp.clip(x, -15.0, 15.0)))

def detection_loss(pred, target, lc=5.0, lno=0.5):
    N, dpred = pred.shape[0], xp.zeros_like(pred)
    obj_p = _sig(pred[..., 0]); obj_t = target[..., 0]
    has_obj = obj_t > 0.5; mask4 = has_obj[..., None]
    bce = -(obj_t*xp.log(obj_p+1e-7) + (1-obj_t)*xp.log(1-obj_p+1e-7))
    dpred[..., 0] = xp.where(has_obj, obj_p-obj_t, lno*(obj_p-obj_t))
    loss = xp.where(has_obj, bce, lno*bce).sum()
    bd = pred[...,1:5] - target[...,1:5]
    loss += lc * xp.where(mask4, bd**2, 0.0).sum()
    dpred[...,1:5] = xp.where(mask4, 2*lc*bd, 0.0)
    cp = _sig(pred[...,5:]); cd = cp - target[...,5:]
    loss += xp.where(mask4, cd**2, 0.0).sum()
    dpred[...,5:] = xp.where(mask4, 2*cd*cp*(1-cp), 0.0)
    return float(loss)/N, dpred/N

class _SGD:
    def __init__(self, lr=0.001, mom=0.9, wd=1e-4, clip=1.0):
        self.lr, self.mom, self.wd, self.clip = lr, mom, wd, clip
        self.vel = {}
    def step(self, pgs):
        for i,(p,g) in enumerate(pgs):
            if g is None: continue
            if i not in self.vel: self.vel[i] = xp.zeros_like(p)
            gn = float(xp.sqrt(xp.sum(g**2)))
            if gn > self.clip: g = g*(self.clip/(gn+1e-8))
            self.vel[i] = self.mom*self.vel[i] - self.lr*(g + self.wd*p)
            p += self.vel[i]
    def decay(self, f=0.1):
        self.lr *= f; print(f"   LR -> {self.lr:.6f}")

EPOCHS, BATCH_SIZE = 30, 16
opt = _SGD(lr=0.001, mom=0.9, wd=1e-4, clip=1.0)
best_val, best_state = float('inf'), None
train_losses, val_losses = [], []

# Re-init model weights for clean start
cnn_scratch = SimpleCNN(num_classes=10, grid=8)

print(f"🚀 Training  |  LR={opt.lr}  |  grad_clip={opt.clip}")
print(f"   Epochs={EPOCHS}  |  Batch={BATCH_SIZE}")
print(f"   Train={len(train_imgs)}  |  Val={len(val_imgs)}")
print("="*55)

for epoch in range(EPOCHS):
    t0 = time.time()
    idx = xp.random.permutation(len(train_imgs))
    xi, yi = train_imgs[idx], train_tgts[idx]
    ep_loss, nb = 0.0, 0
    for s in range(0, len(xi), BATCH_SIZE):
        xb, yb = xi[s:s+BATCH_SIZE], yi[s:s+BATCH_SIZE]
        pred = cnn_scratch.forward(xb, training=True)
        loss, dpred = detection_loss(pred, yb)
        if not math.isfinite(loss): continue
        ep_loss += loss; nb += 1
        cnn_scratch.backward(dpred)
        opt.step(cnn_scratch.params())
    if nb == 0:
        print(f"Ep {epoch+1:3d}/{EPOCHS} | ALL NaN"); train_losses.append(float('nan')); val_losses.append(float('nan')); continue
    vp = cnn_scratch.forward(val_imgs[:32], training=False)
    vl, _ = detection_loss(vp, val_tgts[:32])
    tl = ep_loss/nb; train_losses.append(tl); val_losses.append(float(vl))
    star = ""
    if math.isfinite(vl) and vl < best_val:
        best_val = float(vl)
        if GPU_AVAILABLE: best_state = {k:cp.asnumpy(v) for k,v in {'c1W':cnn_scratch.conv1.W,'c2W':cnn_scratch.conv2.W,'c3W':cnn_scratch.conv3.W,'f1W':cnn_scratch.fc1.W,'f2W':cnn_scratch.fc2.W}.items()}
        star = " ⭐"
    if epoch == 15: opt.decay()
    print(f"Ep {epoch+1:3d}/{EPOCHS} | Train:{tl:.4f} | Val:{float(vl):.4f} | {time.time()-t0:.1f}s{star}")

np.save("scratch_cnn_best.npy", best_state)
print(f"\n✅ Done. Best val loss: {best_val:.4f}")
