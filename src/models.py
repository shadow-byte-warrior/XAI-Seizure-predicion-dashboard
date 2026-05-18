import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# ============================================================
# UTILS & SHARED LAYERS
# ============================================================

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(alpha)
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        alpha, = ctx.saved_tensors
        return -alpha * grad, None

class GRL(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, alpha=None):
        if alpha is None:
            alpha = torch.tensor(1.0)
        return GradientReversalFunction.apply(x, alpha.to(x.device))

class SpikeEncoder(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold))
    def forward(self, x):
        spikes = (x.abs() > self.threshold).float()
        mem = torch.zeros_like(x)
        for t in range(x.shape[1]):
            prev = mem[:, t - 1] if t > 0 else torch.zeros_like(x[:, 0])
            mem[:, t] = 0.9 * prev + x[:, t]
        return spikes + 0.1 * torch.tanh(mem)

class SpikingConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, k, padding=k // 2)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(2)
    def forward(self, x):
        return F.relu(self.pool(self.bn(self.conv(x))))

class AttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)
        self.weights = None
    def forward(self, x):
        self.weights = torch.softmax(self.attn(x), dim=1)
        return (self.weights * x).sum(dim=1)

# ============================================================
# CORE ARCHITECTURES
# ============================================================

class SCT(nn.Module):
    def __init__(self, n_ch=18, T=1024, n_classes=1, d_model=64, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.spike_enc = SpikeEncoder()
        self.conv1 = SpikingConvBlock(n_ch, 32, k=7)
        self.conv2 = SpikingConvBlock(32, 64, k=5)
        self.conv3 = SpikingConvBlock(64, d_model, k=3)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pool_fc = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, n_classes), nn.Sigmoid())

    def forward(self, x):
        x = self.spike_enc(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        x = x.permute(0, 2, 1)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        return self.pool_fc(x[:, 0]).squeeze(-1)

class GRLSeizureModel(nn.Module):
    def __init__(self, n_ch=18, T=1024, d=64, dropout=0.3):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(n_ch, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(16),
        )
        self.label_clf = nn.Sequential(nn.Linear(128 * 16, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1), nn.Sigmoid())
        self.grl = GRL()
        self.domain_clf = nn.Sequential(nn.Linear(128 * 16, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x, alpha=1.0):
        x = x.permute(0, 2, 1)
        f = self.feat(x).view(x.size(0), -1)
        label_out = self.label_clf(f)
        domain_out = self.domain_clf(self.grl(f, torch.tensor(alpha)))
        return label_out.squeeze(-1), domain_out

class FederatedModel(nn.Module):
    def __init__(self, n_ch=18, T=1024, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_ch, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 8, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.classifier(self.features(x)).squeeze(-1)

class DASCT(nn.Module):
    def __init__(self, n_ch=18, T=1024, n_domains=2, d_model=64, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.spike_enc = SpikeEncoder()
        self.conv1 = SpikingConvBlock(n_ch, 32, k=7); self.conv2 = SpikingConvBlock(32, 64, k=5); self.conv3 = SpikingConvBlock(64, d_model, k=3)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.label_clf = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1), nn.Sigmoid())
        self.grl = GRL(); self.domain_clf = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_domains))

    def encode(self, x):
        x = self.spike_enc(x).permute(0, 2, 1)
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        x = x.permute(0, 2, 1)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = self.transformer(torch.cat([cls, x], dim=1))
        return x[:, 0]

    def forward(self, x, alpha=1.0):
        feat = self.encode(x)
        label_out = self.label_clf(feat).squeeze(-1)
        domain_out = self.domain_clf(self.grl(feat, torch.tensor(alpha)))
        return label_out, domain_out

class DAGRLModel(nn.Module):
    def __init__(self, n_ch=18, d_model=128, n_domains=2, dropout=0.3):
        super().__init__()
        self.conv_s = nn.Sequential(nn.Conv1d(n_ch, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
        self.conv_m = nn.Sequential(nn.Conv1d(n_ch, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
        self.conv_l = nn.Sequential(nn.Conv1d(n_ch, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
        self.deep = nn.Sequential(nn.Conv1d(96, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(128, d_model, 3, padding=1), nn.BatchNorm1d(d_model), nn.ReLU())
        self.attn_pool = AttentionPool(d_model)
        self.label_clf = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        self.grl = GRL(); self.domain_clf = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, n_domains))

    def encode(self, x):
        x = x.permute(0, 2, 1)
        f = torch.cat([self.conv_s(x), self.conv_m(x), self.conv_l(x)], dim=1)
        f = self.deep(f).permute(0, 2, 1)
        return self.attn_pool(f)

    def forward(self, x, alpha=1.0):
        feat = self.encode(x)
        label_out = self.label_clf(feat).squeeze(-1)
        domain_out = self.domain_clf(self.grl(feat, torch.tensor(alpha)))
        return label_out, domain_out
