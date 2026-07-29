"""Model architectures.

- LeNet-5           : MNIST (AdaptFU / FuGuard use LeNet-5 for MNIST/FEMNIST)
- ResNet-18 (BN)    : CIFAR-10 / CIFAR-100 (AdaptFU); a compact CNN is fine too
- ConvVAE           : FuGuard's pretrained proxy generator

Every classifier exposes `.features(x)` returning the penultimate embedding, which
FuGuard's optimal-transport regularizer operates on (representation layer before
the final classifier).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, channels=1, n_classes=10, img=28):
        super().__init__()
        self.c1 = nn.Conv2d(channels, 6, 5, padding=2)
        self.c2 = nn.Conv2d(6, 16, 5)
        # infer flatten dim
        with torch.no_grad():
            d = self._conv(torch.zeros(1, channels, img, img)).shape[1]
        self.f1 = nn.Linear(d, 120)
        self.f2 = nn.Linear(120, 84)
        self.head = nn.Linear(84, n_classes)

    def _conv(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return x.flatten(1)

    def features(self, x):
        x = self._conv(x)
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        return x

    def forward(self, x):
        return self.head(self.features(x))


class BasicBlock(nn.Module):
    exp = 1
    def __init__(self, inp, out, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(inp, out, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(out)
        self.c2 = nn.Conv2d(out, out, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(out)
        self.sc = nn.Sequential()
        if stride != 1 or inp != out:
            self.sc = nn.Sequential(nn.Conv2d(inp, out, 1, stride, bias=False),
                                    nn.BatchNorm2d(out))
    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        return F.relu(o + self.sc(x))


class ResNet18(nn.Module):
    def __init__(self, channels=3, n_classes=10):
        super().__init__()
        self.inp = 64
        self.c1 = nn.Conv2d(channels, 64, 3, 1, 1, bias=False)
        self.b1 = nn.BatchNorm2d(64)
        self.l1 = self._layer(64, 2, 1)
        self.l2 = self._layer(128, 2, 2)
        self.l3 = self._layer(256, 2, 2)
        self.l4 = self._layer(512, 2, 2)
        self.head = nn.Linear(512, n_classes)

    def _layer(self, out, n, stride):
        strides = [stride] + [1] * (n - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.inp, out, s))
            self.inp = out
        return nn.Sequential(*layers)

    def features(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = self.l1(x); x = self.l2(x); x = self.l3(x); x = self.l4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x

    def forward(self, x):
        return self.head(self.features(x))


class TextCNN(nn.Module):
    """TextCNN as used by both papers for AG News: embedding -> parallel 1D convs
    with kernel sizes 3/4/5 -> ReLU -> max-over-time pool -> dropout -> FC.

    forward() accepts EITHER token ids (LongTensor, [B,L]) or embeddings
    (FloatTensor, [B,L,D]). The second path is what lets FuGuard synthesise its
    proxy in embedding space -- its ConvVAE assumes images and cannot be used for
    text, so PCA perturbation is applied to embeddings instead."""

    def __init__(self, vocab_size=20000, embed_dim=128, n_classes=4,
                 kernels=(3, 4, 5), n_filters=100, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, n_filters, k)
                                    for k in kernels])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(n_filters * len(kernels), n_classes)

    def _embed(self, x):
        return self.emb(x) if x.dtype == torch.long else x

    def features(self, x):
        e = self._embed(x).transpose(1, 2)             # [B, D, L]
        z = [F.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.drop(torch.cat(z, dim=1))

    def forward(self, x):
        return self.head(self.features(x))


def build_classifier(dataset_name, meta):
    if dataset_name == "agnews":
        return TextCNN(meta["vocab_size"], meta["embed_dim"], meta["n_classes"])
    if dataset_name in ("mnist", "femnist"):
        return LeNet5(meta["channels"], meta["n_classes"], meta["img"])
    return ResNet18(meta["channels"], meta["n_classes"])


# --------------------------- FuGuard VAE ---------------------------
class ConvVAE(nn.Module):
    """Small convolutional VAE used to synthesize FuGuard proxy samples."""
    def __init__(self, channels=1, img=28, latent=32):
        super().__init__()
        self.channels, self.img, self.latent = channels, img, latent
        h = img // 4
        self.h = h
        self.enc = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU())
        self.fc_mu = nn.Linear(64 * h * h, latent)
        self.fc_lv = nn.Linear(64 * h * h, latent)
        self.fc_dec = nn.Linear(latent, 64 * h * h)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, channels, 4, 2, 1))

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)

    def reparam(self, mu, lv):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 64, self.h, self.h)
        return self.dec(h)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        return self.decode(z), mu, lv
