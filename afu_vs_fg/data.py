"""Datasets, non-IID partitioning, and backdoor trigger injection.

The Dirichlet(alpha) label-skew partition is the standard non-IID protocol used
by both papers. A smaller alpha => more skew. We keep the partition deterministic
given the seed so every method sees identical client data.
"""
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms


DATASET_META = {
    "mnist":   dict(n_classes=10,  channels=1, img=28),
    "femnist": dict(n_classes=62,  channels=1, img=28),
    # AG News is TEXT: no channels/img. vocab/max_len drive the TextCNN.
    "agnews":  dict(n_classes=4, channels=None, img=None,
                    vocab_size=20000, max_len=64, embed_dim=128),
    "cifar10": dict(n_classes=10,  channels=3, img=32),
    "cifar100":dict(n_classes=100, channels=3, img=32),
}


def get_datasets(name, root, augment=True):
    """Train/test datasets. Augmentation (RandomCrop+HFlip) is applied to the
    TRAIN set only -- it is standard for CIFAR and worth ~10-15 points; without
    it ResNet-18 memorizes CIFAR-100 and caps around 0.55. MNIST/LeNet is left
    unaugmented (already saturated, and augmentation would only slow it)."""
    if name == "mnist":
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))])
        tr = datasets.MNIST(root, train=True, download=True, transform=tf)
        te = datasets.MNIST(root, train=False, download=True, transform=tf)
        return tr, te

    if name == "femnist":
        tr, te, _ = find_femnist(root)
        return tr, te

    if name == "agnews":
        m = DATASET_META["agnews"]
        return get_agnews(root, m["vocab_size"], m["max_len"])

    if name == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        ctor = datasets.CIFAR10
    elif name == "cifar100":
        mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        ctor = datasets.CIFAR100
    else:
        raise ValueError(name)

    test_tf = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean, std)])
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    else:
        train_tf = test_tf
    tr = ctor(root, train=True, download=True, transform=train_tf)
    te = ctor(root, train=False, download=True, transform=test_tf)
    return tr, te


def _targets(ds):
    t = ds.targets
    return np.array(t) if not isinstance(t, np.ndarray) else t


def dirichlet_partition(dataset, n_clients, alpha, n_classes, seed=0):
    """Return list of index arrays, one per client, via Dirichlet label skew."""
    rng = np.random.default_rng(seed)
    y = _targets(dataset)
    idx_by_class = [np.where(y == c)[0] for c in range(n_classes)]
    for c in idx_by_class:
        rng.shuffle(c)
    client_idx = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        cuts = (np.cumsum(proportions) * len(idx_by_class[c])).astype(int)[:-1]
        splits = np.split(idx_by_class[c], cuts)
        for k in range(n_clients):
            client_idx[k].extend(splits[k].tolist())
    return [np.array(sorted(ix)) for ix in client_idx]


class FEMNISTDataset(Dataset):
    """LEAF FEMNIST held in memory, retaining the writer id per sample.

    LEAF ships JSON shards shaped
      {"users":[...], "num_samples":[...],
       "user_data":{"<writer>":{"x":[[784 floats]...], "y":[int...]}}}
    The writer ids are the whole point: FEMNIST's non-IID structure is the
    natural per-writer split, not a synthetic Dirichlet draw."""

    MEAN, STD = 0.9637, 0.1591          # FEMNIST is mostly white background

    def __init__(self, json_dir, normalize=True):
        xs, ys, ws = [], [], []
        files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
        if not files:
            raise FileNotFoundError(f"no LEAF json shards in {json_dir}")
        for fp in files:
            with open(fp) as f:
                blob = json.load(f)
            for w, rec in blob["user_data"].items():
                arr = np.asarray(rec["x"], dtype=np.float32)
                xs.append(arr)
                ys.extend(int(v) for v in rec["y"])
                ws.extend([w] * len(rec["y"]))
        x = np.concatenate(xs, 0).reshape(-1, 1, 28, 28)
        if normalize:
            x = (x - self.MEAN) / self.STD
        self.x = torch.from_numpy(x)
        self.targets = np.asarray(ys, dtype=np.int64)
        self.writers = np.asarray(ws)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.x[i], int(self.targets[i])


def _emnist_byclass(root):
    """Fallback when LEAF FEMNIST isn't mounted: EMNIST/byclass has the same 62
    classes but NO writer ids, so the natural partition is unavailable and the
    caller must fall back to Dirichlet. Flagged loudly by find_femnist()."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        # EMNIST images arrive transposed relative to MNIST convention
        transforms.Lambda(lambda t: t.transpose(1, 2)),
        transforms.Normalize((0.1736,), (0.3317,))])
    tr = datasets.EMNIST(root, split="byclass", train=True, download=True,
                         transform=tf)
    te = datasets.EMNIST(root, split="byclass", train=False, download=True,
                         transform=tf)
    return tr, te


def find_femnist(root, femnist_root=None):
    """Return (train, test, natural) where natural=True iff real LEAF FEMNIST
    with writer ids was found. Searches femnist_root then common Kaggle mounts."""
    cands = []
    if femnist_root:
        cands.append(femnist_root)
    cands += sorted(glob.glob("/kaggle/input/*/femnist")) + \
             sorted(glob.glob("/kaggle/input/*femnist*")) + \
             [os.path.join(root, "femnist")]
    for base in cands:
        tr_dir = os.path.join(base, "train")
        te_dir = os.path.join(base, "test")
        if os.path.isdir(tr_dir) and os.path.isdir(te_dir):
            try:
                return (FEMNISTDataset(tr_dir), FEMNISTDataset(te_dir), True)
            except FileNotFoundError:
                continue
    print("[femnist] LEAF shards not found -> falling back to EMNIST/byclass "
          "(62 classes, but NO writer ids: partition will be Dirichlet, not "
          "the natural per-writer split).")
    tr, te = _emnist_byclass(root)
    return tr, te, False


def natural_partition(dataset, n_clients, seed=0):
    """Writer-based FEMNIST partition: assign whole writers to clients, greedily
    balancing sample counts. Keeping writers intact is what makes the split
    'natural' -- each client is a real person's handwriting."""
    writers = getattr(dataset, "writers", None)
    if writers is None:
        raise ValueError("dataset has no writer ids; use dirichlet_partition")
    rng = np.random.default_rng(seed)
    uniq = np.unique(writers)
    rng.shuffle(uniq)
    idx_by_writer = {w: np.where(writers == w)[0] for w in uniq}
    # largest-first greedy into the currently-smallest client
    order = sorted(uniq, key=lambda w: -len(idx_by_writer[w]))
    buckets = [[] for _ in range(n_clients)]
    load = np.zeros(n_clients, dtype=np.int64)
    for w in order:
        j = int(load.argmin())
        buckets[j].extend(idx_by_writer[w].tolist())
        load[j] += len(idx_by_writer[w])
    return [np.array(sorted(b)) for b in buckets]


AGNEWS_URLS = {
    "train": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/"
             "master/data/ag_news_csv/train.csv",
    "test":  "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/"
             "master/data/ag_news_csv/test.csv",
}


def _tok(s):
    """Lowercase word-level tokeniser: keep alphanumerics, split on anything else."""
    out, cur = [], []
    for ch in s.lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            out.append("".join(cur)); cur = []
    if cur:
        out.append("".join(cur))
    return out


class AGNewsDataset(Dataset):
    """AG News as fixed-length token-id sequences.

    Labels in the CSV are 1..4; we shift to 0..3. Title and description are
    concatenated then truncated/padded to `max_len`. The vocabulary is built on
    TRAIN only and passed to the test split, so there is no leakage."""

    def __init__(self, rows, vocab=None, max_len=64, vocab_size=20000):
        import collections
        self.max_len = max_len
        texts = [t for t, _ in rows]
        self.targets = np.asarray([y for _, y in rows], dtype=np.int64)
        toks = [_tok(t) for t in texts]
        if vocab is None:
            cnt = collections.Counter(w for tt in toks for w in tt)
            # 0 = <pad>, 1 = <unk>
            self.vocab = {w: i + 2 for i, (w, _) in
                          enumerate(cnt.most_common(vocab_size - 2))}
        else:
            self.vocab = vocab
        X = np.zeros((len(toks), max_len), dtype=np.int64)
        for i, tt in enumerate(toks):
            ids = [self.vocab.get(w, 1) for w in tt[:max_len]]
            X[i, :len(ids)] = ids
        self.x = torch.from_numpy(X)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.x[i], int(self.targets[i])


def _agnews_rows(path):
    import csv
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) >= 3:
                rows.append((r[1] + " " + r[2], int(r[0]) - 1))
    return rows


def get_agnews(root, vocab_size=20000, max_len=64):
    import urllib.request
    os.makedirs(root, exist_ok=True)
    paths = {}
    for split, url in AGNEWS_URLS.items():
        fp = os.path.join(root, f"agnews_{split}.csv")
        if not os.path.exists(fp):
            print(f"[agnews] downloading {split} ...")
            urllib.request.urlretrieve(url, fp)
        paths[split] = fp
    tr_rows, te_rows = _agnews_rows(paths["train"]), _agnews_rows(paths["test"])
    tr = AGNewsDataset(tr_rows, None, max_len, vocab_size)
    te = AGNewsDataset(te_rows, tr.vocab, max_len, vocab_size)
    return tr, te


def get_eval_train(name, root):
    """Train set with the TEST-time transform (no augmentation).

    Forget-set accuracy and MIA *member* features must be computed on clean
    images: if members were augmented while non-members (test) are clean, members
    would look artificially less confident and the membership comparison would be
    biased downward. Client training still uses the augmented view."""
    return get_datasets(name, root, augment=False)[0]


class BackdoorWrapper(Dataset):
    """Adds a white square trigger and relabels to target. Used for FuGuard's
    backdoor-ASR axis. `poison_all=True` triggers every sample (test ASR set)."""
    def __init__(self, base, indices, target, patch=4, poison_frac=0.5,
                 poison_all=False, seed=0):
        self.base = base
        self.indices = list(indices)
        self.target = target
        self.patch = patch
        self.poison_all = poison_all
        rng = np.random.default_rng(seed)
        if poison_all:
            self.poison = set(range(len(self.indices)))
        else:
            k = int(poison_frac * len(self.indices))
            self.poison = set(rng.choice(len(self.indices), k, replace=False).tolist())

    def __len__(self):
        return len(self.indices)

    def _trigger(self, x):
        x = x.clone()
        if x.dim() == 1:           # TEXT: overwrite the tail with a rare token id
            x[-self.patch:] = 1    # <unk>, acts as a trigger phrase
            return x
        p = self.patch
        x[:, -p:, -p:] = x.max()   # bright square in the bottom-right corner
        return x

    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        if i in self.poison:
            x = self._trigger(x)
            y = self.target
        return x, y


def make_client_loaders(dataset, partition, batch_size, backdoor_client=None,
                        attack_cfg=None, seed=0):
    loaders = []
    for k, idx in enumerate(partition):
        if backdoor_client is not None and k == backdoor_client and attack_cfg \
                and attack_cfg.backdoor:
            ds = BackdoorWrapper(dataset, idx, attack_cfg.backdoor_target,
                                 poison_frac=attack_cfg.backdoor_frac, seed=seed)
        else:
            ds = Subset(dataset, idx)
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True,
                                  num_workers=2, drop_last=False))
    return loaders


def make_backdoor_testset(test_ds, target, batch_size):
    """Every test sample gets the trigger; ASR = fraction predicted as target."""
    idx = list(range(len(test_ds)))
    ds = BackdoorWrapper(test_ds, idx, target, poison_all=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
