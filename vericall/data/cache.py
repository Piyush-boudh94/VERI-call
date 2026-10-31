"""Reader for the extracted training cache."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class CachedFold:
    """Memory-mapped access to one extracted fold.

    Memory-mapped rather than loaded: the cache runs to tens of GB and the OS
    page cache handles reuse across epochs better than we would.
    """

    root: Path
    wave: np.ndarray
    feat: np.ndarray
    label: np.ndarray
    codec: np.ndarray
    meta: list[dict]
    info: dict

    @classmethod
    def load(cls, root: str | Path) -> "CachedFold":
        root = Path(root)
        info = json.loads((root / "info.json").read_text())
        n = info["rows"]
        meta_path = root / "meta.jsonl"
        meta = [json.loads(l) for l in meta_path.read_text().splitlines() if l.strip()]
        return cls(
            root=root,
            wave=np.load(root / "wave.npy", mmap_mode="r")[:n],
            feat=np.load(root / "feat.npy", mmap_mode="r")[:n],
            label=np.load(root / "label.npy")[:n],
            codec=np.load(root / "codec.npy")[:n],
            meta=meta[:n],
            info=info,
        )

    def __len__(self) -> int:
        return len(self.label)

    @property
    def groups(self) -> np.ndarray:
        """Utterance id per row -- the unit that bootstrap CIs resample over.

        Resampling rows would treat several crops of one clip as independent
        observations and report an interval several times too narrow.
        """
        return np.array([m["utt_id"] for m in self.meta])

    @property
    def attacks(self) -> np.ndarray:
        return np.array([m["attack_id"] for m in self.meta])

    def batches(self, batch: int, shuffle: bool = True, seed: int = 0,
                use: str = "feat", device: torch.device | str = "cpu",
                balanced: bool = False):
        """Iterate batches.

        `balanced` is accepted and ignored: a single corpus has nothing to
        balance across. It exists so a caller can pass the same arguments to a
        CachedFold and a ConcatFold without branching -- which is exactly what
        the layer sweep does, and its absence was a TypeError that killed the
        sweep after the whole extraction had finished.
        """
        idx = np.arange(len(self))
        if shuffle:
            np.random.default_rng(seed).shuffle(idx)
        src = self.feat if use == "feat" else self.wave
        for s in range(0, len(idx), batch):
            sel = np.sort(idx[s:s + batch])          # sorted: mmap likes locality
            x = torch.from_numpy(np.asarray(src[sel], dtype=np.float32)).to(device)
            y = torch.from_numpy(self.label[sel]).to(device)
            c = torch.from_numpy(self.codec[sel]).to(device)
            yield x, y, c, sel

    def domain_ids(self, sel: np.ndarray) -> np.ndarray:
        """All one domain. Present for the same interface reason as `balanced`."""
        return np.zeros(len(sel), dtype=np.int64)


def class_weights(labels: np.ndarray) -> torch.Tensor:
    """Prior-weighted loss.

    ASVspoof LA is ~1:8.8 bona fide:spoof. Training unweighted bakes that ratio
    into the output, which then has to be undone before the score can be treated
    as a prior-free LLR. Weighting here keeps the branch prior-free and lets the
    operational prior be injected once, at decision time.
    """
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (2.0 * counts)
    return torch.tensor(w, dtype=torch.float32)


class _LazyConcat:
    """Fancy-indexes across several memmaps without materialising them.

    Mixed-corpus training needs one logical fold over two caches. Calling
    np.concatenate on the feature arrays would pull them into RAM -- ASVspoof
    train at 4 crops plus RTCFake train at 4 crops is ~12 GB of fp16 features
    against a 32 GiB pod, shared. The arrays stay memmapped and only the
    requested rows are read, which is what CachedFold already relies on.
    """

    __slots__ = ("parts", "offsets", "_shape")

    def __init__(self, parts):
        self.parts = list(parts)
        self.offsets = np.cumsum([0] + [len(p) for p in self.parts])
        tail = self.parts[0].shape[1:]
        for p in self.parts[1:]:
            if p.shape[1:] != tail:
                raise ValueError(
                    f"cannot concatenate folds with different row shapes: "
                    f"{tail} vs {p.shape[1:]}. Extraction settings differ "
                    f"(window length or frame count) -- re-extract, do not pad.")
        self._shape = (int(self.offsets[-1]), *tail)

    @property
    def shape(self):
        return self._shape

    def __len__(self) -> int:
        return self._shape[0]

    def __getitem__(self, idx):
        idx = np.asarray(idx)
        if idx.ndim == 0:
            part = int(np.searchsorted(self.offsets, idx, side="right")) - 1
            return self.parts[part][int(idx) - int(self.offsets[part])]
        out = np.empty((len(idx), *self._shape[1:]), dtype=self.parts[0].dtype)
        part_of = np.searchsorted(self.offsets, idx, side="right") - 1
        for p in np.unique(part_of):
            sel = np.flatnonzero(part_of == p)
            local = idx[sel] - self.offsets[p]
            # Read in ascending order -- mmap hates random access -- but write
            # back to the caller's slots. Sorting the reads WITHOUT carrying the
            # permutation through to the destination silently pairs row i's
            # features with row j's label, which trains fine and just performs
            # worse. tests/test_concat_fold.py pins this.
            order = np.argsort(local, kind="stable")
            out[sel[order]] = self.parts[p][local[order]]
        return out


@dataclass
class ConcatFold:
    """Several CachedFolds presented as one, for mixed-corpus training.

    Duck-types CachedFold rather than subclassing it: the consumers only ever
    touch label / codec / meta / feat / wave / groups / batches, and keeping the
    surface explicit makes it obvious what a new consumer may rely on.

    `sources` is the addition -- it records which corpus each row came from, so
    a mixed run can report per-corpus metrics instead of one pooled number that
    hides which half of the data the model is actually fitting.
    """

    folds: list
    names: list

    label: np.ndarray = None
    codec: np.ndarray = None
    meta: list = None
    sources: np.ndarray = None
    feat: object = None
    wave: object = None
    info: dict = None

    def __post_init__(self):
        if not self.folds:
            raise ValueError("ConcatFold needs at least one fold")
        self.label = np.concatenate([f.label for f in self.folds])
        self.codec = np.concatenate([f.codec for f in self.folds])
        self.meta = [m for f in self.folds for m in f.meta]
        self.sources = np.concatenate([
            np.full(len(f), n, dtype=object)
            for f, n in zip(self.folds, self.names)])
        self.feat = _LazyConcat([f.feat for f in self.folds])
        self.wave = _LazyConcat([f.wave for f in self.folds])
        self.info = {"concat": list(self.names),
                     "rows": [len(f) for f in self.folds],
                     "window_s": self.folds[0].info.get("window_s"),
                     "channels": sorted({c for f in self.folds
                                         for c in f.info.get("channels", [])})}

    def __len__(self) -> int:
        return len(self.label)

    @property
    def groups(self) -> np.ndarray:
        """Corpus-qualified utterance ids.

        Two corpora can and do reuse an utterance id string. Unqualified, a
        collision would make the bootstrap resample two unrelated clips as one
        unit -- silently, and in the direction of a narrower interval.
        """
        return np.array([f"{s}:{m['utt_id']}"
                         for s, m in zip(self.sources, self.meta)])

    @property
    def attacks(self) -> np.ndarray:
        return np.array([m["attack_id"] for m in self.meta])

    def batches(self, batch: int, shuffle: bool = True, seed: int = 0,
                use: str = "feat", device: torch.device | str = "cpu",
                balanced: bool = False):
        """Iterate batches. `balanced` draws equally from each source corpus.

        Without it, a mixed fold is dominated by whichever corpus contributed
        most rows -- ASVspoof is 64% of the ASVspoof+RTCFake mixture -- and
        "which corpus is this" becomes both the cheapest available cue and one
        the majority corpus is happy to supply. docs/RESULTS.md section 16
        measured what that costs: WaveFake 30.08% -> 49.89%, destroyed rather
        than diluted.

        Sampling with replacement from the smaller corpus rather than
        truncating the larger one: truncation throws away real data to fix a
        ratio, which is the expensive way to do it.
        """
        src = self.feat if use == "feat" else self.wave
        rng = np.random.default_rng(seed)

        if balanced:
            groups = [np.flatnonzero(self.sources == n) for n in self.names]
            groups = [g for g in groups if len(g)]
            per = max(1, batch // max(len(groups), 1))
            # One epoch = enough batches to cover the LARGEST corpus once, so
            # no corpus is under-seen just because another is small.
            n_batches = max(1, max(len(g) for g in groups) // per)
            for _ in range(n_batches):
                sel = np.concatenate([rng.choice(g, size=per, replace=len(g) < per)
                                      for g in groups])
                sel = np.sort(sel)
                x = torch.from_numpy(np.asarray(src[sel], dtype=np.float32)).to(device)
                y = torch.from_numpy(self.label[sel]).to(device)
                c = torch.from_numpy(self.codec[sel]).to(device)
                yield x, y, c, sel
            return

        idx = np.arange(len(self))
        if shuffle:
            rng.shuffle(idx)
        for s in range(0, len(idx), batch):
            sel = np.sort(idx[s:s + batch])
            x = torch.from_numpy(np.asarray(src[sel], dtype=np.float32)).to(device)
            y = torch.from_numpy(self.label[sel]).to(device)
            c = torch.from_numpy(self.codec[sel]).to(device)
            yield x, y, c, sel

    def domain_ids(self, sel: np.ndarray) -> np.ndarray:
        """Corpus index per selected row, for the domain adversary."""
        lut = {n: i for i, n in enumerate(self.names)}
        return np.array([lut[s] for s in self.sources[sel]], dtype=np.int64)
