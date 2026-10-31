# RTCFake — drop-in slot

RTCFake is the one corpus VeriCall cannot fetch automatically. This directory
marks its place and records exactly how to fill it.

## Why it matters more than the others

It is the **only** corpus containing speech actually transmitted through real
real-time-communication platforms — WeChat, Zoom, QQ, DingTalk, Lark, VooV and
Telegram — carrying genuine noise suppression, echo cancellation, codec chains
and packet loss, rather than simulated degradation. Its eval split holds out
*unseen* platforms.

That makes it the only honest test of the design's "survives compression"
constraint. Until it is here, codec robustness is evidenced by **simulated**
G.711/Opus augmentation, which is weaker, and every claim about it is written
that way.

## Why it is not downloaded

`JunXueTech/RTCFake` is a **gated** HuggingFace repo. The file tree is public
(which is why it first appeared available), but every file returns:

```
401  Access to dataset JunXueTech/RTCFake is restricted.
     You must have access to it and be authenticated to access it.
```

Gating requires a human: an account, and approval from the dataset owner.

## How to fill it

1. Create a HuggingFace account.
2. Request access at <https://huggingface.co/datasets/JunXueTech/RTCFake> and
   wait for the owner to approve.
3. Create a read token at <https://huggingface.co/settings/tokens>.
4. Then either:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
./scripts/pod.sh py 'scripts/fetch_datasets.py --datasets rtcfake --dest /tmp/vericall-data'
```

or, if you already have the archives, drop them straight onto pod scratch at
`/tmp/vericall-data/rtcfake/` with the repo's own layout:

```
rtcfake/
  data/
    wav/train.tar.gz          12.73 GB
    wav/dev.tar.gz             2.48 GB
    wav/progress.tar.gz        5.18 GB
    train_label.txt
    meta/train_offline_online_pairs.csv
```

`scripts/fetch_datasets.py` is idempotent: it skips anything already present.

## What happens when it lands

Nothing else needs changing. `scripts/dataset_status.py` will show it, and the
manifest/split layer treats it as another corpus. The pipeline degrades
gracefully without it rather than failing — that is deliberate, not an oversight
— but the compression claim stays qualified until it is here.
