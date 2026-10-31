#!/usr/bin/env bash
# Verify this repo is actually exportable: clone it somewhere else, boot it,
# and stream real audio through it.
#
# Why this exists as a script rather than a one-time check. Four separate
# defects were invisible from inside the working tree and all four would have
# reached the target laptop:
#
#   1. .gitignore's blanket *.npy dropped both calibration cohorts.
#   2. The five "vendored" ECAPA files were symlinks into the developer's own
#      ~/.cache/huggingface. Git stores a symlink as its target TEXT, so every
#      clone elsewhere got dangling links -- and the preflight reported them
#      as "cached locally", because it tested the DIRECTORY rather than the
#      files in it.
#   3. requirements.txt pinned a torch nine releases from the one every
#      number in docs/RESULTS.md was measured on.
#   4. The calibration integrity check covered *.json only, so the two .npy
#      files were sealed in MANIFEST.sha256 and never verified.
#
# Only 3 is visible by reading. The rest need a real clone, because the bug
# IS the difference between the tree and the clone.
#
#   scripts/verify_export.sh [port]
#
# Exits nonzero on the first failure. Leaves nothing behind.
set -euo pipefail

PORT="${1:-8799}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'kill $(cat "$TMP/pid" 2>/dev/null) 2>/dev/null || true; rm -rf "$TMP"' EXIT

say() { printf "\n\033[1m== %s\033[0m\n" "$*"; }
fail() { printf "\033[31mFAIL: %s\033[0m\n" "$*"; exit 1; }

say "cloning $SRC"
git clone -q "$SRC" "$TMP/tree"
cd "$TMP/tree"

say "no symlinks may escape the tree"
# A tracked symlink is mode 120000. Any absolute target is machine-specific
# and will dangle on every other computer.
while read -r mode _ _ path; do
    [ "$mode" = "120000" ] || continue
    tgt="$(git cat-file -p "HEAD:$path")"
    case "$tgt" in /*) fail "$path -> $tgt (absolute, will dangle)";; esac
done < <(git ls-files -s)
echo "  ok -- $(git ls-files -s | awk '$1=="120000"' | wc -l) tracked symlinks"

say "runtime artifacts the server loads at startup"
for f in calib/A@pooled.json calib/weights.json calib/subsets.json \
         calib/B_bonafide_cohort.npy calib/C_cohort.npy \
         calib/MANIFEST.sha256 checkpoints/MANIFEST.sha256 \
         checkpoints/branch_A_seed0.pt checkpoints/branch_A2_seed0.pt \
         checkpoints/branch_B.pt web/index.html web/call.html \
         calib_hindi_ch/A@clean.json calib_hindi_ch/A@g711.json \
         calib_hindi_ch/A@noisy_line.json calib_hindi_ch/subsets.json \
         calib_hindi_ch/MANIFEST.sha256 \
         checkpoints/branch_Ahindich_seed0.pt \
         checkpoints/branch_A2hindich_seed0.pt \
         web/demo/hindi_genuine.wav web/demo/hindi_clone_vc.wav \
         web/demo/hindi_clone_tts.wav; do
    [ -f "$f" ] || fail "missing from the clone: $f"
    [ -s "$f" ] || fail "empty in the clone: $f"
done
echo "  ok -- all present and non-empty"

say "vendored ECAPA must be real files, not links"
for f in .cache/ecapa/*; do
    [ -L "$f" ] && fail "$f is a symlink"
    [ -r "$f" ] || fail "$f unreadable"
done
sz=$(stat -c%s .cache/ecapa/embedding_model.ckpt)
[ "$sz" -gt 40000000 ] || fail "embedding_model.ckpt is $sz bytes, expected >40 MB"
echo "  ok -- $(du -sh .cache/ecapa | cut -f1) of real weights"

say "preflight"
python run_local.py --check || fail "preflight reported a blocking problem"

say "boot on port $PORT"
python -m uvicorn vericall.serving.app:app --host 127.0.0.1 --port "$PORT" \
    > "$TMP/serve.log" 2>&1 &
echo $! > "$TMP/pid"
for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 2
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null \
    || { tail -30 "$TMP/serve.log"; fail "server never became healthy"; }

python - "$PORT" <<'PY'
import json, os, sys, urllib.request

# /health is graduated: with VERICALL_AUTH_TOKEN set it answers liveness only
# unless the token is presented, and the keys read below would KeyError. The
# gate does not set a token itself, but it can be pointed at an instance that
# has one, so it forwards whatever is in the environment.
url = f"http://127.0.0.1:{sys.argv[1]}/health"
req = urllib.request.Request(url)
if (tok := os.environ.get("VERICALL_AUTH_TOKEN", "").strip()):
    req.add_header("x-vericall-token", tok)
h = json.load(urllib.request.urlopen(req))
if "branches_loaded" not in h:
    sys.exit("FAIL: /health returned the liveness payload -- "
             "VERICALL_AUTH_TOKEN is set to something this gate cannot match")
need = {"A", "A2", "B", "C"}
got = set(h["branches_loaded"])
if need - got:
    sys.exit(f"FAIL: branches not loaded: {sorted(need - got)}")
if not h["calibrated"]:
    sys.exit("FAIL: server started uncalibrated")
for kind, rep in h["integrity"].items():
    if not rep["clean"]:
        sys.exit(f"FAIL: {kind} integrity unclean: {rep}")
print(f"  ok -- branches {sorted(got)}, calibrated, integrity clean "
      f"({h['integrity']['calibration']['checked']} calibration files)")
PY

say "stream the demo clips and check the verdicts"
python - "$PORT" <<'PY'
import asyncio, json, sys
import numpy as np, soundfile as sf, websockets

PORT = sys.argv[1]
# genuine must not alert; both attacks must reach the top tier.
WANT = {"genuine": 0, "spoofed": 3, "midcall_swap": 3}

async def run(name, want):
    wav, sr = sf.read(f"web/demo/{name}.wav", dtype="int16", always_2d=True)
    wav, n, last = wav[:, 0], int(16000 * 0.32), None
    assert sr == 16000, sr
    async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws/verify-{name}",
                                  max_size=2**20) as ws:
        async def rx():
            nonlocal last
            try:
                async for m in ws:
                    d = json.loads(m)
                    if "error" in d:
                        raise SystemExit(f"FAIL: server error on {name}: {d}")
                    last = d
            except websockets.ConnectionClosed:
                pass
        t = asyncio.create_task(rx())
        for i in range(0, len(wav) - n, n):
            await ws.send(wav[i:i + n].tobytes())
            await asyncio.sleep(0.32)
        await asyncio.sleep(1.2)
        await ws.close()
        await t
    if last is None:
        raise SystemExit(f"FAIL: no updates for {name}")
    tier, conf = last["tier"], last["confidence"]
    if tier != want:
        raise SystemExit(f"FAIL: {name} reached tier {tier}, expected {want}")
    if last.get("dropped_chunks"):
        raise SystemExit(f"FAIL: {name} dropped {last['dropped_chunks']} chunks")
    print(f"  ok -- {name:14s} tier {tier}  conf {conf['score']:.3f} "
          f"({conf['level']})  {conf['audio_scored_s']:.1f}s scored")

for name, want in WANT.items():
    asyncio.run(run(name, want))
PY

say "call history must have recorded all three"
python - "$PORT" <<'PY'
import json, sys, urllib.request
d = json.load(urllib.request.urlopen(
    f"http://127.0.0.1:{sys.argv[1]}/api/calls"))
ids = {c["call_id"] for c in d["calls"]}
for name in ("genuine", "spoofed", "midcall_swap"):
    if f"verify-{name}" not in ids:
        sys.exit(f"FAIL: verify-{name} not in call history")
for c in d["calls"]:
    if c["max_risk"] is None:
        sys.exit(f"FAIL: {c['call_id']} recorded no peak risk")
print(f"  ok -- {d['stats']['calls']} calls, {d['stats']['events']} events "
      f"persisted to sqlite")
PY

say "the HINDI profile -- the deployment target, so it is inside the gate"
# Restart on the next port with the channel-robust Hindi profile: the
# Hindi-trained heads plus clean/g711/noisy_line buckets. The assertions live
# in scripts/verify_hindi.py because shell heredocs do not nest and this
# script already has two.
kill "$(cat "$TMP/pid")" 2>/dev/null || true
sleep 2
HPORT=$((PORT + 1))
VERICALL_CALIB_DIR=calib_hindi_ch \
VERICALL_CKPT_A=branch_Ahindich_seed0.pt \
VERICALL_CKPT_A2=branch_A2hindich_seed0.pt \
    python -m uvicorn vericall.serving.app:app \
        --host 127.0.0.1 --port "$HPORT" > "$TMP/hindi.log" 2>&1 &
echo $! > "$TMP/pid"
for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$HPORT/health" >/dev/null 2>&1 && break
    sleep 2
done
curl -sf "http://127.0.0.1:$HPORT/health" >/dev/null \
    || { tail -30 "$TMP/hindi.log"; fail "hindi gateway never became healthy"; }
python scripts/verify_hindi.py "$HPORT" || fail "hindi profile verification failed"

printf "\n\033[32mEXPORT VERIFIED\033[0m -- a fresh clone of this repo boots, "
printf "loads all four branches,\nserves calibrated verdicts on all three demo "
printf "clips, and persists them.\n"
printf "                  The Hindi profile boots with every channel bucket and\n"
printf "                  separates real Hindi speech from both clone families.\n\n"
