#!/usr/bin/env python
"""Run VeriCall on a local machine. One command, no Kubernetes.

    python run_local.py              # check everything, then serve
    python run_local.py --check      # check only, change nothing
    python run_local.py --port 8080

Everything in this project was developed against a Kubernetes pod with an
18 GB MIG slice, and every helper script went through `scripts/pod.sh`. None of
that is needed to RUN it: the library itself has no absolute paths and no
cluster assumptions -- only the training and staging scripts do. This is the
entry point that makes that true in practice rather than in principle.

**On a 6 GB consumer GPU it fits with room to spare.** Measured
(scripts/vram_budget.py, docs/vram_budget.json): all four branches are 0.44 GiB
of weights, and a batch of 32 peaks at 0.96 GiB. The constraint on a laptop GPU
is compute, not memory, so the knob that matters is concurrency rather than
batch size.

The checks below run BEFORE the server starts, because every one of them
otherwise fails later as something less legible: a missing checkpoint becomes an
untrained head that scores at chance, a missing calibration becomes confident
numbers on the wrong scale, and a CPU-only torch becomes a demo that cannot keep
up with real time.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OK, WARN, BAD = "  ok  ", " warn ", " FAIL "


class Check:
    def __init__(self) -> None:
        self.failed = 0
        self.warned = 0

    def ok(self, what: str, detail: str = "") -> None:
        print(f"[{OK}] {what}" + (f" — {detail}" if detail else ""))

    def warn(self, what: str, detail: str) -> None:
        self.warned += 1
        print(f"[{WARN}] {what} — {detail}")

    def bad(self, what: str, detail: str) -> None:
        self.failed += 1
        print(f"[{BAD}] {what} — {detail}")


def check_python(c: Check) -> None:
    v = sys.version_info
    if v < (3, 9):
        c.bad("python", f"{v.major}.{v.minor} is too old; 3.9+ required")
    else:
        c.ok("python", f"{v.major}.{v.minor}.{v.micro}")


def pinned_version(pkg: str) -> str | None:
    """The version requirements.txt pins for `pkg`, or None if unpinned.

    Read rather than duplicated, so there is exactly one source of truth for
    "the version these checkpoints were validated against".
    """
    req = ROOT / "requirements.txt"
    if not req.exists():
        return None
    for line in req.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith(f"{pkg}==") and "[" not in line:
            return line.split("==", 1)[1].strip()
    return None


def check_torch(c: Check) -> "object | None":
    try:
        import torch
    except ImportError:
        c.bad("torch", "not installed — run: pip install -r requirements.txt")
        return None

    # Compare against the pin, because the two had silently drifted: the file
    # pinned 2.4.0 and claimed the checkpoints were calibrated against it,
    # while the environment that produced every number in docs/RESULTS.md ran
    # 2.13.0. A mismatch is a warning and not a failure -- checkpoints are
    # state_dicts of tensors and usually load across versions -- but it is the
    # first thing to suspect if local scores disagree with the documented ones.
    # Two torches are validated, on different halves of the project: 2.4.0
    # trained every checkpoint (in the GPU pod) and 2.13.0 served and evaluated
    # them (on the head node). Neither is wrong, so neither should warn. An
    # earlier version compared against a single pin and would have flagged the
    # environment that actually produced the weights.
    want = pinned_version("torch")
    validated = {v for v in (want, "2.4.0", "2.13.0") if v}
    running = torch.__version__.split("+")[0]
    if running in validated:
        c.ok("torch", f"{torch.__version__} (validated)")
    else:
        c.warn("torch", f"{torch.__version__} running; validated versions are "
                        f"{', '.join(sorted(validated))} — checkpoints load "
                        f"across versions, but scores may not reproduce "
                        f"docs/RESULTS.md, so suspect this first if they differ")

    if not torch.cuda.is_available():
        # Not fatal: it will run, just not in real time. Saying so here is the
        # difference between "slow demo" and "mysterious demo".
        c.warn("cuda", "not available — the gateway will run on CPU and will "
                       "NOT keep up with real-time audio. Install a CUDA build "
                       "of torch for your driver.")
        return torch

    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    c.ok("cuda", f"{name}, {total:.1f} GiB")
    # 0.44 GiB of weights + ~0.5 GiB of activations at batch 32, measured.
    if total < 2.0:
        c.bad("vram", f"{total:.1f} GiB is below the ~1.5 GiB this needs")
    elif total < 4.0:
        c.warn("vram", f"{total:.1f} GiB is tight; set VERICALL_MAX_CALLS lower")
    return torch


def check_artifacts(c: Check) -> None:
    need_ckpt = ["branch_A_seed0.pt", "branch_A2_seed0.pt", "branch_B.pt"]
    missing = [f for f in need_ckpt if not (ROOT / "checkpoints" / f).exists()]
    if missing:
        c.bad("checkpoints", f"missing {missing} — without these the heads are "
                             f"untrained and score at chance")
    else:
        c.ok("checkpoints", f"{len(need_ckpt)} branch weights present")

    calib = ROOT / "calib"
    need_cal = ["A@clean.json", "A2@clean.json", "subsets.json"]
    miss_cal = [f for f in need_cal if not (calib / f).exists()]
    if miss_cal:
        c.bad("calibration", f"missing {miss_cal} — branch outputs would be raw "
                             f"logits, not LLRs, and every threshold assumes LLRs")
    else:
        c.ok("calibration", f"{len(list(calib.glob('*.json')))} files")

    if (calib / "weights.json").exists():
        c.ok("fusion weights", "fitted")
    else:
        c.warn("fusion weights", "not fitted; falling back to the hand-set "
                                 "defaults (quote 3.04% EER, not 2.79%)")

    clips = sorted((ROOT / "web" / "demo").glob("*.wav"))
    if len(clips) >= 3:
        c.ok("demo clips", f"{len(clips)} present")
    else:
        c.warn("demo clips", "missing; the live-microphone path still works")


# ECAPA is vendored in-tree under .cache/ecapa. Each entry is
# (filename, minimum plausible size in bytes) -- the size floor is what
# distinguishes a real checkpoint from a truncated download or a 130-byte
# symlink path, and it is the cheapest check that would have caught the bug
# described in check_pretrained below.
ECAPA_FILES = (
    ("embedding_model.ckpt", 40_000_000),
    ("classifier.ckpt", 2_000_000),
    ("hyperparams.yaml", 500),
    ("label_encoder.ckpt", 10_000),
    ("mean_var_norm_emb.ckpt", 500),
)


def check_pretrained(c: Check, torch_mod) -> None:
    """WavLM downloads on first use; ECAPA is vendored in-tree.

    Worth a check rather than a surprise: the first run needs network access,
    and on a machine that has none the failure appears deep inside model
    construction as an HTTP error.

    This check used to be `ecapa.exists() and any(ecapa.iterdir())`, which
    tests the DIRECTORY, not the files in it. The five vendored files were
    committed as symlinks into the developer's own ~/.cache/huggingface, and
    git stores a symlink as its target TEXT -- so every fresh clone on any
    other machine got five dangling links to a path that did not exist there.
    The directory still existed and iterdir() still yielded five entries, so
    the check printed "cached locally" and SpeechBrain then failed deep inside
    model construction: exactly the failure this function exists to prevent,
    produced by the function itself. Each file is now stat'd individually with
    a size floor, and a dangling link fails is_file().
    """
    hub = Path(os.environ.get("TORCH_HOME") or (Path.home() / ".cache" / "torch"))
    wavlm = list(hub.rglob("*wavlm*")) if hub.exists() else []
    if wavlm:
        # Note this lives in $HOME, not in the tree, so it does NOT travel
        # with a clone. An "ok" here describes this machine only.
        c.ok("WavLM weights", f"cached in {hub}")
    else:
        c.warn("WavLM weights", "not cached — the FIRST run downloads ~360 MB. "
                                "Needs internet once, then works offline.")

    ecapa = ROOT / ".cache" / "ecapa"
    bad: list[str] = []
    for name, floor in ECAPA_FILES:
        f = ecapa / name
        if f.is_symlink() and not f.exists():
            bad.append(f"{name}: dangling symlink -> {os.readlink(f)}")
        elif not f.is_file():
            bad.append(f"{name}: missing")
        elif f.stat().st_size < floor:
            bad.append(f"{name}: {f.stat().st_size} B, expected >= {floor}")
    if not bad:
        total = sum((ecapa / n).stat().st_size for n, _ in ECAPA_FILES)
        c.ok("ECAPA weights", f"vendored in-tree, {total / 1e6:.0f} MB")
    else:
        c.warn("ECAPA weights", "unusable (" + "; ".join(bad) + ") — Branch C "
                                "will be disabled and the other three still "
                                "work; first run re-downloads ~80 MB")


def check_deps(c: Check) -> None:
    """Report only what is actually true.

    An earlier version printed "python deps ok" unconditionally after the loop,
    so a missing FastAPI produced a FAIL line and an ok line together. A check
    that reports success alongside its own failure trains you to skim it.
    """
    required = {"fastapi": "the gateway", "uvicorn": "the server",
                "websockets": "audio streaming",
                "soundfile": "reading wav files", "numpy": "everything"}
    optional = {"speechbrain": "Branch C (identity)"}
    missing = []
    for mod, why in required.items():
        try:
            __import__(mod)
        except ImportError:
            c.bad(mod, f"missing — required for {why}")
            missing.append(mod)
    for mod, why in optional.items():
        try:
            __import__(mod)
        except ImportError:
            c.warn(mod, f"missing — {why} will be disabled")
    if not missing:
        c.ok("python deps", f"{len(required)} required packages present")


def ensure_tls(cert: str | None, key: str | None):
    """Return (cert, key) paths, generating a self-signed pair if needed.

    **Why this is here at all.** Browsers expose `getUserMedia` only on a
    secure origin, and `localhost` is the one http exception. So the live
    microphone and the whole of /call work over plain http on this machine and
    silently do nothing from any other -- the page loads, the button does
    nothing, and there is no error worth reading. Two tabs on one laptop need
    no TLS; a laptop and a phone do.

    **What a self-signed certificate is and is not.** It encrypts the link and
    it authenticates nothing: every browser will show a warning that has to be
    clicked through, and anyone who can intercept the connection can present
    their own. That is acceptable for a demo on a network you control and is
    not a deployment posture. A real deployment terminates TLS at a reverse
    proxy with a certificate someone actually trusts.
    """
    if cert and key:
        if Path(cert).is_file() and Path(key).is_file():
            return Path(cert), Path(key)
        print(f"[{BAD}] tls — certificate or key not found at the given paths")
        return None, None

    out = ROOT / ".tls"
    cert_p, key_p = out / "dev-cert.pem", out / "dev-key.pem"
    if cert_p.is_file() and key_p.is_file():
        print(f"[{OK}] tls — reusing the self-signed certificate in .tls/")
        return cert_p, key_p

    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        print(f"[{BAD}] tls — needs `pip install cryptography`, or pass "
              f"--cert/--key for a certificate you already have")
        return None, None

    import datetime
    import ipaddress
    import socket

    out.mkdir(exist_ok=True)
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    # SANs for every name this box answers to, because a certificate valid for
    # "localhost" alone is useless for the case that needs TLS at all -- the
    # other machine reaches this one by LAN IP.
    names = [x509.DNSName("localhost")]
    addrs = {"127.0.0.1"}
    try:
        addrs.add(socket.gethostbyname(socket.gethostname()))
    except OSError:
        pass
    for a in sorted(addrs):
        try:
            names.append(x509.IPAddress(ipaddress.ip_address(a)))
        except ValueError:
            pass
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "vericall-local")])
    now = datetime.datetime.now(datetime.timezone.utc)
    crt = (x509.CertificateBuilder()
           .subject_name(subject).issuer_name(subject)
           .public_key(k.public_key())
           .serial_number(x509.random_serial_number())
           .not_valid_before(now - datetime.timedelta(minutes=5))
           .not_valid_after(now + datetime.timedelta(days=365))
           .add_extension(x509.SubjectAlternativeName(names), critical=False)
           .sign(k, hashes.SHA256()))
    cert_p.write_bytes(crt.public_bytes(serialization.Encoding.PEM))
    key_p.write_bytes(k.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()))
    key_p.chmod(0o600)
    print(f"[{OK}] tls — generated a self-signed certificate in .tls/ "
          f"(valid 365 days, SANs: {', '.join(str(n.value) for n in names)})")
    print("       Browsers will warn that it is not trusted. That is expected: "
          "it encrypts\n       the link and authenticates nothing. Click "
          "through on a network you control.")
    return cert_p, key_p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="check only")
    ap.add_argument("--profile", default=os.environ.get("VERICALL_PROFILE"),
                    help="deployment profile: heads + the calibration fitted "
                         "for them, selected together. --list-profiles shows "
                         "them. Defaults to the one named in profiles.json.")
    ap.add_argument("--list-profiles", action="store_true",
                    help="show the profiles and exit")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("VERICALL_PORT", 8001)))
    ap.add_argument("--host", default=os.environ.get("VERICALL_HOST", "127.0.0.1"))
    ap.add_argument("--max-calls", type=int, default=None,
                    help="concurrency cap. A laptop GPU is compute-bound, so "
                         "this is the knob that matters, not batch size.")
    ap.add_argument("--tls", action="store_true",
                    help="serve HTTPS with a self-signed certificate, "
                         "generating one on first use. REQUIRED to use the "
                         "microphone or /call from any machine other than "
                         "this one: browsers only expose getUserMedia on a "
                         "secure origin, and localhost is the sole http "
                         "exception.")
    ap.add_argument("--cert", default=os.environ.get("VERICALL_TLS_CERT"),
                    help="TLS certificate (implies --tls)")
    ap.add_argument("--key", default=os.environ.get("VERICALL_TLS_KEY"),
                    help="TLS private key (implies --tls)")
    args = ap.parse_args()

    from vericall import profiles as _profiles
    if args.list_profiles:
        print(_profiles.describe())
        return 0

    # Before any check runs: the checks read the calibration directory and the
    # checkpoint names, so they must see what the profile selected, not the
    # defaults it is about to replace.
    chosen = args.profile or _profiles.default_id()
    if chosen:
        try:
            applied = _profiles.apply_to_env(chosen)
        except (KeyError, RuntimeError) as exc:
            print(f"profile error: {exc}\n")
            print(_profiles.describe())
            return 1
        prof = _profiles.get(chosen)
        kept = {k: os.environ[k] for k in
                ("VERICALL_CALIB_DIR", "VERICALL_CKPT_A", "VERICALL_CKPT_A2")
                if k in os.environ and k not in applied}
        print(f"profile: {chosen} — {prof.display}")
        print(f"         {', '.join(prof.languages)} | calib {prof.calib_dir}")
        if kept:
            # An explicit variable beating the profile is the exact shape of
            # the mismatch that served an untrained head, so it is said out
            # loud rather than merged silently.
            for k, v in kept.items():
                print(f"         NOTE {k}={v} from the environment overrides "
                      f"this profile")
        print()

    print("VeriCall — local startup check\n")
    c = Check()
    check_python(c)
    check_deps(c)
    torch_mod = check_torch(c)
    check_artifacts(c)
    if torch_mod is not None:
        check_pretrained(c, torch_mod)

    print()
    if c.failed:
        print(f"{c.failed} blocking problem(s). Fix those and re-run; the "
              f"server is not started.")
        return 1
    if c.warned:
        print(f"{c.warned} warning(s) — the server will start, and the "
              f"warnings above say what is degraded.")
    if args.check:
        print("\n--check given; not starting the server.")
        return 0

    env = dict(os.environ)
    if args.max_calls:
        env["VERICALL_MAX_CALLS"] = str(args.max_calls)
    # 127.0.0.1 by default: a detector with no auth token bound to 0.0.0.0 is
    # reachable by anything on the network, and the laptop default should not
    # be the insecure one.
    if args.host == "0.0.0.0" and not env.get("VERICALL_AUTH_TOKEN"):
        print("\nNOTE: binding 0.0.0.0 with no VERICALL_AUTH_TOKEN set means "
              "anyone on your network can\n      connect. Set a token, or bind "
              "127.0.0.1 (the default).")

    cert = key = None
    if args.tls or args.cert or args.key:
        cert, key = ensure_tls(args.cert, args.key)
        if cert is None:
            print("\nTLS was requested and could not be set up; not starting.")
            return 1

    scheme = "https" if cert else "http"
    if scheme == "http" and args.host not in ("127.0.0.1", "localhost"):
        # The failure this prevents is confusing rather than loud: the page
        # loads fine and the microphone button silently does nothing, because
        # getUserMedia is undefined on an insecure origin.
        print("\nNOTE: serving http:// on a non-loopback address. Browsers "
              "block microphone\n      access off a secure origin, so /call "
              "and the live-mic demo will NOT work\n      from another "
              "machine. Add --tls.")

    print(f"\nstarting gateway on {scheme}://{args.host}:{args.port}")
    print("  dashboard:  open that URL in a browser")
    print("  health:     /health     metrics: /metrics/prom")
    print("  history:    /api/calls")
    print("  stop:       Ctrl-C\n")
    cmd = [sys.executable, "-m", "uvicorn", "vericall.serving.app:app",
           "--host", args.host, "--port", str(args.port)]
    if cert:
        cmd += ["--ssl-certfile", str(cert), "--ssl-keyfile", str(key)]
    try:
        return subprocess.call(cmd, cwd=str(ROOT), env=env)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
