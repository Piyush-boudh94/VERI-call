"""VeriCall serving gateway: WebSocket audio in, live calibrated risk out.

This replaces a mock that returned `random.uniform(-2.0, 2.0)` per branch. That
mock is the single most dangerous state this project can be in for a demo: the
dashboard animates, the gauge climbs, and it means nothing. If a component is
not ready the UI must say so -- see the `calibrated` flag in /health and in the
first frame of every call.

Architecture note. There is no Kafka here and that is deliberate. A broker costs
~2 cores and 2-4 GiB of a 16-core / 32 GiB quota to provide *durability*, which
is anti-valuable for audio: a three-second-old chunk is worthless to a real-time
detector. What the design does borrow is the part that applies -- cross-call
batching on a shared tick grid, so N concurrent calls present N windows to the
GPU at the same instant instead of dribbling in unsynchronised.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

# Thread oversubscription, and it is not a micro-optimisation. torch defaults to
# an intra-op pool sized from the host's core count -- measured at 112 threads
# on this 16-core pod -- so every CPU-side op inside a forward fans out across
# more threads than there are cores. That is harmless for one request at a time,
# which is why every offline script here already sets it to 1 and the serving
# path never did. Under concurrency it is the opposite of harmless: N in-flight
# forwards each try to claim the whole machine, and the resulting context
# switching inflates a 9 ms forward into hundreds of milliseconds.
#
# Serving gets its parallelism from concurrent requests, not from splitting one
# small forward across cores, so 1 is the right number. Override only if you
# have measured otherwise.
torch.set_num_threads(int(os.environ.get("VERICALL_TORCH_THREADS", "1") or 1))
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from ..fusion.calibration import load_calibration
from ..prevent.audit import AuditLog, build_record
from ..data.audio_io import load_audio
from ..store.db import CallStore
from ..store.enroll import (
    EnrollmentStore, valid_speaker_id,
)
from ..prevent.challenge import ChallengeKind, issue, risk_adjustment, verify
from ..prevent.policy import Action, CONTEXTS, PreventionPolicy
from ..fusion.config import (
    FusionConfig, TierPolicy, load_subset_calib, load_weights,
)
from .batcher import BatcherOverloaded, CrossCallBatcher
from .integrity import enforcement_required, verify as verify_integrity
from .signaling import (
    CLOSE_BAD_ROOM, CLOSE_ROOM_FULL, CLOSE_TOO_MANY_ROOMS,
    CLOSE_UNAUTHORIZED, SignalingHub, validate_signal,
)
from .limits import (
    Admission, Limits, auth_token, check_auth, resolve_channel, valid_call_id,
)
from .scorer import BranchCalibrations, BranchScorer
from ..audio.codec_detect import CodecDetector
from .session import CallSession, SessionConfig

log = logging.getLogger("vericall.serving")
ROOT = Path(__file__).resolve().parents[2]

# Calibration directory and per-branch checkpoints are configurable, because a
# deployment is not a corpus. docs/RESULTS.md section 17: every fitted component
# -- branch heads, Platt mappings, subset tables -- carries the domain it was
# fitted on, and refitting calibration is cheap (kB of JSON, seconds, no GPU)
# where retraining is not. Serving a different corpus's heads used to require
# editing this file.
#
# The pairing is load-bearing: a head must be served with the calibration fitted
# FOR IT. Applying one head's Platt mapping to another head's scores produces
# confident nonsense, not a degraded number, and nothing downstream can tell
# from the numbers alone. fit_subsets.py writes a provenance.json naming the
# heads it fitted for; the loader compares it against what it actually loaded
# and reports the result in /health. A warning rather than a refusal, because
# a legitimate mismatch exists (a different seed of the same model) and failing
# a live call over it would be worse than flagging it.
# A profile names the heads and the calibration together, so the two cannot
# drift apart. Applied here as well as in run_local.py, because the gateway is
# also started directly with uvicorn and the variables have to be in place
# before this module reads them. It never overrides an explicit variable.
ACTIVE_PROFILE: Optional[str] = None
_PROFILE_ERROR: Optional[str] = None
if (_want := os.environ.get("VERICALL_PROFILE")):
    try:
        from ..profiles import apply_to_env as _apply
        _apply(_want)
        ACTIVE_PROFILE = _want
    except Exception as _exc:          # unknown or unservable: say so, do not guess
        _PROFILE_ERROR = f"{type(_exc).__name__}: {_exc}"

CALIB = Path(os.environ.get("VERICALL_CALIB_DIR") or (ROOT / "calib"))
if not CALIB.is_absolute():
    CALIB = ROOT / CALIB

# ICE servers handed to the browser at join time.
#
# The default is a STUN server only, and the consequence is worth stating
# plainly rather than discovering during a demo: STUN lets two peers find each
# other when at least one side's NAT is permissive, which covers two tabs on
# one laptop (host candidates, no STUN needed at all) and most home networks.
# It does NOT cover symmetric NAT or restrictive corporate firewalls -- those
# need a TURN relay, which cannot be run for free at any useful bandwidth
# because it carries the media rather than just the handshake.
#
# So: the local demo this is built for works with no external dependency, a
# call across two ordinary home networks usually works, and a call from behind
# a university or corporate NAT may not connect at all. Set
# VERICALL_ICE_SERVERS to a JSON array to point at a TURN server when one
# exists.
def _ice_servers() -> list[dict]:
    raw = os.environ.get("VERICALL_ICE_SERVERS")
    if raw:
        try:
            got = json.loads(raw)
            if isinstance(got, list):
                return got
            log.warning("VERICALL_ICE_SERVERS is not a JSON array; ignoring")
        except ValueError:
            log.warning("VERICALL_ICE_SERVERS is not valid JSON; ignoring")
    return [{"urls": ["stun:stun.l.google.com:19302"]}]


# Substrings that mark a note as a real loss of capability rather than an
# observation. VERICALL_STRICT=1 refuses to start when any of them appears.
#
# Why a marker list and not a flag at each of the nineteen notes.append sites:
# the list is auditable in one place, and test_strict_markers_all_match_a_real_
# note asserts every marker still occurs in this module's source -- so if a
# note is reworded, the test fails rather than the marker silently going dead.
# Per-site flags would have no such check.
#
# The gateway's default is to degrade and say so, which is right for a laptop
# and wrong for a deployment: nineteen ways to lose a branch, a calibration or
# the audit trail and still answer /health with status "ok" is a production
# hazard, not a convenience.
DEGRADATION_MARKERS: tuple[str, ...] = (
    "checkpoint missing",
    "branch_C unavailable",
    "cohort missing",
    "identity calibration missing",
    "subset calibration table missing",
    "NOT CALIBRATED",
    "fell back to",
    "buckets are missing",
    "INTEGRITY:",
    "call history unavailable",
    "enrollment store unavailable",
    "warmup failed",
    "NO AUTH TOKEN",
)


def strict_mode() -> bool:
    """Refuse to serve anything degraded.

    Separate from VERICALL_REQUIRE_SEALED, which covers artifact integrity
    only. This covers capability: a missing branch, an unfitted calibration, a
    dead audit trail. Both default off because a hard failure on a laptop mid
    demo is worse than the risk it prevents there, and in production it is the
    opposite.
    """
    return os.environ.get("VERICALL_STRICT", "0") == "1"


ICE_SERVERS = _ice_servers()

# Enrollment bounds. 30 s of 16 kHz mono wav is ~1 MB; 8 MB leaves room for a
# larger format without letting an unauthenticated-in-practice laptop deploy
# be used to post arbitrary blobs.
MAX_ENROLL_BYTES = 8 * 1024 * 1024
MIN_ENROLL_S = 3.0

app = FastAPI(title="VeriCall Real-Time Gateway")


class ServerState:
    """Models and calibration, loaded once at startup rather than per call."""

    def __init__(self) -> None:
        self.scorer: Optional[BranchScorer] = None
        self.batcher: Optional[CrossCallBatcher] = None
        self.session_cfg = SessionConfig()
        # Set only when calib/weights@opus.json exists. None means "no
        # channel-specific weights fitted", and a declared opus call then
        # still gets the opus CALIBRATION with the default weights, which is
        # better than nothing and honest about what was fitted.
        self.opus_weights = None
        self.fusion_opus = None
        self.session_cfg_opus = None
        self.branch_c_calibrated = False
        self.degraded: list[str] = []
        self.enroll: Optional[EnrollmentStore] = None
        self.sessions: Dict[str, CallSession] = {}
        self.calibrated = False
        self.device = "cpu"
        self.channel = "clean"
        self.codec_detect = True
        self.weights_fitted = False
        self.draining = False
        self.integrity: Dict[str, dict] = {}
        self.store: Optional[CallStore] = None
        self.limits = Limits()
        self.admission = Admission(self.limits)
        self.calib_by_bucket: Dict[str, BranchCalibrations] = {}
        self.batching = True
        self.policy = PreventionPolicy()
        self.audit = AuditLog()
        self.challenges: Dict[str, object] = {}
        self.notes: list[str] = []
        self.ckpts: Dict[str, str] = {}
        self.calib_pairing = "unknown"
        self.calib_provenance: Dict[str, object] = {}

    @staticmethod
    def _load_into(module, state, what: str) -> None:
        """load_state_dict with a legible failure.

        Bare load_state_dict raises a PyTorch shape-mismatch wall of text that
        does not say the useful thing: the architecture in this checkout no
        longer matches the checkpoint on disk. That happens for a good reason
        -- someone corrected a layer -- and the operator needs to be told to
        retrain rather than to read a hundred tensor names.

        The project has already been bitten once by a checkpoint path that
        failed quietly (a bare VERICALL_CKPT_A served an untrained head). The
        opposite failure, failing loudly but illegibly, wastes the same time.
        """
        try:
            module.load_state_dict(state)
        except RuntimeError as exc:
            raise RuntimeError(
                f"{what} does not match the current architecture. The most "
                f"likely cause is that the model code changed since this "
                f"checkpoint was trained -- retrain it, or check out the "
                f"revision it belongs to.\n  {str(exc)[:300]}") from exc

    @staticmethod
    def _load_ckpt(path: Path, device):
        """torch.load with weights_only=True. Not a style preference.

        weights_only=False runs the pickle machinery, which executes arbitrary
        code during deserialisation -- so loading a checkpoint is equivalent to
        running whatever the person who wrote that file wanted, in the serving
        process, as the serving user. Anyone able to write to checkpoints/ owns
        the gateway. PyTorch made True the default for this reason; this code
        had explicitly opted out of it in all three serving load sites.

        Our checkpoints are {state_dict, branch, seed, dev_eer, history} --
        tensors and plain scalars -- so nothing here needs the unsafe path. If a
        future checkpoint does, the right answer is to change the checkpoint,
        not to reopen this.
        """
        return torch.load(path, map_location=device, weights_only=True)

    def load(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = str(device)

        from ..branches.a2_phase import BranchA2, BranchA2Config
        from ..branches.a_ssl import BranchA, BranchAConfig
        from ..branches.b_prosody import BranchB, BranchBConfig

        def _ckpt(env: str, default: str) -> tuple[Path, bool]:
            """Resolve a checkpoint path, and say whether it was CONFIGURED.

            The old version prepended `checkpoints/` to the default and not to
            an override, so the most natural possible setting --
            VERICALL_CKPT_A=branch_A_seed0.pt, which is exactly what the
            default value looks like -- resolved to <root>/branch_A_seed0.pt,
            did not exist, and left the gateway serving an UNTRAINED head. It
            appended one line to /health notes and carried on producing
            confident numbers, while /health's `checkpoints` block reported the
            configured filename as though it had loaded. Measured: a Hindi
            profile served this way scored 75% call-level EER and every
            diagnosis pointed at the model.

            So a bare filename now resolves against checkpoints/, and the
            caller is told whether a human named this file. A default that is
            absent is a fresh checkout; a CONFIGURED file that is absent is an
            operator error and must not be degraded past.
            """
            raw = os.environ.get(env)
            configured = bool(raw)
            p = Path(raw) if raw else Path(default)
            if p.is_absolute():
                return p, configured
            # A bare name means checkpoints/<name>; a path with separators is
            # taken relative to the project root, so both forms work.
            base = ROOT if p.parent != Path(".") else ROOT / "checkpoints"
            return base / p, configured

        def _require(path: Path, configured: bool, what: str) -> None:
            if configured and not path.exists():
                raise RuntimeError(
                    f"{what} was configured but does not exist: {path}. "
                    f"Refusing to start rather than serve an untrained head -- "
                    f"an unset default may fall back, a named file may not.")

        branch_a = BranchA(BranchAConfig()).to(device).eval()
        ckpt, ckpt_set = _ckpt("VERICALL_CKPT_A", "branch_A_seed0.pt")
        _require(ckpt, ckpt_set, "VERICALL_CKPT_A")
        self.ckpts = {"A": str(ckpt)}
        if ckpt.exists():
            blob = self._load_ckpt(ckpt, device)
            self._load_into(branch_a.head, blob["state_dict"], "branch_A")
        else:
            self.notes.append("branch_A checkpoint missing -- untrained head")

        branch_a2 = None
        ck2, ck2_set = _ckpt("VERICALL_CKPT_A2", "branch_A2_seed0.pt")
        _require(ck2, ck2_set, "VERICALL_CKPT_A2")
        self.ckpts["A2"] = str(ck2)
        if ck2.exists():
            branch_a2 = BranchA2(BranchA2Config()).to(device)
            self._load_into(branch_a2,
                            self._load_ckpt(ck2, device)["state_dict"],
                            "branch_A2")
            branch_a2.eval()
        else:
            self.notes.append("branch_A2 checkpoint missing")

        # Branch C: fitted (calib/C@asv.json, C_cohort.npy), evaluated offline,
        # documented as one of four branches -- and for a long time never
        # loaded here at all. Loading it made speaker-turn detection real: the
        # fusion engine diffs consecutive C embeddings, so that signal works
        # and is what the UI reports.
        #
        # It did NOT revive identity_conflict, and an earlier version of this
        # comment claimed it did. Corrected, because the claim was load-bearing
        # for a documented escalation path. identity_conflict needs
        # `enrolled and identity_llr > 0` (prevent/policy.py) and BOTH are
        # still unreachable in the served system:
        #
        #   1. Nothing sets enrolled=True. SessionConfig.enrolled defaults
        #      False and no serving call site passes it, because there is no
        #      enrollment endpoint.
        #   2. Even with enrollment, identity_llr would be 0.0. engine.py sets
        #      st.llr_target = ev.llr, and the scorer emits C events as
        #      _event("C", w, 0.0) -- hardcoded. BranchC.llr_target(), the
        #      method that would compute a real one, has no call site anywhere
        #      in the package. `0.0 > 0` is False, so the conflict cannot fire
        #      even if condition 1 were met.
        #
        # This is also why calib/C@asv.json (a=1.0219, b=0.4283) and the
        # AS-Norm cohort have no effect on serving: their only consumer is that
        # dead method. The buffers stay at 1.0/0.0. web/index.html already says
        # "identity branch used for turn detection only", which is the accurate
        # description; docs/PRODUCTION_READINESS.md carries the gap.
        branch_c = None
        ckc = ROOT / "calib" / "C_cohort.npy"
        try:
            from ..branches.c_speaker import ASNorm, BranchC, BranchCConfig
            branch_c = BranchC(BranchCConfig(), device=device).to(device).eval()
            if ckc.exists():
                # ASNorm directly, NOT fit_cohort. C_cohort.npy holds 1500x192
                # EMBEDDINGS (scripts/fit_branch_c.py wrote them post-embedder),
                # and fit_cohort takes waveforms and embeds them -- feeding it
                # embeddings would push 192-sample "audio" through ECAPA and
                # produce a cohort of noise. It would not raise; AS-Norm would
                # just normalise against nonsense.
                cohort = torch.from_numpy(np.load(ckc)).float().to(device)
                branch_c.asnorm = ASNorm(cohort,
                                         top_k=BranchCConfig().cohort_size)
            else:
                # Said nothing before: a clone with no C_cohort.npy ran with
                # asnorm=None and reported a healthy branch C. This used to be
                # harmless because its only consumer was dead code; now that
                # identity LLRs are live it is not. Without AS-Norm the score
                # falls back to a raw cosine, which carries a speaker-dependent
                # offset -- so one identity_conflict threshold across speakers
                # stops meaning the same thing for each of them.
                self.notes.append(
                    "branch_C AS-Norm cohort missing (calib/C_cohort.npy): "
                    "identity scores fall back to raw cosine, which is "
                    "speaker-dependent, so identity_conflict is not comparable "
                    "across callers. Speaker-turn detection is unaffected")
            # The ASV score -> LLR mapping, which nothing loaded until now
            # because its only consumer had no call site. An identity LLR on
            # the default 1.0/0.0 scale is worse than none: the policy tests
            # `identity_llr > 0` and the intercept alone decides which trials
            # land on which side of that.
            cal_c = CALIB / "C@asv.json"
            if cal_c.exists():
                cc = load_calibration(cal_c)
                branch_c.load_calibration(cc.a, cc.b)
                self.branch_c_calibrated = True
            else:
                self.notes.append(
                    "branch_C identity calibration missing (calib/C@asv.json): "
                    "identity LLRs stay on the raw AS-Norm scale, so "
                    "identity_conflict thresholds are not meaningful")
        except Exception as exc:                              # noqa: BLE001
            # C depends on speechbrain pulling a pretrained ECAPA model. If that
            # is unavailable the other three branches are still a working
            # detector, so degrade rather than refuse to start -- and say so.
            branch_c = None
            self.notes.append(f"branch_C unavailable ({type(exc).__name__}); "
                              "speaker-turn detection and identity conflict "
                              "are disabled")

        branch_b = None
        ckb, ckb_set = _ckpt("VERICALL_CKPT_B", "branch_B.pt")
        _require(ckb, ckb_set, "VERICALL_CKPT_B")
        self.ckpts["B"] = str(ckb)
        if ckb.exists():
            branch_b = BranchB(BranchBConfig()).to(device)
            self._load_into(branch_b,
                            self._load_ckpt(ckb, device)["state_dict"],
                            "branch_B")
            branch_b.eval()
        else:
            self.notes.append("branch_B checkpoint missing")

        # Channel bucket. Branch A and A2 both have per-channel calibration
        # (fitted slopes differ 1.24x for A and 1.67x for A2), so the wrong
        # bucket mis-scales the LLR by roughly that factor -- and A2 is the
        # most channel-sensitive component in the system (docs/RESULTS.md
        # section 10: 10.54x EER degradation on a noisy line).
        #
        # Runtime codec detection IS implemented (vericall/audio/codec_detect.py,
        # 98.8% per-call on held-out labelled audio). Every available bucket is
        # loaded here and a per-call detector picks between them, because the
        # channel is a property of the call, not of the process.
        #
        # VERICALL_CHANNEL still exists and now means "the bucket to use before
        # the detector is confident, and the only bucket if detection is off".
        # VERICALL_CODEC_DETECT=0 pins it, which is what the demo wants: the
        # demo clips are uncompressed and there is nothing to detect.
        self.channel = os.environ.get("VERICALL_CHANNEL", "clean")
        self.codec_detect = os.environ.get("VERICALL_CODEC_DETECT", "1") != "0"

        # Branch B's LLR is a cohort tail probability, so with no cohort the
        # scorer emits llr=0.0 for EVERY B window (scorer.py: "no cohort yet:
        # contribute nothing").
        #
        # The scores do not move, and the reason is worth stating precisely
        # rather than guessing at: calib/weights.json fits only A and A2, so
        # weight_of("B") is 0.0 and the fusion engine drops B events before
        # they can add either LLR or credited time. B is already inert BY
        # DESIGN -- prosody carried ~39% supervised ceiling on this corpus
        # (docs/RESULTS.md) and was zero-weighted for it. A missing cohort
        # therefore costs nothing at the current operating point.
        #
        # Still worth a note, because that is true only while B's weight is
        # zero. Re-weight B without restoring the cohort and the branch
        # silently contributes nothing while appearing to be fused.
        cohort = None
        bcoh = CALIB / "B_bonafide_cohort.npy"
        if bcoh.exists():
            cohort = torch.from_numpy(np.load(bcoh)).float()
        elif branch_b is not None:
            self.notes.append(
                "branch_B cohort missing (calib/B_bonafide_cohort.npy): B "
                "emits llr=0.0 on every window. Harmless while B's fusion "
                "weight is 0.0, which it is -- but restore the cohort before "
                "ever giving B a nonzero weight")

        def load_bucket(bucket: str) -> Optional[BranchCalibrations]:
            a = CALIB / f"A@{bucket}.json"
            a2 = CALIB / f"A2@{bucket}.json"
            if not a.exists():
                return None
            c = BranchCalibrations()
            c.A = load_calibration(a)
            if a2.exists():
                c.A2 = load_calibration(a2)
            c.B_cohort = cohort
            return c

        self.calib_by_bucket = {}
        # "opus" is in this list but NOT in the runtime detector's vocabulary,
        # and that asymmetry is deliberate. Opus is wideband and leaves no
        # packet-loss signature, so it looks exactly like clean audio to a
        # detector built to separate clean from G.711 from a lossy line --
        # measured at 97-98% "clean" (docs/RESULTS.md section 23). The channel
        # is therefore not detectable from the audio and has to be DECLARED by
        # the client, which knows its own transport. See the `channel` query
        # parameter on /ws/.
        for bucket in ("clean", "g711", "noisy_line", "opus", "pooled"):
            got = load_bucket(bucket)
            if got is not None:
                self.calib_by_bucket[bucket] = got

        calib = (self.calib_by_bucket.get(self.channel)
                 or self.calib_by_bucket.get("pooled")
                 or BranchCalibrations())
        if self.channel not in self.calib_by_bucket and self.calib_by_bucket:
            self.notes.append(
                f"no {self.channel} bucket; fell back to "
                f"{'pooled' if 'pooled' in self.calib_by_bucket else 'none'}")
        if calib.B_cohort is None:
            calib.B_cohort = cohort

        missing = [b for b in ("clean", "g711", "noisy_line")
                   if b not in self.calib_by_bucket]
        if self.codec_detect and missing:
            # Detecting a channel we have no calibration for is worse than not
            # detecting: it would select a bucket and silently get the fallback.
            self.notes.append(
                f"codec detection is on but these buckets are missing: "
                f"{', '.join(missing)} -- calls detected as those keep the "
                f"startup bucket")
        # Integrity, before anything is trusted enough to serve. A tampered
        # checkpoint no longer executes code (weights_only=True) but it still
        # SCORES, and that failure is silent: altered weights do not crash and
        # do not look wrong, they produce confident numbers nothing downstream
        # can distinguish from correct ones. Three edited digits in a Platt
        # JSON move every threshold in the system while /health still says
        # calibrated: true.
        sealed = [Path(v) for v in self.ckpts.values()]
        ck_root = sealed[0].parent if sealed else ROOT / "checkpoints"
        self.integrity = {
            "checkpoints": verify_integrity(ck_root, sealed).as_dict(),
            # *.json AND *.npy. calib/MANIFEST.sha256 seals both cohorts,
            # but must_cover was glob("*.json") only, so the two .npy files
            # were sealed and never verified -- the manifest promised coverage
            # it did not deliver. C_cohort.npy is exactly what an attacker
            # would prefer to edit: altering it changes AS-Norm silently,
            # with no crash and nothing downstream able to tell.
            "calibration": verify_integrity(
                CALIB, sorted([*CALIB.glob("*.json"),
                               *CALIB.glob("*.npy")])).as_dict(),
        }
        bad = [k for k, v in self.integrity.items() if not v["clean"]]
        if bad:
            detail = "; ".join(
                f"{k}: {v['mismatched'] or v['missing'] or v['unsealed']}"
                for k, v in self.integrity.items() if not v["clean"])
            msg = f"INTEGRITY: {detail}"
            if enforcement_required():
                # VERICALL_REQUIRE_SEALED=1 means a deployment has decided that
                # serving unverified weights is worse than not serving.
                raise RuntimeError(
                    msg + " -- refusing to start with VERICALL_REQUIRE_SEALED=1")
            self.notes.append(
                msg + " (not enforced; set VERICALL_REQUIRE_SEALED=1 to refuse)")

        self.calibrated = calib.fitted
        if not self.calibrated:
            self.notes.append(
                "NOT CALIBRATED: branch outputs are raw logits, not LLRs. "
                "Every risk threshold assumes LLRs, so the numbers are not "
                "meaningful at this operating point.")

        fusion = FusionConfig()
        subsets = CALIB / "subsets.json"
        if subsets.exists():
            # eta comes from FusionConfig, where all three branches carry
            # measured values. This used to override A's from summary.json,
            # which still holds the RETRACTED 0.080 -- measured while the
            # chunker emitted byte-identical windows, so its autocorrelation
            # was float non-determinism (docs/RESULTS.md section 15). Reading
            # that file here would silently reinstate the bad number in the
            # serving path.
            # Weights come from the profile too. calib/ carries fitted ones;
            # a profile without weights.json keeps the FusionConfig default,
            # which is correct for one that has never been fitted.
            w = load_weights(CALIB / "weights.json", FusionConfig().w)
            self.weights_fitted = (CALIB / "weights.json").exists()
            fusion = FusionConfig(subset_calib=load_subset_calib(subsets), w=w)

            # Per-channel fusion weights, for opus only and for one reason:
            # refitting A2's Platt mapping removes its calibration loss but
            # cannot recover its information. Through Opus, A2's min_Cllr is
            # 0.56 against 0.052 clean -- min_Cllr being the floor a PERFECT
            # mapping reaches -- so nine tenths of what A2 knew is simply gone.
            # A branch in that state must not keep the weight it earned when it
            # had the information, or the accumulator banks confident evidence
            # from a branch that is barely better than a coin.
            self.opus_weights = None
            wopus = CALIB / "weights@opus.json"
            if wopus.exists():
                self.opus_weights = load_weights(wopus, w)
                self.fusion_opus = FusionConfig(
                    subset_calib=load_subset_calib(subsets),
                    w=self.opus_weights)

            prov_path = CALIB / "provenance.json"
            if prov_path.exists():
                prov = json.loads(prov_path.read_text())
                self.calib_provenance = prov
                want = prov.get("fitted_for", {})
                bad = [b for b, name in want.items()
                       if b in self.ckpts and Path(self.ckpts[b]).name != name]
                self.calib_pairing = "mismatch" if bad else "ok"
                for b in bad:
                    self.notes.append(
                        f"CALIBRATION PAIRING: branch {b} is serving "
                        f"{Path(self.ckpts[b]).name} but {CALIB.name} was fitted "
                        f"for {want[b]}. A Platt mapping applied to a head it "
                        f"was not fitted for is confident nonsense, not a "
                        f"degraded score.")
            else:
                self.calib_pairing = "unknown"
                self.notes.append(
                    f"{CALIB.name} has no provenance.json, so the head/"
                    f"calibration pairing cannot be checked. Re-run "
                    f"scripts/fit_subsets.py to record it.")
        else:
            self.notes.append("subset calibration table missing -- every "
                              "branch subset falls back to (alpha=0, scale=1)")

        self.session_cfg = SessionConfig(fusion=fusion, policy=TierPolicy())
        if self.fusion_opus is not None:
            self.session_cfg_opus = SessionConfig(fusion=self.fusion_opus,
                                                  policy=TierPolicy())
        self.scorer = BranchScorer(branch_a, branch_a2, branch_b, calib, device,
                                   branch_c=branch_c)
        # Fast lane only. It carries 2/3 of the windows and all of the SSL cost;
        # the slow lane runs at 1 Hz and is cheap enough that queueing it would
        # cost more in added latency than it saves in GPU efficiency.
        # Toggleable so the two paths can be compared at identical load.
        # docs/BENCHMARKS.md argues for batching from offline microbenchmarks of
        # the raw forward pass; whether that transfers to online serving at this
        # arrival rate is an empirical question, not a settled one.
        # Defaults OFF, on measurement rather than on the design doc.
        # A/B at identical load (8 calls x 15 s, scripts/ab_batching.sh):
        #   direct   chunk p50   7.8 ms  p99  97.8 ms   <- meets the 250 ms SLO
        #   batched  chunk p50 197.8 ms  p99 779.3 ms   <- breaches it 3x
        # The GPU forward is 7.8 ms, so the GPU was never the bottleneck at this
        # arrival rate and batching it buys nothing while adding a collect
        # window and a queue. docs/BENCHMARKS.md argued for batching from
        # OFFLINE microbenchmarks of the raw forward pass; that did not transfer
        # to online serving here. Kept behind a flag because it should win again
        # once concurrency is high enough to saturate the device.
        # Now DEFAULTS ON. The A/B above was run when (a) the batcher batched
        # score_fast whole, calibration included, under a GPU lock, and (b) the
        # non-batched path ran its GPU call synchronously ON THE EVENT LOOP --
        # so "direct" was winning by blocking every other call, and the measured
        # comparison was between two broken things. Both are fixed: the batcher
        # batches the raw forward only (per-call calibration applied after), and
        # the direct path goes through asyncio.to_thread.
        #
        # The old comment's own prediction was that batching "should win again
        # once concurrency is high enough to saturate the device". 50 concurrent
        # calls is that regime: at a 0.5 s hop they present ~100 windows/s, and
        # a MIG slice fed one window at a time clocks down between kernels
        # (batch 1 varies 14x in latency, batch 32 varies 1.07x).
        self.batching = os.environ.get("VERICALL_BATCHING", "1") != "0"
        self.batcher = (CrossCallBatcher(self.scorer.forward_fast)
                        if self.batching else None)
        self.limits = Limits()
        self.admission = Admission(self.limits)
        if auth_token() is None:
            self.notes.append(
                "NO AUTH TOKEN: VERICALL_AUTH_TOKEN is unset, so every "
                "WebSocket connection is accepted. Correct for a laptop, wrong "
                "for anything reachable by anyone else.")
        # Call history. Separate from the audit log on purpose: the audit log
        # is a narrow legal artefact (one record per action change, no scores,
        # no timeline) and cannot answer "show me yesterday's flagged calls",
        # which is the first question anyone operating this would ask.
        try:
            self.store = CallStore()
            log.info("call history at %s", self.store.path)
        except Exception as exc:                              # noqa: BLE001
            # History is valuable, not load-bearing. A detector that refuses to
            # score calls because it cannot write history is worse than one
            # that scores them and says the history is off.
            self.store = None
            self.notes.append(f"call history unavailable "
                              f"({type(exc).__name__}); calls are not recorded")

        # Enrolled voiceprints. Same failure philosophy as history: identity
        # escalation is valuable and not load-bearing, and a detector that
        # refuses to score calls because it cannot read a voiceprint file is
        # worse than one that scores them and says identity is off.
        try:
            # Anchored to the project root, not the CWD. A relative override
            # here would scatter biometric data wherever the gateway was
            # started from -- the same trap as VERICALL_DB and the checkpoint
            # paths, and worse in consequence: a voiceprint written to an
            # unexpected directory is personal data nobody knows to delete.
            _enr = os.environ.get("VERICALL_ENROLL_DB")
            _enr_p = Path(_enr) if _enr else Path("enroll/voiceprints.npz")
            self.enroll = EnrollmentStore(
                _enr_p if _enr_p.is_absolute() else ROOT / _enr_p)
            if branch_c is None and self.enroll.stats()["enrolled"]:
                self.notes.append(
                    "voiceprints are enrolled but branch_C is unavailable, so "
                    "identity comparison cannot run")
        except Exception as exc:                              # noqa: BLE001
            self.enroll = None
            self.notes.append(f"enrollment store unavailable "
                              f"({type(exc).__name__}); identity_conflict "
                              f"escalation is disabled")

        # Last, so every note has been collected. Refusing here rather than
        # per-site keeps the decision in one place and makes the message list
        # everything that is wrong instead of only the first thing.
        self.degraded = [n for n in self.notes
                         if any(m in n for m in DEGRADATION_MARKERS)]
        if self.degraded and strict_mode():
            raise RuntimeError(
                "VERICALL_STRICT=1 and the gateway is degraded:\n  - "
                + "\n  - ".join(self.degraded)
                + "\nFix these or unset VERICALL_STRICT to serve anyway.")

        self._warmup(device)
        log.info("loaded on %s | calibrated=%s | notes=%s",
                 self.device, self.calibrated, self.notes or "none")

    def _warmup(self, device) -> None:
        """Run each branch once on dummy audio before serving traffic.

        The first CUDA forward pays kernel autotuning and lazy module init. Left
        unwarmed it lands on a real caller: measured p99 was 830 ms at N=1 and
        N=2 -- worse than at N=16, purely because one cold inference dominates a
        short sample. That would hit the first call of a live demo. Warming here
        moves the cost to startup, where nobody is waiting on it.
        """
        import time as _t
        t0 = _t.perf_counter()
        try:
            dummy = torch.zeros(1, 16000, device=device)
            if self.scorer.a is not None:
                self.scorer.a.score(dummy)
            if self.scorer.a2 is not None:
                self.scorer.a2.score(dummy)
            if self.scorer.b is not None:
                self.scorer.b.score(torch.zeros(1, 48000, device=device),
                                    torch.zeros(1, dtype=torch.long, device=device))
            self.scorer.stats.latencies_ms.clear()   # don't bill warmup to p99
            self.scorer.stats.batches = 0
            self.scorer.stats.windows = 0
            log.info("warmup complete in %.0f ms", (_t.perf_counter() - t0) * 1000)
        except Exception as exc:                                  # noqa: BLE001
            self.notes.append(f"warmup failed: {type(exc).__name__}: {exc}"[:120])


STATE = ServerState()


@app.on_event("startup")
def _startup() -> None:
    STATE.load()


@app.on_event("shutdown")
async def _shutdown() -> None:
    """Stop admitting, then let live calls finish.

    Kubernetes sends SIGTERM and waits terminationGracePeriodSeconds before
    SIGKILL. Without a drain, a rolling restart kills every call in flight
    mid-sentence -- and for this service that means a fraud decision is simply
    never made on calls that were being monitored, which is the failure nobody
    sees in a dashboard.
    """
    STATE.draining = True
    deadline = time.monotonic() + STATE.limits.drain_timeout_s
    while STATE.sessions and time.monotonic() < deadline:
        await asyncio.sleep(0.25)
    if STATE.sessions:
        log.warning("shutdown: %d call(s) still active after %.0fs drain",
                    len(STATE.sessions), STATE.limits.drain_timeout_s)
    else:
        log.info("shutdown: drained cleanly")


@app.get("/health")
def health(request: Request) -> dict:
    """Liveness for anyone; configuration for whoever holds the token.

    This endpoint answers two different questions for two different callers,
    and they do not deserve the same answer. A load balancer needs to know
    whether to send traffic here. An operator needs to know which heads and
    which calibration are loaded and what is degraded. Only the second is
    reconnaissance -- checkpoint names, calibration provenance, the ICE server
    list, how many voiceprints are enrolled, the exact limits to push against.

    So the payload is graduated, and `check_auth` already draws the line in
    the right place: it returns True when no token is configured, which means
    a laptop deployment -- the case this project is demoed in -- still gets
    the full payload with no token and no changed tooling. Configure a token
    and the unauthenticated answer narrows to liveness.

    It stays 200 either way. A probe that needs a secret is a probe that gets
    switched off, and a 401 to Kubernetes reads as an unhealthy pod, which
    would take the service out rather than protect it.
    """
    presented = (request.query_params.get("token")
                 or request.headers.get("x-vericall-token"))
    status = ("draining" if STATE.draining
              else "ok" if STATE.scorer is not None else "loading")
    if not check_auth(presented):
        return {
            "status": status,
            "active_calls": len(STATE.sessions),
            # Whether anything is lost, but not what -- an operator who needs
            # the thirteen notes can authenticate for them.
            "degraded": bool(STATE.degraded),
            "auth_required": True,
            "detail": "authenticate for configuration detail",
        }
    cfg = STATE.session_cfg.fusion
    return {
        "status": status,
        "device": STATE.device,
        "calibrated": STATE.calibrated,
        "channel_bucket": STATE.channel,
        "runtime_codec_detection": STATE.codec_detect,
        "calibration_buckets": sorted(STATE.calib_by_bucket),
        "fusion_weights": dict(STATE.session_cfg.fusion.w),
        "fusion_weights_fitted": STATE.weights_fitted,
        # Channels a client may DECLARE with ?channel=, as opposed to the ones
        # the runtime detector can recognise on its own. Opus appears here and
        # not there on purpose: it is undetectable from the audio.
        # Named separately from `notes`: an operator wants to know whether
        # anything is actually LOST, not to read thirteen informational lines.
        "degraded": STATE.degraded,
        "strict_mode": strict_mode(),
        "declarable_channels": sorted(STATE.calib_by_bucket),
        "opus_bucket": {
            "calibration": "opus" in STATE.calib_by_bucket,
            "fusion_weights": (dict(STATE.opus_weights)
                               if STATE.opus_weights else None),
        },
        "active_calls": len(STATE.sessions),
        "limits": STATE.limits.as_dict(),
        "admission": STATE.admission.stats(),
        "auth_required": auth_token() is not None,
        "batching_enabled": STATE.batching,
        "subset_calibration_entries": len(cfg.subset_calib),
        "eta": {"A": cfg.eta.get("A"), "A2": cfg.eta.get("A2"),
                "B": cfg.eta.get("B")},
        # All three are measured now (800 calls, each branch on its own lane --
        # scripts/measure_eta_branches.py). Kept as an explicit list rather than
        # dropped, because a health endpoint that stops distinguishing measured
        # values from defaults cannot warn you when the next branch is added.
        "eta_measured": ["A", "A2", "B"],
        "inference_precision": STATE.scorer.precision if STATE.scorer else None,
        "branches_loaded": sorted(
            b for b, m in (("A", STATE.scorer.a), ("A2", STATE.scorer.a2),
                           ("B", STATE.scorer.b), ("C", STATE.scorer.c))
            if m is not None) if STATE.scorer else [],
        # Basenames, not absolute paths. /health is the endpoint most likely to
        # be exposed to a load balancer, a status page or a screenshot, and the
        # operator question it answers ("which calibration and which heads?") is
        # answered just as well by the name. The directory layout of the host is
        # reconnaissance, not an answer.
        "profile": ACTIVE_PROFILE,
        "profile_error": _PROFILE_ERROR,
        "calib_dir": CALIB.name,
        "checkpoints": {b: Path(p).name for b, p in STATE.ckpts.items()},
        "calibration_pairing": STATE.calib_pairing,
        "integrity": STATE.integrity,
        "calibration_fitted_for": STATE.calib_provenance,
        "scorer_latency_ms": STATE.scorer.stats.percentiles() if STATE.scorer else {},
        "notes": STATE.notes,
        "prevention": {"contexts": sorted(CONTEXTS),
                       "audit_records": STATE.audit.written},
        "webrtc": {**HUB.stats(),
                   "ice_servers": [u for sv in ICE_SERVERS
                                   for u in (sv.get("urls") or [])],
                   "turn_configured": any(
                       str(u).startswith("turn") for sv in ICE_SERVERS
                       for u in (sv.get("urls") or []))},
        "identity": {
            # Every condition identity_conflict needs, reported separately, so
            # "why did it not escalate" is answerable without reading code.
            # All four were silently false before: the branch was loaded but
            # its calibration was not, its LLR was hardcoded 0.0, and nothing
            # could set enrolled.
            "branch_c_loaded": STATE.scorer is not None and STATE.scorer.c is not None,
            "calibration_loaded": STATE.branch_c_calibrated,
            "asnorm_cohort": (STATE.scorer is not None and STATE.scorer.c is not None
                              and STATE.scorer.c.asnorm is not None),
            "voiceprints": (STATE.enroll.stats() if STATE.enroll
                            else {"enrolled": 0, "path": None}),
            "identity_errors": (STATE.scorer.stats.identity_errors
                                if STATE.scorer else 0),
        },
        "call_history": (STATE.store.stats() if STATE.store
                         else {"available": False}),
    }


@app.get("/metrics")
def metrics() -> dict:
    """Scorer-side latency, for the load generator.

    chunk_inference_latency is the window-buffered -> score-available leg only.
    It is deliberately NOT the same number as time_to_alert, which is bounded by
    evidence accumulation rather than compute; conflating them is how a
    millisecond figure ends up advertised for a system whose real detection
    latency is seconds.
    """
    st = STATE.scorer.stats if STATE.scorer else None
    pct = st.percentiles() if st else {}
    return {
        "chunk_inference_latency_ms": pct,
        "windows_scored": st.windows if st else 0,
        "batches": st.batches if st else 0,
        "batching_enabled": STATE.batching,
        "cross_call_batching": (STATE.batcher.stats()
                                if STATE.batcher else {}),
        "active_calls": len(STATE.sessions),
        "calibrated": STATE.calibrated,
        "channel_bucket": STATE.channel,
    }


@app.get("/metrics/prom")
def metrics_prom() -> PlainTextResponse:
    """Prometheus text exposition.

    /metrics returns JSON because the load generator reads it. Nothing in a
    normal monitoring stack does: Prometheus, and every hosted product that
    speaks its protocol, scrapes this format. A service whose numbers cannot be
    scraped has no alerting, and a fraud detector with no alerting on ITS OWN
    health is the thing that fails quietly at 3 a.m.

    The four that matter operationally, and why:
      active/limit        how close admission is to shedding
      rejections          shedding is happening RIGHT NOW
      chunk p99           the SLO this service is sized against
      queue depth         the leading indicator -- it rises before p99 does
    """
    st = STATE.scorer.stats if STATE.scorer else None
    pct = st.percentiles() if st else {}
    b = STATE.batcher.stats() if STATE.batcher else {}
    adm = STATE.admission.stats()
    lines = [
        "# HELP vericall_up 1 when models are loaded and not draining",
        "# TYPE vericall_up gauge",
        f"vericall_up {int(STATE.scorer is not None and not STATE.draining)}",
        "# HELP vericall_active_calls Calls currently being scored",
        "# TYPE vericall_active_calls gauge",
        f"vericall_active_calls {adm['active']}",
        "# HELP vericall_max_calls Admission limit",
        "# TYPE vericall_max_calls gauge",
        f"vericall_max_calls {adm['limit']}",
        "# HELP vericall_rejected_total Calls refused because the server was full",
        "# TYPE vericall_rejected_total counter",
        f"vericall_rejected_total {adm['rejected']}",
        "# HELP vericall_rejected_per_client_total Calls refused by the fair-share cap",
        "# TYPE vericall_rejected_per_client_total counter",
        f"vericall_rejected_per_client_total {adm.get('rejected_per_client', 0)}",
        "# HELP vericall_distinct_clients Clients holding at least one call",
        "# TYPE vericall_distinct_clients gauge",
        f"vericall_distinct_clients {adm.get('distinct_clients', 0)}",
        "# HELP vericall_shed_windows_total Windows shed by a full inference queue",
        "# TYPE vericall_shed_windows_total counter",
        f"vericall_shed_windows_total {b.get('rejected', 0)}",
        "# HELP vericall_inference_queue_depth Windows waiting for a forward pass",
        "# TYPE vericall_inference_queue_depth gauge",
        f"vericall_inference_queue_depth {b.get('queue_depth', 0)}",
        "# HELP vericall_windows_scored_total Fast-lane windows scored",
        "# TYPE vericall_windows_scored_total counter",
        f"vericall_windows_scored_total {st.windows if st else 0}",
        "# HELP vericall_mean_batch_size Mean cross-call batch size",
        "# TYPE vericall_mean_batch_size gauge",
        f"vericall_mean_batch_size {b.get('mean_batch', 0.0)}",
        "# HELP vericall_chunk_inference_ms Forward-pass latency",
        "# TYPE vericall_chunk_inference_ms gauge",
        f'vericall_chunk_inference_ms{{quantile="0.5"}} {pct.get("p50", 0.0)}',
        f'vericall_chunk_inference_ms{{quantile="0.95"}} {pct.get("p95", 0.0)}',
        f'vericall_chunk_inference_ms{{quantile="0.99"}} {pct.get("p99", 0.0)}',
        "# HELP vericall_calibrated 1 when branch outputs are LLRs",
        "# TYPE vericall_calibrated gauge",
        f"vericall_calibrated {int(STATE.calibrated)}",
        "# HELP vericall_calibration_pairing_ok 1 when heads match their calibration",
        "# TYPE vericall_calibration_pairing_ok gauge",
        f"vericall_calibration_pairing_ok {int(STATE.calib_pairing == 'ok')}",
        "# HELP vericall_integrity_ok 1 when served artifacts match their manifest",
        "# TYPE vericall_integrity_ok gauge",
        f"vericall_integrity_ok "
        f"{int(all(v.get('clean') for v in STATE.integrity.values()) if STATE.integrity else 0)}",
        "# HELP vericall_audit_records_total Prevention decisions written",
        "# TYPE vericall_audit_records_total counter",
        f"vericall_audit_records_total {STATE.audit.written}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n",
                             media_type="text/plain; version=0.0.4")


@app.get("/api/calls")
def api_calls(request: Request, limit: int = 50,
              alerted_only: bool = False) -> dict:
    if (bad := _require_auth(request)):
        return bad
    """Recent calls, newest first. The operator's first question.

    `limit` is clamped rather than trusted: an unbounded limit on a public
    read endpoint is a cheap way to make the server serialise a large result
    set on request.
    """
    if STATE.store is None:
        return {"error": "call history unavailable", "calls": []}
    limit = max(1, min(int(limit), 500))
    return {"calls": STATE.store.recent(limit=limit,
                                        alerted_only=bool(alerted_only)),
            "stats": STATE.store.stats()}


@app.get("/api/calls/{call_id}")
def api_call(call_id: str, request: Request) -> dict:
    if (bad := _require_auth(request)):
        return bad
    """One call with its tier transitions, for reviewing a decision."""
    if STATE.store is None:
        return {"error": "call history unavailable"}
    if not valid_call_id(call_id):
        # Same charset rule as the WebSocket. This id reaches a SQL parameter
        # and a log line, and "it is only a read" is not a reason to skip it.
        return {"error": "invalid call_id"}
    got = STATE.store.call(call_id)
    return got or {"error": "not found"}


@app.post("/api/calls/prune")
def api_prune(request: Request, older_than_days: float = 30.0) -> dict:
    """Apply retention. Exposed so it can be driven by a cron rather than
    living as a sentence in a policy document.

    Authenticated, and this one matters most of the three: it DESTROYS call
    history. Unauthenticated, anyone who could reach the port could erase the
    record of which calls were flagged -- which is the record an incident
    review starts from.

    All three /api/calls endpoints were unauthenticated while /ws/ and
    /api/enroll/ both required a token, so the intent was never in doubt; the
    REST surface was simply never wired to it. GET leaked per-call risk
    scores, timings and client addresses; POST deleted them.
    """
    if (bad := _require_auth(request)):
        return bad
    if STATE.store is None:
        return {"error": "call history unavailable"}
    removed = STATE.store.prune(older_than_days=max(0.0, float(older_than_days)))
    return {"removed": removed, "older_than_days": older_than_days,
            "stats": STATE.store.stats()}


# ------------------------------------------------------------- enrollment
# A voiceprint is biometric personal data. These endpoints are guarded by the
# same token as the WebSocket, they never return an embedding, and DELETE
# exists because the right to erasure applies. See vericall/store/enroll.py
# for what is and is not stored.

def _require_auth(request: Request) -> Optional[dict]:
    presented = (request.query_params.get("token")
                 or request.headers.get("x-vericall-token"))
    return None if check_auth(presented) else {"error": "unauthorized"}


@app.get("/api/enroll")
def api_enroll_list(request: Request) -> dict:
    """Metadata only. Embeddings are never returned over an API: a leaked
    voiceprint is not revocable the way a password is."""
    if (bad := _require_auth(request)):
        return bad
    if STATE.enroll is None:
        return {"error": "enrollment unavailable"}
    return {"enrolled": STATE.enroll.list(), "stats": STATE.enroll.stats()}


@app.post("/api/enroll/{speaker_id}")
async def api_enroll(speaker_id: str, request: Request) -> dict:
    """Enroll one speaker from a wav body. Replaces any existing voiceprint.

    Audio in, embedding stored, audio discarded -- it is never written to disk
    and never leaves this function. Keeping it would turn a detector into a
    corpus of customer recordings that is worth stealing.
    """
    if (bad := _require_auth(request)):
        return bad
    if STATE.enroll is None:
        return {"error": "enrollment unavailable"}
    if STATE.scorer is None or STATE.scorer.c is None:
        return {"error": "branch_C unavailable, cannot compute a voiceprint"}
    if not valid_speaker_id(speaker_id):
        # Length only in the log line; the value is client-controlled.
        log.warning("rejected enrollment: malformed speaker id (%d bytes)",
                    len(speaker_id))
        return {"error": "invalid speaker_id"}

    raw = await request.body()
    if not raw:
        return {"error": "empty body; POST a wav file"}
    if len(raw) > MAX_ENROLL_BYTES:
        return {"error": f"body over {MAX_ENROLL_BYTES} bytes"}

    try:
        wav = load_audio(raw)
    except Exception as exc:                                  # noqa: BLE001
        return {"error": f"could not decode audio ({type(exc).__name__})"}

    seconds = wav.numel() / 16_000.0
    if seconds < MIN_ENROLL_S:
        # Refused, not warned. A voiceprint from a second of audio is a bad
        # voiceprint that then silently governs an escalation decision, and
        # the caller can trivially send more audio.
        return {"error": f"need at least {MIN_ENROLL_S:g}s of speech, "
                         f"got {seconds:.1f}s"}

    emb = await asyncio.to_thread(
        lambda: STATE.scorer.c.embed(wav.unsqueeze(0).to(STATE.scorer.device))
        .float().cpu().numpy().reshape(-1))
    try:
        rec = STATE.enroll.enroll(speaker_id, emb, seconds=seconds)
    except ValueError as exc:
        return {"error": str(exc)}
    log.info("enrolled a speaker from %.1fs of audio", seconds)
    return {"ok": True, **rec}


@app.delete("/api/enroll/{speaker_id}")
def api_enroll_delete(speaker_id: str, request: Request) -> dict:
    if (bad := _require_auth(request)):
        return bad
    if STATE.enroll is None:
        return {"error": "enrollment unavailable"}
    if not valid_speaker_id(speaker_id):
        return {"error": "invalid speaker_id"}
    return {"deleted": STATE.enroll.delete(speaker_id)}


@app.websocket("/ws/{call_id}")
async def websocket_endpoint(websocket: WebSocket, call_id: str) -> None:
    # --- gate BEFORE accepting -------------------------------------------
    # Each of these refuses with a distinct close code so a client can tell a
    # bad token from an overload from a malformed id, and retry the one that is
    # worth retrying. Closing without accepting would drop the reason on the
    # floor, so accept, say why, close.
    await websocket.accept()

    presented = (websocket.query_params.get("token")
                 or websocket.headers.get("x-vericall-token"))
    if not check_auth(presented):
        await websocket.send_json({"error": "unauthorized"})
        await websocket.close(code=4401)
        log.warning("rejected call: bad token from %s",
                    websocket.client.host if websocket.client else "?")
        return

    if not valid_call_id(call_id):
        # Never echo the rejected id back into a log line or a response: it is
        # attacker-controlled and this is exactly how log injection works.
        await websocket.send_json({"error": "invalid call_id"})
        await websocket.close(code=4400)
        log.warning("rejected call: malformed call_id (%d bytes)", len(call_id))
        return

    if STATE.scorer is None:
        await websocket.send_json({"error": "models not loaded"})
        await websocket.close(code=1013)      # try again later
        return

    if STATE.draining:
        # 1013 "try again later" is the correct signal to a client that should
        # reconnect to another replica, not give up.
        await websocket.send_json({"error": "draining"})
        await websocket.close(code=1013)
        return

    if call_id in STATE.sessions:
        # Without this, a second connection with the same id silently REPLACES
        # the first in the session table: the original call keeps streaming into
        # a session nobody will clean up, and whichever disconnects first pops
        # the other's entry. A caller can also simply guess an id and take over.
        await websocket.send_json({"error": "call_id already active"})
        await websocket.close(code=4409)
        return

    # Per-client identity for the fair-share cap. Behind a proxy the peer
    # address is the proxy, so a forwarded header is preferred when present --
    # and that header is only trustworthy if the ingress sets it and strips any
    # client-supplied copy, which is the deployer's job to get right. Falling
    # back to the peer address means the cap still bites when it is absent.
    client_key = (websocket.headers.get("x-forwarded-for", "").split(",")[0].strip()
                  or (websocket.client.host if websocket.client else "unknown"))
    if not await STATE.admission.acquire(client_key):
        at_cap = STATE.admission.active >= STATE.limits.max_calls
        await websocket.send_json({
            "error": "at capacity" if at_cap else "per-client limit",
            "active": STATE.admission.active,
            "limit": STATE.limits.max_calls,
            "limit_per_client": STATE.limits.max_per_client,
        })
        await websocket.close(code=1013)
        log.warning("rejected call: %s (active %d)",
                    "at capacity" if at_cap else "per-client limit",
                    STATE.admission.active)
        return

    # Transaction context drives the intervention. A real deployment gets it
    # from the IVR/CRM leg the call is on; here it is a query parameter so the
    # demo can show the same score producing different actions.
    context = websocket.query_params.get("context", "inbound_unclassified")
    if context not in CONTEXTS:
        context = "inbound_unclassified"
    last_action = None

    # Declared transport. The channel detector reads the audio, and for every
    # channel it was fitted on that is the right way round -- the channel is a
    # property of the call, not of the process. Opus is the exception: it is
    # wideband and loses no frames, so it is indistinguishable from clean audio
    # by measurement (97-98% "clean", docs/RESULTS.md section 23) while being
    # the channel that costs Branch A2 nine tenths of its information.
    #
    # It cannot be detected, so it is declared. The browser placing a WebRTC
    # call is the one component that knows for certain what codec is in the
    # path, and web/call.html sends channel=opus. Unknown values are ignored
    # rather than rejected: a wrong bucket is a scale error, not a reason to
    # drop a call, and the value is attacker-controlled.
    declared, complaint = resolve_channel(
        websocket.query_params.get("channel"), STATE.calib_by_bucket)
    if complaint:
        log.warning("ignoring declared channel: %s", complaint)

    cfg_for_call = STATE.session_cfg
    if declared == "opus" and STATE.session_cfg_opus is not None:
        cfg_for_call = STATE.session_cfg_opus

    # Who does this caller CLAIM to be? Identity is a different question from
    # synthesis, and the answer is only interesting in combination: a
    # convincing clone of an enrolled customer is the worst cell in the policy
    # table, not the safest, because identity matching and a high spoof score
    # together mean someone is successfully impersonating a known person.
    #
    # A real deployment takes the claim from the IVR/CRM leg (the account the
    # caller entered, the number they dialled from). Here it is a query
    # parameter so the behaviour is demonstrable. Either way the claim is
    # UNTRUSTED input -- it selects which voiceprint to compare against, it
    # does not assert that the comparison succeeded.
    enroll_emb = None
    claims = websocket.query_params.get("claims")
    if claims and STATE.enroll is not None:
        if not valid_speaker_id(claims):
            log.warning("ignoring malformed claims id (%d bytes)", len(claims))
        else:
            vec = STATE.enroll.get(claims)
            if vec is None:
                # Not an error and not a refusal: an unknown claimed identity
                # is an ordinary event (a new customer, a typo), and the call
                # still needs scoring for synthesis.
                log.info("call claims an identity with no enrolled voiceprint")
            else:
                enroll_emb = torch.from_numpy(vec).float()
                cfg_for_call = replace(cfg_for_call, enrolled=True)

    session = CallSession(call_id, cfg_for_call)
    # A declared channel pins the bucket. Running the detector alongside it
    # would let it overrule a declaration it cannot make itself -- it has no
    # opus verdict to give, so it would vote the call back to clean.
    detector = (CodecDetector(default=STATE.channel)
                if STATE.codec_detect and declared is None else None)
    last_bucket = declared or STATE.channel
    STATE.sessions[call_id] = session
    if STATE.store is not None:
        await asyncio.to_thread(
            STATE.store.open_call, call_id, context, client_key,
            STATE.channel,
            [b for b, m in (("A", STATE.scorer.a), ("A2", STATE.scorer.a2),
                            ("B", STATE.scorer.b), ("C", STATE.scorer.c))
             if m is not None],
            STATE.calibrated)
    # Per-call history state. Tracked here rather than queried back out of the
    # database, because a read per chunk would be the one thing that made this
    # table cost anything.
    # None, not 0.0. A genuine call's S is strongly NEGATIVE (about -94 nats
    # on the demo clip), so seeding the max at 0.0 reported +0.0 for every
    # clean call -- which reads as "no evidence either way" when the truth was
    # "overwhelming evidence of a real person".
    hist_max_risk = None
    hist_alert_at = None
    hist_tier = 0
    await websocket.send_json({
        "call_id": call_id, "state": "ready",
        "calibrated": STATE.calibrated, "notes": STATE.notes,
    })

    try:
        while True:
            # An idle timeout, because TCP will not tell us. A half-open socket
            # that is never collected is the documented WebSocket session leak:
            # the session object stays in the table and memory grows until the
            # process dies. This bounds it at a known number of seconds.
            try:
                data = await asyncio.wait_for(
                    websocket.receive_bytes(),
                    timeout=STATE.limits.idle_timeout_s)
            except asyncio.TimeoutError:
                log.info("call %s idle for %.0fs; closing",
                         call_id, STATE.limits.idle_timeout_s)
                await websocket.close(code=1001)
                break

            if len(data) > STATE.limits.max_frame_bytes:
                # Refuse the frame, keep the call. A single oversized frame is
                # more likely a client bug than an attack, and dropping the call
                # for it would be its own denial of service.
                session.record_drop()
                await websocket.send_json({
                    "error": "frame too large",
                    "max_frame_bytes": STATE.limits.max_frame_bytes,
                })
                continue
            # Wire format is 16-bit signed PCM at 16 kHz, little-endian -- the
            # telephony standard, what a real SIP/WebRTC path delivers, and half
            # the bandwidth of float32. Reading int16 bytes as float32 (or the
            # reverse) does not error, it silently produces garbage audio, so
            # this is one format everywhere rather than a per-client guess.
            if len(data) % 2:
                session.record_drop()
                continue
            pcm = torch.from_numpy(
                np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)
            windows = session.push_audio(pcm)
            if not windows:
                continue

            fast = [w for w in windows if w.lane == "fast"]
            slow = [w for w in windows if w.lane == "slow"]

            # Channel detection, per call. Fast-lane windows only: the detector
            # was fitted on 1 s windows, and 3 s slow-lane windows would change
            # the noise-floor and packet-loss statistics it reads.
            call_calib = (STATE.calib_by_bucket.get(declared)
                          if declared else None)
            if detector is not None:
                for w in fast:
                    detector.update(w.samples)
                bucket = detector.bucket
                call_calib = STATE.calib_by_bucket.get(bucket)
                if bucket != last_bucket:
                    last_bucket = bucket
                    log.info("call %s channel -> %s", call_id, bucket)

            # Both paths run the GPU forward OFF the event loop. Running it
            # inline was the single biggest scaling defect in this service: one
            # synchronous forward blocks every other call's socket reads for its
            # duration, so N calls do not share the server, they queue behind
            # each other -- and the queue is invisible because each individual
            # forward still looks fast.
            try:
                if STATE.batcher is not None:
                    # Batch the raw forward across calls, then apply THIS
                    # call's channel calibration to its own rows.
                    raw_rows = await STATE.batcher.submit(fast)
                    events = []
                    for w, row in zip(fast, raw_rows):
                        events += STATE.scorer.events_from_raw(
                            [w], {b: v.unsqueeze(0) for b, v in row.items()},
                            call_calib)
                else:
                    events = await asyncio.to_thread(
                        STATE.scorer.score_fast, fast, call_calib)
                if slow:
                    events += await asyncio.to_thread(
                        STATE.scorer.score_slow, slow, enroll_emb)
            except BatcherOverloaded as exc:
                # Shed rather than queue. The caller is told, and the call stays
                # up so the next tick can succeed once the spike passes.
                session.record_drop(len(fast))
                await websocket.send_json({"warning": "overloaded",
                                           "detail": str(exc)})
                continue
            if not events:
                continue

            view, fired = session.apply_events(events)
            payload = session.to_json()

            # Prevention. Detection answers "is this synthetic"; this decides
            # what the call flow should DO about it, conditioned on what the
            # caller is attempting. Runs every update so the action tracks the
            # score rather than firing once at a threshold crossing.
            if view is not None:
                decision = STATE.policy.decide(
                    risk_nats=view.S_lo,
                    tier=session.state.tier,
                    confidence=view.confidence.score,
                    context=context,
                    identity_llr=view.llr_target,
                    enrolled=view.enrolled,
                )
                payload["prevention"] = decision.to_json()

                if decision.action is Action.CHALLENGE and call_id not in STATE.challenges:
                    ch = issue(call_id, ChallengeKind.DIGITS)
                    STATE.challenges[call_id] = ch
                    payload["challenge"] = {"prompt": ch.prompt,
                                            "kind": ch.kind.value}

                # One audit record per action change, not per chunk: a log that
                # writes twice a second per call is not reviewable evidence.
                if decision.action.value != last_action:
                    last_action = decision.action.value
                    try:
                        rec = build_record(
                            call_id=call_id, decision=decision, view=view,
                            challenge=STATE.challenges.get(call_id))
                    except ValueError as exc:
                        log.error("audit refused a record: %s", exc)
                    else:
                        # Off the event loop. append() opens a file and writes,
                        # and blocking file I/O on the loop stalls every other
                        # call exactly the way a blocking GPU forward does --
                        # less often, but on a path that runs during the most
                        # interesting moment of a call.
                        await asyncio.to_thread(STATE.audit.append, rec)
            # Peak risk, and the tier -- which lives on the session state, not
            # on the RiskView. `view.tier` was an AttributeError that killed
            # the call mid-stream, found by reading the history back and seeing
            # only one of three calls recorded.
            cur_tier = session.state.tier
            hist_max_risk = (view.S if hist_max_risk is None
                             else max(hist_max_risk, view.S))
            if cur_tier != hist_tier:
                hist_tier = cur_tier
                if STATE.store is not None:
                    # Transitions only. At 2 Hz per call and 64 calls, a row
                    # per chunk would be 128 rows/second of near-identical
                    # state; the transitions are what anyone reviewing a call
                    # actually reads.
                    await asyncio.to_thread(
                        STATE.store.add_event, call_id, "tier",
                        session.chunker.elapsed_s, cur_tier, view.S)
            if fired and hist_alert_at is None:
                hist_alert_at = session.chunker.elapsed_s
                if STATE.store is not None:
                    await asyncio.to_thread(
                        STATE.store.add_event, call_id, "alert",
                        hist_alert_at, cur_tier, view.S,
                        {"reason": fired[0].reason})
            if fired:
                payload["alerts"] = [
                    {"tier": c.tier, "direction": c.direction,
                     "at": round(c.at, 2), "reason": c.reason} for c in fired
                ]
            await websocket.send_json(payload)

    except WebSocketDisconnect:
        log.info("call %s disconnected after %.1fs",
                 call_id, session.chunker.elapsed_s)
    finally:
        # finally, not after the loop: an exception anywhere above must still
        # free the slot and the session, or capacity leaks one call at a time
        # until the service refuses everything.
        STATE.sessions.pop(call_id, None)
        STATE.challenges.pop(call_id, None)
        if STATE.store is not None:
            try:
                await asyncio.to_thread(
                    STATE.store.close_call, call_id, hist_max_risk,
                    hist_tier, hist_alert_at is not None, hist_alert_at,
                    last_action or "", STATE.notes)
            except Exception as exc:                          # noqa: BLE001
                log.error("could not close call history for %s: %s",
                          call_id, exc)
        await STATE.admission.release(client_key)


# ------------------------------------------------------------------ WebRTC
# Signalling for two-party browser calls. The media path does NOT come through
# here: WebRTC carries SRTP directly between the peers, so this server never
# sees the audio and could not decrypt it if it did. Each browser taps its own
# decoded remote track and streams it to /ws/{call_id} like any other source.
#
# That is worth being explicit about rather than letting the architecture
# diagram imply otherwise. VeriCall scores what a PARTICIPANT hears. It is not
# a network-side interceptor, and on this transport it cannot be one.
HUB = SignalingHub()


@app.websocket("/rtc/{room}")
async def signaling_endpoint(websocket: WebSocket, room: str) -> None:
    """Relay offer/answer/ICE between the two peers of one room.

    Mirrors /ws/{call_id}'s gating deliberately: accept first, then say why on
    refusal, so a client can distinguish a bad token from a full room and
    retry the one worth retrying. Closing before accepting drops the reason.
    """
    await websocket.accept()

    presented = (websocket.query_params.get("token")
                 or websocket.headers.get("x-vericall-token"))
    if not check_auth(presented):
        await websocket.send_json({"error": "unauthorized"})
        await websocket.close(code=CLOSE_UNAUTHORIZED)
        return

    if not valid_call_id(room):
        # Same rule as call_id: never echo an attacker-controlled string into
        # a log line. Length only.
        await websocket.send_json({"error": "invalid room"})
        await websocket.close(code=CLOSE_BAD_ROOM)
        log.warning("rejected signalling: malformed room (%d bytes)", len(room))
        return

    peer, refusal = await HUB.join(room, websocket)
    if peer is None:
        await websocket.send_json({"error": refusal})
        await websocket.close(code=CLOSE_ROOM_FULL if refusal == "room_full"
                              else CLOSE_TOO_MANY_ROOMS)
        return

    async def send_to_others(payload: dict) -> None:
        for other in await HUB.others(room, peer):
            try:
                await other.socket.send_json(payload)
            except Exception:                                 # noqa: BLE001
                # A peer that has gone away is not this peer's problem. Its
                # own handler will run its finally block and clean up.
                pass

    try:
        occupancy = await HUB.occupancy(room)
        await websocket.send_json({
            "type": "joined", "role": peer.role, "peers": occupancy,
            # The callee knows a caller is already here, so it can wait for an
            # offer; the caller learns of the callee via "peer-joined" below.
            "ice_servers": ICE_SERVERS,
        })
        if occupancy > 1:
            await send_to_others({"type": "peer-joined"})

        while True:
            raw = await websocket.receive_text()
            msg, why = validate_signal(raw)
            if msg is None:
                # Do not close the connection over one bad frame: a peer with
                # a buggy extension would become un-callable. Say no, continue.
                await websocket.send_json({"type": "rejected", "reason": why})
                continue
            await send_to_others(msg)

    except WebSocketDisconnect:
        pass
    except Exception as exc:                                  # noqa: BLE001
        log.warning("signalling error in room: %s", type(exc).__name__)
    finally:
        # finally, not after the loop: an exception above must still free the
        # room slot, or a two-seat room fills up permanently.
        for other in await HUB.leave(room, peer):
            try:
                await other.socket.send_json({"type": "peer-left"})
            except Exception:                                 # noqa: BLE001
                pass


WEB = ROOT / "web"
if WEB.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(WEB / "index.html"))

    @app.get("/call")
    def call_page() -> FileResponse:
        """The two-party demo. Separate page from `/`, which stays the
        single-stream detector: two working demos are better than one page
        with a mode switch that can be left in the wrong position."""
        return FileResponse(str(WEB / "call.html"))
