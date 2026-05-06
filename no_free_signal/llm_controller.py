"""Per-creature LLM controller backed by Bedrock (Anthropic / Nova / Llama).

Critical architectural commitment: NEVER block the world tick loop.
`__call__(obs)` returns the most-recent cached (action, confidence) immediately
and conditionally enqueues a refresh on the controller's own daemon thread.
The world keeps stepping while the LLM thinks; actions lag by ~1 tick.
"""
from __future__ import annotations

import datetime as _dt
import os
import threading
import time
from collections import deque
from typing import Any, Callable, Optional

# Load .env on import so AWS creds and AWS_REGION come from the project file
# instead of whatever stale values the shell happens to have. Idempotent;
# does NOT override variables already set in the actual environment, so
# explicit shell overrides still win.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(override=True)
except ImportError:
    pass

from foresight.envs.unified_world import ACTION_NOOP, Creature
from foresight.evolution.genome import Genome
from no_free_signal.llm_prose import (
    ACTION_NAMES,
    genome_to_prose,
    obs_to_prose,
    parse_llm_response,
)
from no_free_signal.observation import RawObservation


# Default Bedrock id; override via NFS_BEDROCK_MODEL_ID.
# Bedrock requires cross-region inference profiles for on-demand invocation
# of newer Claude models — the regional prefix routes the request to the
# nearest active region. "us." for us-east-1 / us-west-2 etc.; if the
# AWS_REGION is in EU/APAC, override via env var.
DEFAULT_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# Globally shared daily call counter — naive in-memory budget guard.
_DAILY_LOCK = threading.Lock()
_DAILY_COUNT = 0
_DAILY_RESET_DATE = _dt.date.today()


def _today_count() -> int:
    """Reset the daily counter at UTC midnight."""
    global _DAILY_COUNT, _DAILY_RESET_DATE
    today = _dt.date.today()
    if today != _DAILY_RESET_DATE:
        _DAILY_COUNT = 0
        _DAILY_RESET_DATE = today
    return _DAILY_COUNT


def _daily_limit() -> int:
    try:
        return int(os.environ.get("NFS_LLM_DAILY_LIMIT", "500"))
    except ValueError:
        return 500


# Global runtime kill-switch. When False, ALL LLM refresh attempts no-op
# without ever reaching Bedrock. Belt-and-suspenders on top of: (a) pausing
# auto-step (which already prevents __call__ from firing), (b) detaching
# individual controllers, (c) the daily limit. Any one of these stops calls;
# this is the most explicit one.
_LLM_ENABLED_LOCK = threading.Lock()
_LLM_ENABLED: bool = True


def set_llm_enabled(value: bool) -> None:
    global _LLM_ENABLED
    with _LLM_ENABLED_LOCK:
        _LLM_ENABLED = bool(value)


def is_llm_enabled() -> bool:
    with _LLM_ENABLED_LOCK:
        return _LLM_ENABLED


# Pricing for the cost meter. Bedrock Haiku 4.5 ≈ $1/M input tokens,
# $5/M output tokens. Rough average per call: 250 input + 40 output =
# 0.000250 * 1 + 0.000040 * 5 = $0.00045/call.
USD_PER_CALL_ESTIMATE: float = 0.00045


# Auto-disable kill-switch: when N consecutive auth-class errors land
# across ALL controllers, flip the global enabled flag off so a stale
# token doesn't drain the daily budget on errors. Reset on first success.
_AUTH_FAIL_LOCK = threading.Lock()
_CONSECUTIVE_AUTH_FAILS: int = 0
_AUTO_DISABLE_THRESHOLD: int = 5
_LAST_DISABLE_REASON: str = ""


def get_last_disable_reason() -> str:
    with _AUTH_FAIL_LOCK:
        return _LAST_DISABLE_REASON


def _classify_auth_error(err_repr: str) -> bool:
    """Return True if the error looks like a credentials problem rather than
    a transient model error. We rollback the daily-counter increment for
    these (the call never reached the model) and count toward auto-disable."""
    needles = (
        "security token", "InvalidClientTokenId", "ExpiredToken",
        "ExpiredTokenException", "UnrecognizedClientException",
        "AccessDeniedException", "PermissionDenied",
        "401", "403",
    )
    return any(s in err_repr for s in needles)


def _record_auth_failure(reason: str) -> bool:
    """Increment the consecutive-fails counter; if threshold hit, flip the
    global LLM-enabled flag off and return True (caller can log it)."""
    global _CONSECUTIVE_AUTH_FAILS, _LAST_DISABLE_REASON
    with _AUTH_FAIL_LOCK:
        _CONSECUTIVE_AUTH_FAILS += 1
        if _CONSECUTIVE_AUTH_FAILS >= _AUTO_DISABLE_THRESHOLD:
            _LAST_DISABLE_REASON = (
                f"auto-disabled after {_CONSECUTIVE_AUTH_FAILS} consecutive "
                f"auth errors. Last: {reason[:200]}"
            )
            set_llm_enabled(False)
            _CONSECUTIVE_AUTH_FAILS = 0
            return True
    return False


def _record_auth_success() -> None:
    global _CONSECUTIVE_AUTH_FAILS, _LAST_DISABLE_REASON
    with _AUTH_FAIL_LOCK:
        _CONSECUTIVE_AUTH_FAILS = 0
        _LAST_DISABLE_REASON = ""


# ----------------------------------------------------------------------
# Bedrock cadence protections (per-process; experiment workers run as
# separate subprocesses so these counters are naturally per-run).
# ----------------------------------------------------------------------

# Visible accounting for what _refresh_in_flight silently swallowed,
# what 429s we hit, and how many retries we did. Surfaced into the
# run summary by the harness so a cadence audit can detect collapse.
_RUNTIME_LOCK = threading.Lock()
_SKIPPED_REFRESHES: int = 0
_THROTTLED_429S: int = 0
_THROTTLE_RETRIES: int = 0


def _bump_skipped_refresh() -> None:
    global _SKIPPED_REFRESHES
    with _RUNTIME_LOCK:
        _SKIPPED_REFRESHES += 1


def _bump_throttled() -> None:
    global _THROTTLED_429S
    with _RUNTIME_LOCK:
        _THROTTLED_429S += 1


def _bump_throttle_retry() -> None:
    global _THROTTLE_RETRIES
    with _RUNTIME_LOCK:
        _THROTTLE_RETRIES += 1


def get_runtime_counters() -> dict[str, int]:
    """Snapshot of cadence-protection counters for inclusion in a run
    summary. Reset at process start (per-subprocess in the parallel
    harness)."""
    with _RUNTIME_LOCK:
        return {
            "skipped_refreshes": int(_SKIPPED_REFRESHES),
            "throttled_429s": int(_THROTTLED_429S),
            "throttle_retries": int(_THROTTLE_RETRIES),
        }


# Global per-process token bucket. Default ~13 RPS = 780 RPM, well below
# the typical 1000 RPM Bedrock account cap. With 8 worker processes
# under one launcher, total request rate is bounded by the slower
# of (a) per-worker cadence (refresh_every_ticks * tick rate) and
# (b) this limiter — not by Bedrock's throttle response.
def _bedrock_rps_limit() -> float:
    try:
        return float(os.environ.get("NFS_BEDROCK_RPS", "13.0"))
    except ValueError:
        return 13.0


_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_NEXT_OK_AT: float = 0.0


def _acquire_rate_limit() -> None:
    """Block until the per-process pacing window allows another Bedrock
    call. Caller must hold no other locks (this can sleep for up to
    ~80ms at default 13 RPS)."""
    rps = _bedrock_rps_limit()
    if rps <= 0:
        return
    min_interval = 1.0 / rps
    while True:
        with _RATE_LIMIT_LOCK:
            now = time.monotonic()
            global _RATE_LIMIT_NEXT_OK_AT
            wait = _RATE_LIMIT_NEXT_OK_AT - now
            if wait <= 0:
                _RATE_LIMIT_NEXT_OK_AT = now + min_interval
                return
        time.sleep(min(wait, 0.5))


def _classify_throttle_error(err_repr: str) -> bool:
    """Return True if the error looks like a Bedrock 429 / throttle.
    Anthropic SDK surfaces `APIStatusError` with `status_code == 429`;
    boto3 surfaces `ThrottlingException` / `TooManyRequestsException`.
    String-match is robust across both shapes."""
    needles = (
        "ThrottlingException",
        "TooManyRequestsException",
        "Too Many Requests",
        "RateLimitError",
        "status_code: 429",
        "status_code=429",
        "429",
    )
    return any(s in err_repr for s in needles)


def has_aws_credentials() -> bool:
    """True if standard AWS env vars are available, OR boto3 can find them
    via the default chain (config file, etc.)."""
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return True
    try:
        import boto3  # noqa: WPS433
        sess = boto3.Session()
        creds = sess.get_credentials()
        return creds is not None
    except Exception:
        return False


_CLIENT_LOCK = threading.Lock()
_ANTHROPIC_CLIENT: Any = None
_BEDROCK_CLIENT: Any = None


def _bedrock_client(model_id: str) -> Any:
    """Return whichever client can talk to this model id.

    Anthropic models (``us.anthropic.*``) keep using the dedicated SDK
    because its native shape matches our existing call site exactly. Other
    Bedrock providers (Amazon Nova, Meta Llama) go through boto3's
    Converse API, wrapped by ``no_free_signal.bedrock_client.BedrockClient`` to
    expose the same ``.messages.create(...)`` surface."""
    global _ANTHROPIC_CLIENT, _BEDROCK_CLIENT
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if "anthropic" in model_id:
        if _ANTHROPIC_CLIENT is None:
            with _CLIENT_LOCK:
                if _ANTHROPIC_CLIENT is None:
                    from anthropic import AnthropicBedrock
                    kwargs = {"aws_region": region} if region else {}
                    _ANTHROPIC_CLIENT = AnthropicBedrock(**kwargs)
        return _ANTHROPIC_CLIENT
    if _BEDROCK_CLIENT is None:
        with _CLIENT_LOCK:
            if _BEDROCK_CLIENT is None:
                from no_free_signal.bedrock_client import BedrockClient
                _BEDROCK_CLIENT = BedrockClient(region=region)
    return _BEDROCK_CLIENT


SYSTEM_TEMPLATE = """You are inhabiting the body of a creature in a 2D ecology simulation.

Your innate biology (cannot change): {bio}

Your personality: {personality}

Each turn you will be told what you sense and you must pick exactly one action.
Available actions: {actions}.

You always emit an audio signal alongside your action. Audio is an 8-bin
frequency vector with NO fixed meaning — no shared dictionary, no labels.
Meaning is whatever the population converges on under selection pressure.
The shape is your contribution; the amplitude is how loud you call.

You MUST always include both `vocalize_wave` (8 numbers in 0..1) and
`vocalize_amp` (a number in 0..1) in your response.

- `vocalize_wave`: choose 8 numbers between 0 and 1 that reflect what you
  are responding to. Vary them across context — the same shape every turn
  carries no information. There is no "right answer"; pick what feels
  right for what you sense, and let evolution sort out which shapes
  matter.
- `vocalize_amp`: how loud — 0 means effectively silent (no energy spent,
  nobody hears), values up to 1 are progressively louder calls that cost
  more energy. Use higher amplitude when context seems worth signalling
  (danger, food, kin in distress); lower amplitude when nothing notable
  is happening.

Speech (the "say" field) is short and in your creature's voice — small
animal with drives, fears, instincts. Keep it under 80 characters. Emit
"" when you have nothing to say.

Respond ONLY with valid JSON, no other text:
{{"thought": "<one sentence on why>", "action": "<one of {actions}>", "confidence": <number 0..1>, "say": "<short utterance or empty string>", "vocalize_wave": [<8 numbers in 0..1>], "vocalize_amp": <number 0..1>}}"""


class LLMController:
    """One creature's LLM controller. Implements `ControllerFn = (obs)->int`."""

    # Tag identifying which driver kind this controller represents in the
    # vocal-driver resolver. Subclasses (e.g. RandomEmitterController) may
    # override to distinguish themselves from the API-driven default.
    driver_kind: str = "llm"

    def __init__(
        self,
        creature: Creature,
        personality: str,
        lambda_advisory: float = 1.0,
        refresh_every_ticks: int = 5,
        model_id: Optional[str] = None,
        max_log_entries: int = 20,
        world_ref: Any = None,  # weakref-able World instance for dialogue plumbing
        emit_logger: Optional[Callable[[int, int, list[float], float], None]] = None,
        wave_transform: Optional[Callable[[list[float]], list[float]]] = None,
    ):
        self._creature = creature
        self._genome: Genome = creature.genome
        self.personality = personality.strip() or "(no personality)"
        self.lambda_advisory = float(lambda_advisory)
        self.refresh_every_ticks = max(1, int(refresh_every_ticks))
        self.model_id = model_id or os.environ.get("NFS_BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
        self.creature_id = creature.individual_id
        self._world_ref = world_ref  # used for dialogue: broadcast / heard-by
        # Experiment hooks. ``emit_logger`` is called once per validated LLM
        # vocal emission (post-transform) with (tick, creature_id, wave, amp);
        # ``wave_transform`` rewrites the wave before caching/logging — used
        # by arm E (scrambled) to permute bins, and by arm F
        # (context-randomized) to substitute a replay-buffer wave.
        self._emit_logger = emit_logger
        self._wave_transform = wave_transform

        self._cached_action: int = ACTION_NOOP
        self._cached_confidence: float = 0.0
        self._cached_thought: str = "(thinking...)"
        self._cached_say: str = ""
        self._cached_vocal_wave: list[float] | None = None
        self._cached_vocal_amp: float = 0.0
        self._last_heard_step: int = 0
        self._cache_lock = threading.Lock()

        self._tick_count = 0
        self._last_refresh_tick = -10**9
        self._refresh_in_flight = False
        self._refresh_lock = threading.Lock()
        self._closed = threading.Event()

        # ring buffer of decision entries for the UI
        self._log: deque[dict] = deque(maxlen=max_log_entries)
        # ring buffer of recent prompts/personal events used in the prompt body
        self._personal_events: deque[dict] = deque(maxlen=10)

        # System prompt is fixed for this creature's lifetime.
        self._system = SYSTEM_TEMPLATE.format(
            bio=genome_to_prose(self._genome),
            personality=self.personality,
            actions="|".join(ACTION_NAMES),
        )

        # error state
        self.last_error: Optional[str] = None
        self.calls_made: int = 0

    # ------------------------------------------------------------------
    # ControllerFn surface
    # ------------------------------------------------------------------
    def __call__(self, obs: RawObservation) -> int:
        """Synchronous: returns the cached action immediately. Triggers a
        background refresh if it's time."""
        if self._closed.is_set():
            return ACTION_NOOP
        self._tick_count += 1
        if self._tick_count - self._last_refresh_tick >= self.refresh_every_ticks:
            self._maybe_kick_refresh(obs)
        with self._cache_lock:
            return int(self._cached_action)

    def current_advice(self) -> tuple[int, float]:
        """For advisory blending: returns (action, lambda * confidence). The
        brain's `choose_action_scores` adds this boost to the chosen action."""
        with self._cache_lock:
            return int(self._cached_action), self.lambda_advisory * self._cached_confidence

    def current_vocal_intent(self) -> tuple[list[float], float] | None:
        """Latest LLM-emitted (wave, amp) if any, else None. Consumed by the
        driver-resolution step in no_free_signal world.py. Cleared after one read so
        a stale wave doesn't fire repeatedly across refresh windows."""
        with self._cache_lock:
            wave = self._cached_vocal_wave
            amp = self._cached_vocal_amp
            self._cached_vocal_wave = None
            self._cached_vocal_amp = 0.0
        if wave is None or amp <= 0.0:
            return None
        return wave, amp

    def current_thought(self) -> dict[str, Any]:
        """Bulk-roster accessor — cheaper than `stats()` because it skips the
        global daily counter lookup."""
        with self._cache_lock:
            return {
                "thought": self._cached_thought,
                "action": ACTION_NAMES[self._cached_action],
                "confidence": self._cached_confidence,
                "say": self._cached_say,
                "calls_made": self.calls_made,
            }

    # ------------------------------------------------------------------
    # Internal: kick off a refresh
    # ------------------------------------------------------------------
    def _maybe_kick_refresh(self, obs: RawObservation) -> None:
        with self._refresh_lock:
            if self._closed.is_set():
                return
            if self._refresh_in_flight:
                # Visible accounting: a refresh kick fired but the
                # previous call hasn't returned yet. Advance
                # _last_refresh_tick so the gate doesn't re-fire every
                # subsequent tick during in-flight (which would inflate
                # the skip counter by ~refresh_every_ticks per real
                # missed window). With this update, one skip = one
                # refresh-window where the controller wanted to call
                # but couldn't, which is what the cadence audit reads.
                _bump_skipped_refresh()
                self._last_refresh_tick = self._tick_count
                return
            self._refresh_in_flight = True
            self._last_refresh_tick = self._tick_count
        # We need a *snapshot* of obs and the creature so the worker thread
        # doesn't read a half-updated state. The Creature dataclass is
        # mutable, so capture the small fields we need.
        c = self._creature
        snap = {
            "step": self._tick_count,
            "creature_id": c.individual_id,
            "pos": c.pos,
            "energy": float(c.energy),
            "fatigue": float(c.fatigue),
            "health": float(c.health),
            "inv_wood": int(c.inventory_wood),
            "inv_stone": int(c.inventory_stone),
            "obs": obs,
        }
        threading.Thread(
            target=self._do_refresh,
            args=(snap,),
            daemon=True,
            name=f"llm-refresh-c{self.creature_id}",
        ).start()

    def _do_refresh(self, snap: dict) -> None:
        try:
            # Global kill-switch — silently no-op if user has disabled LLMs.
            if not is_llm_enabled():
                self._record_log({
                    "step": snap["step"],
                    "kind": "llm_disabled",
                    "thought": "(LLM runtime disabled by user)",
                    "action": ACTION_NAMES[ACTION_NOOP],
                    "confidence": 0.0,
                    "ms": 0,
                })
                return
            # Daily-limit guard
            global _DAILY_COUNT
            with _DAILY_LOCK:
                if _today_count() >= _daily_limit():
                    self._record_log({
                        "step": snap["step"],
                        "kind": "daily_limit_reached",
                        "thought": "(daily LLM call limit reached)",
                        "action": ACTION_NAMES[ACTION_NOOP],
                        "confidence": 0.0,
                        "ms": 0,
                    })
                    return
                _DAILY_COUNT += 1
                self.calls_made += 1

            env = getattr(self._world_ref, "_env", None) if self._world_ref else None
            user_msg = obs_to_prose(
                snap["obs"], self._creature, env=env,
                recent_personal_events=list(self._personal_events),
            )
            # Append any utterances heard since last refresh (dialogue layer).
            if self._world_ref is not None:
                try:
                    heard = self._world_ref.utterances_heard_by(
                        self.creature_id, since_step=self._last_heard_step,
                    )
                except Exception:
                    heard = []
                if heard:
                    overheard = "; ".join(
                        f'c{u["speaker_id"]} said "{u["text"]}"' for u in heard[-5:]
                    )
                    user_msg += f" You overheard: {overheard}."
                    self._last_heard_step = max(int(u["step"]) for u in heard)
            t0 = time.monotonic()
            client = _bedrock_client(self.model_id)
            text: str | None = None
            err_repr = ""
            # Bounded retry on 429/ThrottlingException with exponential
            # backoff capped at 5s. Non-throttle exceptions exit the loop
            # immediately and fall through to the auth/error path. We
            # rate-limit BEFORE each attempt (including retries) so retries
            # don't pile on an already-hot account.
            max_attempts = 5
            for attempt in range(max_attempts):
                _acquire_rate_limit()
                try:
                    resp = client.messages.create(
                        model=self.model_id,
                        max_tokens=200,
                        temperature=0.7,
                        system=self._system,
                        messages=[{"role": "user", "content": user_msg}],
                    )
                    text = resp.content[0].text if resp.content else ""
                    err_repr = ""
                    break
                except Exception as e:
                    err_repr = repr(e)
                    if _classify_throttle_error(err_repr) and attempt + 1 < max_attempts:
                        _bump_throttled()
                        _bump_throttle_retry()
                        backoff = min(0.25 * (2 ** attempt), 5.0)
                        self._record_log({
                            "step": snap["step"],
                            "kind": "llm_throttled",
                            "thought": (
                                f"(Bedrock 429 attempt {attempt + 1}/"
                                f"{max_attempts}; backoff {backoff:.2f}s)"
                            ),
                            "action": ACTION_NAMES[ACTION_NOOP],
                            "confidence": 0.0,
                            "ms": int((time.monotonic() - t0) * 1000),
                        })
                        time.sleep(backoff)
                        continue
                    if _classify_throttle_error(err_repr):
                        # Final 429 — exhausted retries. Count it and fall
                        # through to the error path so it lands in the run
                        # log (cadence audit will read this).
                        _bump_throttled()
                    break
            if text is None:
                e = Exception(err_repr) if err_repr else Exception("(unknown)")
                self.last_error = err_repr
                # Auth-class errors didn't actually reach the model — refund
                # the budget increment so a stale token can't drain the day.
                # (`global _DAILY_COUNT` is already declared earlier in this
                # function for the per-call increment; redeclaring is a
                # SyntaxError because of the prior assignment.)
                if _classify_auth_error(err_repr):
                    with _DAILY_LOCK:
                        _DAILY_COUNT = max(0, _DAILY_COUNT - 1)
                    self.calls_made = max(0, self.calls_made - 1)
                    auto_disabled = _record_auth_failure(err_repr)
                    if auto_disabled:
                        self._record_log({
                            "step": snap["step"],
                            "kind": "llm_auto_disabled",
                            "thought": (
                                "(LLM runtime auto-disabled — credentials "
                                "rejected by AWS. Refresh creds + flip the "
                                "runtime ON in Settings to retry.)"
                            ),
                            "action": ACTION_NAMES[ACTION_NOOP],
                            "confidence": 0.0,
                            "ms": int((time.monotonic() - t0) * 1000),
                        })
                        return
                self._record_log({
                    "step": snap["step"],
                    "kind": "llm_error",
                    "thought": f"(error: {err_repr[:200]})",
                    "action": ACTION_NAMES[ACTION_NOOP],
                    "confidence": 0.0,
                    "ms": int((time.monotonic() - t0) * 1000),
                })
                return
            # Success — reset the consecutive-fails counter.
            _record_auth_success()
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            parsed = parse_llm_response(text)
            say = (parsed.get("say") or "")[:80].strip()
            wave = parsed.get("vocal_wave")
            amp = float(parsed.get("vocal_amp", 0.0))
            # Apply per-controller wave transform (scramble / replay) before
            # caching so what gets emitted matches what gets logged.
            if wave is not None and self._wave_transform is not None:
                try:
                    wave = self._wave_transform(list(wave))
                except Exception:
                    wave = None
            # Emission log for the experiment harness.
            if wave is not None and amp > 0.0 and self._emit_logger is not None:
                try:
                    self._emit_logger(snap["step"], self.creature_id, list(wave), amp)
                except Exception:
                    pass
            with self._cache_lock:
                self._cached_action = parsed["action_int"]
                self._cached_confidence = parsed["confidence"]
                self._cached_thought = parsed["thought"]
                self._cached_say = say
                self._cached_vocal_wave = wave
                self._cached_vocal_amp = amp
            self._record_log({
                "step": snap["step"],
                "kind": "decision",
                "thought": parsed["thought"],
                "action": parsed["action"],
                "confidence": parsed["confidence"],
                "say": say,
                "ms": elapsed_ms,
            })
            # Broadcast utterance to nearby creatures.
            if say and self._world_ref is not None:
                try:
                    self._world_ref.broadcast_utterance(self.creature_id, say)
                except Exception:
                    pass
        finally:
            with self._refresh_lock:
                self._refresh_in_flight = False

    def _record_log(self, entry: dict) -> None:
        self._log.append(entry)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._closed.set()

    def get_log(self, limit: int = 20) -> list[dict]:
        return list(self._log)[-limit:]

    def stats(self) -> dict:
        return {
            "creature_id": self.creature_id,
            "personality": self.personality,
            "lambda_advisory": self.lambda_advisory,
            "refresh_every_ticks": self.refresh_every_ticks,
            "model_id": self.model_id,
            "calls_made": self.calls_made,
            "daily_total": _today_count(),
            "daily_limit": _daily_limit(),
            "last_error": self.last_error,
            "last_action": ACTION_NAMES[self._cached_action],
            "last_confidence": self._cached_confidence,
            "last_thought": self._cached_thought,
        }
