"""
One-time data preparation for autoresearch experiments — multi-GPU (DDP) variant.
Downloads data shards and trains a BPE tokenizer.

Usage:
    python prepare.py                  # full prep (download + tokenizer)
    python prepare.py --num-shards 8   # download only 8 shards (for testing)

Data and tokenizer are stored in ~/.cache/autoresearch/ (shared with the
single-GPU autoresearch-baseline task).

Multi-GPU contract
------------------
This variant trains one candidate across WORLD_SIZE GPUs with DDP. The world
size is a fixed task constant (default 4); scores are only comparable at the
same world size. The framework-facing contract is unchanged:
`evaluate_config(make_model, params) -> float` (plus the two no-score probes)
keeps its signature and its 1-call = 1-objective-slot accounting; when
WORLD_SIZE > 1 the function internally launches a torchrun process group and
returns rank 0's result.

Environment overrides (regression/debug ONLY — never during experiments):
    AUTORESEARCH_DDP_WORLD_SIZE  — override WORLD_SIZE (e.g. 1 on a 1-GPU box)
    AUTORESEARCH_DDP_TORCHRUN=1  — force the torchrun path even at world size 1,
                                   to exercise the spawn machinery on one GPU.
"""

import os
import sys
import json
import time
import math
import argparse
import pickle
import inspect
import tempfile
import subprocess
from multiprocessing import Pool

import requests
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048       # context length
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_TOKENS = 40 * 524288  # number of tokens for val eval

# Fixed task constant: one evaluation trains across this many GPUs with DDP.
# Scores are comparable only at equal world size. The env override exists for
# single-GPU regression of this file's machinery, not for experiments.
WORLD_SIZE = int(os.environ.get("AUTORESEARCH_DDP_WORLD_SIZE", "4"))
# Regression hook: exercise the torchrun spawn path even when world size is 1.
FORCE_TORCHRUN = os.environ.get("AUTORESEARCH_DDP_TORCHRUN", "") == "1"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542 # the last datashard is shard_06542.parquet
VAL_SHARD = MAX_SHARD  # pinned validation shard (shard_06542)
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
VOCAB_SIZE = 8192

# BPE split pattern (GPT-4 style, with \p{N}{1,2} instead of {1,3})
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_single_shard(index):
    """Download one parquet shard with retries. Returns True on success."""
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return True

    url = f"{BASE_URL}/{filename}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"  Downloaded {filename}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def download_data(num_shards, download_workers=8):
    """Download training shards + pinned validation shard."""
    os.makedirs(DATA_DIR, exist_ok=True)
    num_train = min(num_shards, MAX_SHARD)
    ids = list(range(num_train))
    if VAL_SHARD not in ids:
        ids.append(VAL_SHARD)

    # Count what's already downloaded
    existing = sum(1 for i in ids if os.path.exists(os.path.join(DATA_DIR, f"shard_{i:05d}.parquet")))
    if existing == len(ids):
        print(f"Data: all {len(ids)} shards already downloaded at {DATA_DIR}")
        return

    needed = len(ids) - existing
    print(f"Data: downloading {needed} shards ({existing} already exist)...")

    workers = max(1, min(download_workers, needed))
    with Pool(processes=workers) as pool:
        results = pool.map(download_single_shard, ids)

    ok = sum(1 for r in results if r)
    print(f"Data: {ok}/{len(ids)} shards ready at {DATA_DIR}")

# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def list_parquet_files():
    """Return sorted list of parquet file paths in the data directory."""
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
    """Yield documents from training split (all shards except pinned val shard)."""
    parquet_paths = [p for p in list_parquet_files() if not p.endswith(VAL_FILENAME)]
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer():
    """Train BPE tokenizer using rustbpe, save as tiktoken pickle."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    parquet_files = list_parquet_files()
    if len(parquet_files) < 2:
        print("Tokenizer: need at least 2 data shards (1 train + 1 val). Download more data first.")
        sys.exit(1)

    # --- Train with rustbpe ---
    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    # Build tiktoken encoding from trained merges
    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    # Save tokenizer
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    # --- Build token_bytes lookup for BPB evaluation ---
    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check
    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper. Training is handled above."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _document_batches(split, tokenizer_batch_size=128, rank=0, world_size=1):
    """Infinite iterator over document batches from parquet files.

    With world_size > 1 the stream is deterministically sharded: rank r yields
    only the document batches whose sequence index is congruent to r mod
    world_size, so DDP ranks consume disjoint documents in a fixed order.
    """
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare.py first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
        assert len(parquet_paths) > 0, "No training shards found."
    else:
        parquet_paths = [val_path]
    epoch = 1
    batch_idx = 0
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    if batch_idx % world_size == rank:
                        yield batch[i:i+tokenizer_batch_size], epoch
                    batch_idx += 1
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000, rank=0, world_size=1):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).

    rank/world_size select a deterministic disjoint shard of the document
    stream (used by DDP ranks); with the defaults (0, 1) the stream is
    identical to the single-GPU task.
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split, rank=rank, world_size=world_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    # Pre-allocate buffers: [inputs (B*T) | targets (B*T)]
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.

    Runs on the calling rank only. In a DDP run the candidate calls this on
    rank 0 with the unwrapped model and broadcasts the result.
    """
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# The single config -> score evaluation
# ---------------------------------------------------------------------------

class PretrainEnv:
    """The fixed task environment handed to a candidate's `make_model`.

    Everything a candidate may use lives here: the fixed tokenizer, the fixed
    dataloader factory, the fixed metric, and the fixed run constants. The
    training-time budget (`train_budget_seconds`) is self-enforced by the
    candidate's training loop exactly as in the standalone script (counted
    after warmup steps; startup and compilation excluded).

    Multi-GPU: `rank`/`world_size` identify this process's DDP rank. The bound
    `make_dataloader` automatically serves this rank's deterministic disjoint
    shard of the training stream, so candidate code calls it exactly as in the
    single-GPU task. `evaluate_bpb` is rank-agnostic: the candidate calls it
    on rank 0 only and broadcasts the scalar."""

    def __init__(self, rank: int = 0, world_size: int = 1):
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.tokenizer = Tokenizer.from_directory()
        self.evaluate_bpb = evaluate_bpb
        self.max_seq_len = MAX_SEQ_LEN
        self.train_budget_seconds = TIME_BUDGET
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.device = "cuda"
        self.seed = 42

        def sharded_dataloader(tokenizer, B, T, split, *args, **kwargs):
            return make_dataloader(
                tokenizer, B, T, split, *args,
                rank=self.rank, world_size=self.world_size, **kwargs,
            )

        self.make_dataloader = sharded_dataloader


class PreflightEnv(PretrainEnv):
    """Candidate environment with validation access mechanically disabled.

    Two modes, both no-score:

    * default (`resource_probe=False`) — the correctness check. The candidate
      trains whatever first-step shape `run()` would use, which is what TASK.md
      asks `preflight()` to mirror.
    * `resource_probe=True` — the memory-envelope check. `T` is forced to
      `MAX_SEQ_LEN` so the probe measures the worst-case training shape rather
      than whichever shape the candidate happens to start with. A candidate
      whose `run()` ramps sequence length (a curriculum) peaks far above its
      own first step, and the framework's space clamp needs the upper bound,
      not the opening one.

    Both modes record the largest `(B, T)` actually requested so the caller can
    tell whether the observed peak covers the worst case. Rank sharding is
    inherited from PretrainEnv; the T-pin applies identically on every rank.
    """

    def __init__(self, rank: int = 0, world_size: int = 1, resource_probe: bool = False):
        super().__init__(rank=rank, world_size=world_size)
        self.resource_probe = bool(resource_probe)
        self.max_requested_batch = 0
        self.max_requested_seq_len = 0

        bound_dataloader = self.make_dataloader

        def train_only_dataloader(tokenizer, B, T, split, *args, **kwargs):
            if split != "train":
                raise RuntimeError(
                    "candidate preflight may only request the training split"
                )
            if self.resource_probe:
                T = self.max_seq_len
            self.max_requested_batch = max(self.max_requested_batch, int(B))
            self.max_requested_seq_len = max(self.max_requested_seq_len, int(T))
            return bound_dataloader(tokenizer, B, T, split, *args, **kwargs)

        def validation_disabled(*_args, **_kwargs):
            raise RuntimeError(
                "validation is disabled inside candidate preflight"
            )

        self.make_dataloader = train_only_dataloader
        self.evaluate_bpb = validation_disabled


def preflight_environment() -> dict:
    """Validate fixed runtime resources without constructing a candidate."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the CUDA device does not support bfloat16")

    device_count = torch.cuda.device_count()
    if device_count < WORLD_SIZE:
        raise RuntimeError(
            f"this task requires {WORLD_SIZE} CUDA devices "
            f"(prepare.WORLD_SIZE), found {device_count}"
        )

    tokenizer = Tokenizer.from_directory()
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    if not os.path.isfile(token_bytes_path):
        raise FileNotFoundError(f"missing tokenizer byte table: {token_bytes_path}")
    parquet_paths = list_parquet_files()
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    train_paths = [path for path in parquet_paths if path != val_path]
    if not os.path.isfile(val_path):
        raise FileNotFoundError(f"missing pinned validation shard: {val_path}")
    if not train_paths:
        raise FileNotFoundError(f"no training shards found under {DATA_DIR}")

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    properties = torch.cuda.get_device_properties(0)
    return {
        "device": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "device_count": device_count,
        "world_size": WORLD_SIZE,
        "total_vram_mb": round(total_bytes / 1024 / 1024, 1),
        "free_vram_mb": round(free_bytes / 1024 / 1024, 1),
        "bf16": True,
        "vocab_size": tokenizer.get_vocab_size(),
        "training_shards": len(train_paths),
        "validation_shard": VAL_FILENAME,
    }


def _probe_envelope(env, result: dict) -> dict:
    """Attach the self-describing measurement envelope to a probe result."""
    probe_seq_len = env.max_requested_seq_len or None
    return {
        "status": "ok",
        "objective_calls": 0,
        **result,
        # Self-describing envelope: what shape this peak was actually measured
        # at, so a caller need not assume the probe covered the worst case.
        "probe_seq_len": probe_seq_len,
        "probe_batch_size": env.max_requested_batch or None,
        "max_seq_len": env.max_seq_len,
        "envelope_covers_worst_case": (
            None if probe_seq_len is None else probe_seq_len >= env.max_seq_len
        ),
    }


def _run_no_score_probe(make_model, params: dict, *, resource_probe: bool) -> dict:
    """Shared body of the two no-score probes; never calls score_fn."""
    if WORLD_SIZE > 1 or FORCE_TORCHRUN:
        mode = "resource_probe" if resource_probe else "preflight"
        return _launch_ddp(mode, make_model, params)
    env = PreflightEnv(resource_probe=resource_probe)
    trainer = make_model(env, params)
    preflight = getattr(trainer, "preflight", None)
    if not callable(preflight):
        raise TypeError(
            "candidate trainer must expose preflight() for the fixed no-score check"
        )
    result = preflight()
    if result is None:
        result = {}
    if not isinstance(result, dict):
        raise TypeError("trainer.preflight() must return a dict or None")
    return _probe_envelope(env, result)


def preflight_config(make_model, params: dict) -> dict:
    """Exercise candidate construction + one train step, never validation."""
    return _run_no_score_probe(make_model, params, resource_probe=False)


def resource_probe_config(make_model, params: dict) -> dict:
    """Measure the worst-case training-shape memory envelope, never validation.

    Same no-score contract as `preflight_config` — construction plus one real
    training step, no `evaluate_bpb`, no score — but with `T` pinned to
    `MAX_SEQ_LEN`. This is the framework's resource oracle: the search-space
    clamp needs a peak that bounds the whole run, and a candidate's own first
    step does not provide one when `run()` ramps sequence length. Under DDP the
    reported peak is the max across ranks.

    Known limit: the envelope covers sequence length only. A candidate that
    micro-batches inside its own `preflight()` can still report a peak below
    what its full run reaches.
    """
    return _run_no_score_probe(make_model, params, resource_probe=True)


# ---------------------------------------------------------------------------
# DDP orchestration (internal)
# ---------------------------------------------------------------------------

def _candidate_module_path(make_model) -> str:
    """Locate the source file that defines the candidate's make_model."""
    path = inspect.getsourcefile(make_model)
    if path is None:
        module = sys.modules.get(make_model.__module__)
        path = getattr(module, "__file__", None) if module else None
    if path is None:
        raise RuntimeError("cannot locate the candidate module defining make_model")
    return os.path.abspath(path)


def _launch_ddp(mode: str, make_model, params: dict):
    """Run one no-score probe or full evaluation across WORLD_SIZE GPUs.

    Launches this file as a torchrun worker group (`--ddp-worker`), waits for
    it, and returns rank 0's result. A non-zero worker exit is an honest
    failure: it raises here so the eval is recorded as a crash.

    The candidate's exact source bytes are snapshotted into the launch tmpdir
    and the workers execute that snapshot, so every rank runs the same bytes
    the caller resolved, immune to a candidate file being rewritten (or its
    .pyc going stale) between admission and worker execution.
    """
    candidate_path = _candidate_module_path(make_model)
    this_file = os.path.abspath(__file__)
    with tempfile.TemporaryDirectory(prefix="autoresearch-ddp-") as tmpdir:
        params_file = os.path.join(tmpdir, "params.json")
        out_file = os.path.join(tmpdir, "result.json")
        snapshot_file = os.path.join(tmpdir, "candidate_snapshot.py")
        with open(candidate_path, "rb") as f:
            source = f.read()
        with open(snapshot_file, "wb") as f:
            f.write(source)
        with open(params_file, "w") as f:
            json.dump(params or {}, f)
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", f"--nproc_per_node={WORLD_SIZE}",
            this_file, "--ddp-worker",
            "--mode", mode,
            "--candidate", snapshot_file,
            "--params-file", params_file,
            "--out-file", out_file,
        ]
        proc = subprocess.run(cmd, cwd=os.path.dirname(this_file))
        if proc.returncode != 0:
            raise RuntimeError(
                f"DDP worker group failed (mode={mode}, exit={proc.returncode})"
            )
        with open(out_file) as f:
            return json.load(f)


def _load_worker_module(name: str, path: str):
    """Execute the exact source bytes at `path`, never an mtime-based .pyc.

    Mirrors the canonical candidate loader semantics
    (tools/tuners/_common.py `_load_source_module`): the module is registered
    in sys.modules BEFORE execution — dataclass/typing resolution looks up
    `sys.modules[cls.__module__]` at decoration time and fails on unregistered
    modules — and the registration is rolled back if execution raises.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ImportError(f"cannot create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        with open(path, "rb") as f:
            code = compile(f.read(), path, "exec")
        exec(code, module.__dict__)
    except BaseException:
        if sys.modules.get(name) is module:
            sys.modules.pop(name, None)
        raise
    return module


def _ddp_worker_main(args) -> None:
    """ torchrun worker entry: build the rank-bound env and run the candidate."""
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    with open(args.params_file) as f:
        params = json.load(f)

    module = _load_worker_module("autoresearch_candidate", args.candidate)

    if args.mode == "evaluate":
        env = PretrainEnv(rank=local_rank, world_size=world_size)
        trainer = module.make_model(env, params)
        val_bpb = float(trainer.run())
        if not math.isfinite(val_bpb):
            raise ValueError(f"trainer returned non-finite val_bpb: {val_bpb!r}")
        if local_rank == 0:
            with open(args.out_file, "w") as f:
                json.dump({"val_bpb": val_bpb}, f)
    else:
        env = PreflightEnv(
            rank=local_rank, world_size=world_size,
            resource_probe=(args.mode == "resource_probe"),
        )
        trainer = module.make_model(env, params)
        preflight = getattr(trainer, "preflight", None)
        if not callable(preflight):
            raise TypeError(
                "candidate trainer must expose preflight() for the fixed no-score check"
            )
        result = preflight()
        if result is None:
            result = {}
        if not isinstance(result, dict):
            raise TypeError("trainer.preflight() must return a dict or None")
        # The probe is the framework's memory oracle: report the worst rank,
        # not just rank 0. Done in the fixed surface so candidates cannot
        # under-report by measuring only their own device.
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and "peak_vram_mb" in result:
            peak = torch.tensor([float(result["peak_vram_mb"])], device="cuda")
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            result["peak_vram_mb"] = round(float(peak.item()), 1)
        if local_rank == 0:
            with open(args.out_file, "w") as f:
                json.dump(_probe_envelope(env, result), f)


def evaluate_config(make_model, params: dict) -> float:
    """The single `config -> score` evaluation (lower is better).

    Builds the fixed PretrainEnv, constructs the candidate's trainer via
    `make_model(env, params)`, and runs one full budgeted training run. The
    trainer's returned post-training val_bpb IS the candidate's score — there
    is no separate official run (warm-start eval and Phase-C tuning both call
    this). A non-finite result raises so the eval is recorded as a crash.

    With WORLD_SIZE > 1 the run executes as a WORLD_SIZE-rank DDP job launched
    via torchrun; rank 0's val_bpb is the score. Accounting is unchanged: one
    call consumes one objective slot regardless of world size.
    """
    if WORLD_SIZE > 1 or FORCE_TORCHRUN:
        if torch.cuda.device_count() < WORLD_SIZE:
            raise RuntimeError(
                f"this task requires {WORLD_SIZE} CUDA devices "
                f"(prepare.WORLD_SIZE), found {torch.cuda.device_count()}"
            )
        result = _launch_ddp("evaluate", make_model, params)
        val_bpb = float(result["val_bpb"])
    else:
        env = PretrainEnv()
        trainer = make_model(env, params)
        val_bpb = float(trainer.run())
    if not math.isfinite(val_bpb):
        raise ValueError(f"trainer returned non-finite val_bpb: {val_bpb!r}")
    return val_bpb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for autoresearch (DDP variant)")
    parser.add_argument("--num-shards", type=int, default=10, help="Number of training shards to download (-1 = all). Val shard is always pinned.")
    parser.add_argument("--download-workers", type=int, default=8, help="Number of parallel download workers")
    # Internal torchrun worker entry (launched by _launch_ddp; not for users).
    parser.add_argument("--ddp-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=["evaluate", "preflight", "resource_probe"], help=argparse.SUPPRESS)
    parser.add_argument("--candidate", help=argparse.SUPPRESS)
    parser.add_argument("--params-file", help=argparse.SUPPRESS)
    parser.add_argument("--out-file", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.ddp_worker:
        _ddp_worker_main(args)
        sys.exit(0)

    num_shards = MAX_SHARD if args.num_shards == -1 else args.num_shards

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download data
    download_data(num_shards, download_workers=args.download_workers)
    print()

    # Step 2: Train tokenizer
    train_tokenizer()
    print()
    print("Done! Ready to train")
