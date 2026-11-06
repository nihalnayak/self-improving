"""Primitives shared by sgs_simple (SGS) and grpo_simple (GRPO baseline).

    Solver prompt + verifier    misc/sort_benchmark.py prompt and answer parser
    load_targets                sorting jsonl -> list of integer lists
    PatchedActorRolloutRefWorker  verl worker with the two fixes this env needs
    ray_runtime_env             verl's Ray env + repo on PYTHONPATH + HF/wandb vars
"""

import os
import re
from pathlib import Path

from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.single_controller.base.decorator import Dispatch, register


# =============================================================================
# Solver prompt + verifier
# =============================================================================

# Solver: verbatim from misc/sort_benchmark.py.
SOLVER_PROMPT = (
    "Sort the numbers in the list in ascending order.\n\n"
    "List: {list_str}\n\n"
    "Return only the sorted list on a single line, enclosed in \\boxed{{}}, "
    "e.g. \\boxed{{[1, 2, 3]}}."
)


def user_msg(text):
    """One-turn chat. verl applies the chat template to this, never us."""
    return [{"role": "user", "content": text}]


def solver_messages(xs):
    return user_msg(SOLVER_PROMPT.format(list_str=str(xs)))


_BOXED_RE = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)
_BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")
_NUM_RE = re.compile(r"-?\d+")


def _ints(s):
    nums = _NUM_RE.findall(s)
    return [int(n) for n in nums] if nums else None


def parse_sorted(text):
    """Solver answer: first \\boxed{...}, else last [...] (misc/sort_benchmark.py)."""
    m = _BOXED_RE.search(text)
    if m:
        return _ints(m.group(1))
    brackets = _BRACKET_RE.findall(text)
    return _ints(brackets[-1]) if brackets else None


# =============================================================================
# Data
# =============================================================================

def load_targets(path):
    """Target lists from a jsonl with a `list` field per row."""
    from datasets import load_dataset
    return [list(row["list"]) for row in load_dataset("json", data_files=path, split="train")]


# =============================================================================
# verl plumbing
# =============================================================================

class PatchedActorRolloutRefWorker(ActorRolloutRefWorker):
    """ActorRolloutRefWorker with two env fixes (trl 1.16, vLLM tied embeddings)."""

    def __init__(self, config, role, **kwargs):
        super().__init__(config, role, **kwargs)
        # verl 0.9.1's monkey patch imports trl's PPO value-head class, which
        # trl 1.16 no longer has. It only matters for critic/value models, so skip it.
        import verl.models.transformers.monkey_patch as verl_monkey_patch
        verl_monkey_patch.is_trl_available = lambda: False

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        # verl loads weights into vLLM one bucket at a time, and vLLM >= 0.31
        # raises if a bucket has `lm_head.weight` (tied alias) without
        # `model.embed_tokens.weight`. The receiver dies, the sender waits for
        # its ack forever, and the job hangs. vLLM shares one Parameter for
        # both names, so sending embed_tokens alone updates lm_head too.
        from transformers import AutoConfig
        from verl.workers.rollout.utils import ensure_async_iterator
        if AutoConfig.from_pretrained(self.config.model.path).tie_word_embeddings:
            update_weights = self.rollout.update_weights

            async def drop_tied_lm_head(weights):
                async for name, w in ensure_async_iterator(weights):
                    if name != "lm_head.weight":
                        yield name, w

            self.rollout.update_weights = lambda weights, **kw: update_weights(
                drop_tied_lm_head(weights), **kw)


def ray_runtime_env():
    """verl's PPO Ray runtime env, plus the repo on PYTHONPATH and HF/wandb vars."""
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    repo_root = str(Path(__file__).resolve().parents[1])
    runtime_env = get_ppo_ray_runtime_env()
    runtime_env["env_vars"].update({
        "PYTHONPATH": repo_root + os.pathsep + os.environ.get("PYTHONPATH", ""),
        **{k: os.environ[k] for k in ("HF_HOME", "WANDB_API_KEY") if k in os.environ},
    })
    return runtime_env
