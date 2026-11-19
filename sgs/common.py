"""Primitives shared by sgs_simple (SGS) and solver_simple (GRPO / REINFORCE++ baselines).

    Solver prompt + verifier    re-exported from sgs/tasks.py (sorting)
    load_targets                sorting jsonl -> list of integer lists
    PatchedActorRolloutRefWorker  verl worker with the two fixes this env needs
    ray_runtime_env             verl's Ray env + repo on PYTHONPATH + HF/wandb vars
"""

import os
from pathlib import Path

from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.single_controller.base.decorator import Dispatch, register

# Sorting's Solver prompt + verifier live in sgs/tasks.py; re-exported for old callers.
from sgs.tasks import SOLVER_PROMPT, parse_sorted, solver_messages, user_msg  # noqa: F401


# =============================================================================
# Data
# =============================================================================

def load_targets(path):
    """Sorting target lists from a jsonl with a `list` field per row."""
    from sgs.tasks import Sorting
    return Sorting.load(path)


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
