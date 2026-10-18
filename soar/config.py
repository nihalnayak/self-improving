from dataclasses import dataclass


@dataclass
class SoarConfig:
    # ---- model ----
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")

    # ---- data ----
    # Dir produced by `python -m misc.build_fail128`. Contains train.jsonl / test.jsonl / meta.json.
    fail128_dir: str = "files/data/math_fail128"
    n_reward_questions: int = 32        # |Q_R|: paper uses 64
    seed: int = 0

    # ---- outer loop (teacher) ----
    n_outer_steps: int = 50             # paper uses 200
    outer_group_size: int = 4           # g: datasets per outer step (paper uses up to 8)
    inner_dataset_size: int = 16        # n: problems per dataset X_k (paper uses 64)
    r_parallel_students: int = 1        # r: parallel student runs per dataset (paper uses 4)
    teacher_lr: float = 1e-5
    teacher_max_new_tokens: int = 512
    teacher_temperature: float = 1.0
    teacher_rejection_cap: int = 8      # safety cap on rejection sampling multiplier

    # ---- inner loop (student) ----
    inner_steps: int = 10               # K: student RLOO steps per X_k (paper)
    student_rollouts: int = 8           # N: completions per prompt (paper's batch size 8)
    student_lr: float = 1e-5
    student_max_new_tokens: int = 1024  # matches fail@128 token budget
    student_temperature: float = 1.0

    # ---- rl ----
    kl_beta: float = 0.02
    max_prompt_len: int = 1024

    # ---- promotion ----
    promotion_threshold: float = 0.01   # τ (paper)
    promotion_window: int = 3           # m (paper): moving-average window for teacher reward
    max_promotions_to_track: int = 4    # paper observes a max of 4 promotions in practice

    # ---- vllm ----
    vllm_gpu_memory_utilization: float = 0.35
    vllm_max_model_len: int = 2048
    vllm_max_loras: int = 2

    # ---- eval / logging ----
    wandb_project: str = "soar-mini"
    wandb_mode: str = "online"
    out_dir: str = "runs/soar-v1"
    log_every: int = 1
    ckpt_every: int = 10


def smoke_config() -> SoarConfig:
    """Tiny config for end-to-end smoke test."""
    return SoarConfig(
        n_outer_steps=1,
        outer_group_size=2,
        inner_dataset_size=2,
        r_parallel_students=1,
        inner_steps=1,
        student_rollouts=2,
        n_reward_questions=4,
        teacher_max_new_tokens=256,
        student_max_new_tokens=256,
        wandb_mode="disabled",
        out_dir="runs/smoke",
    )
