from dataclasses import dataclass


@dataclass
class SoarConfig:
    # ---- model ----
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    use_lora: bool = False  # train_trl.py: wrap student with PEFT (ref = base w/ adapter disabled)

    # ---- data ----
    # Dir produced by `python -m misc.build_fail128`. Contains train.jsonl / test.jsonl / meta.json.
    fail128_dir: str = "files/data/math_fail128"
    n_reward_questions: int = 64        # |Q_R|: paper uses 64
    seed: int = 0

    # ---- outer loop (teacher) ----
    n_outer_steps: int = 50             # paper uses 200
    outer_group_size: int = 4           # g: datasets per outer step (paper uses up to 8)
    inner_dataset_size: int = 64        # n: problems per dataset X_k (paper uses 64)
    r_parallel_students: int = 1        # r: parallel student runs per dataset (paper uses 4)
    teacher_lr: float = 1e-5            # paper uses 1e-5
    teacher_max_new_tokens: int = 1024  # paper uses 512
    teacher_temperature: float = 1.0
    teacher_rejection_cap: int = 16      # safety cap on rejection sampling multiplier

    # ---- inner loop (student) ----
    inner_steps: int = 10               # K: student RLOO steps per X_k (paper)
    inner_steps_per_promotion: int = 5  # paper B.7: "+5 steps" per promotion stage
    student_group_size: int = 32         # G: completions per prompt (paper uses 32)
    student_batch_problems: int = 8     # B: paper's batch size (prompts per gradient step)
    student_lr: float = 1e-5            # paper uses 1e-5
    student_max_new_tokens: int = 1024  # matches fail@128 token budget
    student_temperature: float = 1.0

    # ---- rl ----
    kl_beta: float = 0.02
    max_prompt_len: int = 1024
    rl_algo: str = "rloo"               # {"rloo", "grpo"}: inner-loop student trainer

    # ---- promotion ----
    promotion_threshold: float = 0.01   # τ (paper)
    promotion_window: int = 3           # m (paper): moving-average window for teacher reward

    # ---- vllm ----
    vllm_gpu_memory_utilization: float = 0.25
    vllm_max_model_len: int = 2048
    vllm_max_loras: int = 2

    # ---- eval / logging ----
    wandb_project: str = "soar-mini"
    wandb_mode: str = "online"
    out_dir: str = "runs/soar-v1"
    log_every: int = 1
    ckpt_every: int = 10

    # ---- PQ baseline (SOAR paper Table 3 + B.6/B.7) ----
    pq_max_train_steps: int = 200        # paper Table 3 is 1500; capped at 200 for the sweep
    pq_batch_problems: int = 8           # paper batch size: problems per gradient step
    pq_group_size: int = 32              # paper group size: rollouts per problem
    pq_lr: float = 1e-5                  # Table 3
    pq_kl_beta: float = 0.001            # Table 3
    pq_lr_warmup_steps: int = 20         # Table 3; 0 for curriculum (deferred)
    pq_max_new_tokens: int = 1024        # Table 3
    pq_temperature: float = 1.0          # Table 3 (sampling during training)
    pq_diag_every: int = 50              # greedy Q_R diagnostic cadence (internal signal)
    pq_slope_cma_window: int = 25        # B.6 reward-curve smoothing window
    pq_slope_threshold: float = 0.15     # B.6 early-stopping normalized-slope threshold
    pq_test_window: int = 200            # B.6 post-selection test-perf averaging window
    pq_test_ckpt_every: int = 20         # how often to snapshot inside the 200-step window
    pq_pass_at_k_samples: int = 32       # §4.3: 32 samples/problem for pass@k
    pq_pass_at_k_ks: tuple = (1, 4, 8, 16, 32)


def smoke_config() -> SoarConfig:
    """Tiny config for end-to-end smoke test."""
    return SoarConfig(
        n_outer_steps=1,
        outer_group_size=2,
        inner_dataset_size=2,
        r_parallel_students=1,
        inner_steps=1,
        student_group_size=2,
        student_batch_problems=1,
        n_reward_questions=4,
        teacher_max_new_tokens=256,
        student_max_new_tokens=256,
        wandb_mode="disabled",
        out_dir="runs/smoke",
        pq_max_train_steps=2,
        pq_batch_problems=1,
        pq_group_size=2,
        pq_lr_warmup_steps=0,
        pq_max_new_tokens=256,
        pq_diag_every=1,
        pq_slope_cma_window=1,
        pq_test_window=1,
        pq_test_ckpt_every=1,
        pq_pass_at_k_samples=2,
        pq_pass_at_k_ks=(1, 2),
    )
