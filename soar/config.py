from dataclasses import dataclass, field


@dataclass
class SoarConfig:
    # ---- model ----
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")

    # ---- data ----
    subject: str = "math"
    n_eval_questions: int = 32          # Q_R: held-out set for teacher reward
    n_train_exemplars: int = 2          # few-shot examples shown to teacher
    seed: int = 0

    # ---- outer loop (teacher) ----
    n_outer_steps: int = 50
    group_size: int = 8                 # G: candidate problems per outer step
    teacher_lr: float = 1e-5
    teacher_max_new_tokens: int = 384
    teacher_temperature: float = 1.0

    # ---- inner loop (student) ----
    inner_steps: int = 5                # K: GRPO steps on each candidate
    student_rollouts: int = 8           # N: completions per prompt for student GRPO
    student_lr: float = 1e-5
    student_max_new_tokens: int = 256
    student_temperature: float = 1.0

    # ---- grpo ----
    kl_beta: float = 0.02
    max_prompt_len: int = 1024          # truncate prompts to this (safety)

    # ---- vllm ----
    vllm_gpu_memory_utilization: float = 0.35
    vllm_max_model_len: int = 2048
    vllm_max_loras: int = 2

    # ---- eval / logging ----
    eval_rollouts_per_q: int = 1        # 1 => greedy for baseline eval speed
    wandb_project: str = "soar-mini"
    wandb_mode: str = "online"          # "offline" if no net
    out_dir: str = "runs/soar-v1"
    log_every: int = 1
    ckpt_every: int = 10


def smoke_config() -> SoarConfig:
    """Tiny config for end-to-end smoke test."""
    return SoarConfig(
        n_outer_steps=1,
        group_size=2,
        inner_steps=1,
        student_rollouts=2,
        n_eval_questions=4,
        teacher_max_new_tokens=256,
        student_max_new_tokens=128,
        wandb_mode="disabled",
        out_dir="runs/smoke",
    )
