"""Build train + SGS-synthetic sorting datasets for the GRPO (SGS Synth.) baseline.

For each train config, appends the distinct synthetic lists that the matching
5-iter SGS on-policy GRPO run generated (`<run>/synthetic.jsonl`, all iters)
to the original 1000 train targets. Writes
`files/data/sorting_sgs_onpolicy_5iters_synth/<same name as the train file>`
with the `misc/gen_sort_data.py` schema ({"list", "sorted"}).

Usage:
    python misc/make_sgs_synth_sorting.py
"""

import json
from pathlib import Path

from datasets import load_dataset

RUNS = Path("/n/netscratch/dam_lab/Lab/nnayak1/sgs/runs")
DATA = Path("files/data/sorting")
OUT = Path("files/data/sorting_sgs_onpolicy_5iters_synth")
# train file -> SGS on-policy GRPO job (see files/scripts/sgs/run_eval_sorting_qwen3_4b_5ep.sh)
CONFIGS = {
    "sorting_1000_L8-128.jsonl": 59859684,
    "sorting_1000_L8-16-128.jsonl": 59859689,
    "sorting_1000_L8-16-32-128.jsonl": 60150236,
    "sorting_1000.jsonl": 60150258,
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, job in CONFIGS.items():
        (run,) = RUNS.glob(f"sgs-onpolicy-grpo-*-j{job}")
        train = [list(xs) for xs in load_dataset("json", data_files=str(DATA / name),
                                                 split="train")["list"]]
        synth = load_dataset("json", data_files=str(run / "synthetic.jsonl"),
                             split="train")["synthetic"]
        distinct = list(dict.fromkeys(tuple(int(x) for x in xs) for xs in synth))
        rows = train + [list(xs) for xs in distinct]
        with open(OUT / name, "w") as f:
            f.writelines(json.dumps({"list": xs, "sorted": sorted(xs)}) + "\n" for xs in rows)
        print(f"{name}: {len(train)} train + {len(distinct)} distinct synth "
              f"(of {len(synth)} rows, {run.name}) = {len(rows)} -> {OUT / name}")


if __name__ == "__main__":
    main()
