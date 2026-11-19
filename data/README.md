# data

Task datasets used by the trainers and evals in this repo. There are two task families: **sorting** and **multiplication**.

```
data/
  sorting/
    sorting_1000.jsonl               train, all lengths     (config "all")
    sorting_1000_L8-128.jsonl        train, lengths 8, 128  (config "L8-128")
    sorting_1000_L8-16-128.jsonl     train, lengths 8, 16, 128
    sorting_1000_L8-16-32-128.jsonl  train, lengths 8, 16, 32, 128
    sorting_val_100.jsonl            validation, all lengths
  multiplication/
    multiplication_1000.jsonl              train, all digit counts
    multiplication_1000_L2-10.jsonl        train, digits 2, 10
    multiplication_1000_L2-4-10.jsonl      train, digits 2, 4, 10
    multiplication_1000_L2-4-6-10.jsonl    train, digits 2, 4, 6, 10
    multiplication_val_100.jsonl           validation, all digit counts
```

## Sorting task

**Task.** Given an unsorted list of integers, output the list in ascending order.

**Record format.** Each file is JSON Lines, with one problem per line:

```json
{"list": [2201, 9325, 1033, 4179, 1931, 8117, 7364, 7737], "sorted": [1033, 1931, 2201, 4179, 7364, 7737, 8117, 9325]}
```

| field | type | meaning |
| --- | --- | --- |
| `list` | `list[int]` | the problem: integers drawn uniformly from [0, 10000]; duplicates are allowed; never already sorted |
| `sorted` | `list[int]` | the reference answer, `sorted(list)` |

Trainers read only `list`: `sgs.common.load_targets` loads it with `load_dataset("json", ...)`. The gold answer is recomputed as `sorted(list)`, so `sorted` is a redundant sanity field.

**Difficulty = list length.** Lengths are 8, 16, 32, 64, 128. Base Qwen3-4B gets val pass@5 of 0.9 / 0.8 at lengths 8 / 16 and 0.0 at 32, 64 and 128, so the long lists are the hard targets that self-play is meant to unlock. Results are reported per length as pass@5 (`len_8` … `len_128`, plus `all`).

**Prompt.** This is the user message, with no system prompt; the model's chat template is applied by verl or vLLM. It comes from `sgs/common.py` and is identical to `misc/sort_benchmark.py`:

```
Sort the numbers in the list in ascending order.

List: [2201, 9325, 1033, 4179, 1931, 8117, 7364, 7737]

Return only the sorted list on a single line, enclosed in \boxed{}, e.g. \boxed{[1, 2, 3]}.
```

**Verifier** (`sgs.common.parse_sorted`). The parser takes the numbers inside the first `\boxed{...}`. If there is none, it falls back to the last `[...]` in the response. The response is correct (reward 1) only if the parsed numbers exactly equal `sorted(list)`: same values, same order, same length. Otherwise the reward is 0, and an unparsable response also gets 0.

### Files

| file | rows | lengths (rows each) | used as |
| --- | --- | --- | --- |
| `sorting_1000.jsonl` | 1000 | 8, 16, 32, 64, 128 (200) | train config `all` (array task 3) |
| `sorting_1000_L8-128.jsonl` | 1000 | 8, 128 (500) | train config `L8-128` (task 0) |
| `sorting_1000_L8-16-128.jsonl` | 1000 | 8, 16 (333), 128 (334) | train config `L8-16-128` (task 1) |
| `sorting_1000_L8-16-32-128.jsonl` | 1000 | 8, 16, 32, 128 (250) | train config `L8-16-32-128` (task 2) |
| `sorting_val_100.jsonl` | 100 | 8, 16, 32, 64, 128 (20) | held-out validation for every run |

- Rows are grouped by length in ascending order. Row order carries no meaning.
- No validation list appears in any train file.
- The train subsets leave out some lengths (e.g. 32 and 64). Validation always covers every length, so it also measures generalization to lengths that were not trained on.
- The SGS Conjecturer proposes its own synthetic lists during training. Those are not stored here; each run writes them to `<run dir>/synthetic.jsonl`.

### Regenerating

`misc/gen_sort_data.py` is deterministic given its seed. These commands reproduce the files byte for byte:

```bash
python misc/gen_sort_data.py --out data/sorting/sorting_1000.jsonl
python misc/gen_sort_data.py --n 1000 --lengths 8,128        --out data/sorting/sorting_1000_L8-128.jsonl
python misc/gen_sort_data.py --n 1000 --lengths 8,16,128     --out data/sorting/sorting_1000_L8-16-128.jsonl
python misc/gen_sort_data.py --n 1000 --lengths 8,16,32,128  --out data/sorting/sorting_1000_L8-16-32-128.jsonl
python misc/gen_sort_data.py --per-length 20 --seed 1        --out data/sorting/sorting_val_100.jsonl
```

- `--lengths` sets the list lengths.
- `--per-length` sets the number of rows per length. Alternatively, `--n` sets a total that is split evenly across lengths, with the remainder going to the longest.
- `--lo` / `--hi` set the integer range (default 0 / 10000).
- `--seed` defaults to 0.

### Adding a new split

1. Generate it with `misc/gen_sort_data.py` into `data/sorting/`, keeping the `{"list", "sorted"}` schema.
2. To train on it, pass `--dataset data/sorting/<file>` to any `python -m sgs.<trainer>`. For slurm, add it to the `CONFIGS`/`FILES` arrays in the `slurm_scripts/train/*.sbatch` scripts.
3. To evaluate on it, pass `DATASETS="<split>=data/sorting/<file>"` to `slurm_scripts/eval/eval_sort.sbatch`.

## Multiplication task

**Task.** Multiply two non-negative integers.

**Record format.** One problem per line: `{"number1": 4821, "number2": 7350, "product": 35434350}`. Both operands have the same digit count with no leading zeros, and `product == number1 * number2`.

**Difficulty = operand digit count.** Digit counts are 2, 4, 6, 8, 10.

### Files

| file | rows | digits (rows each) |
| --- | --- | --- |
| `multiplication_1000.jsonl` | 1000 | 2, 4, 6, 8, 10 (200) |
| `multiplication_1000_L2-10.jsonl` | 1000 | 2, 10 (500) |
| `multiplication_1000_L2-4-10.jsonl` | 1000 | 2, 4 (333), 10 (334) |
| `multiplication_1000_L2-4-6-10.jsonl` | 1000 | 2, 4, 6, 10 (250) |
| `multiplication_val_100.jsonl` | 100 | 2, 4, 6, 8, 10 (20) |

- No val pair appears in any train file, and pairs are unique within a file. Pairs are compared unordered, so `(a, b)` and `(b, a)` count as the same pair. This matters at 2 digits, where only 8100 ordered pairs exist.

### Regenerating

```bash
V=data/multiplication/multiplication_val_100.jsonl
python misc/gen_multiplication_data.py --per-length 20 --seed 1 --lengths 2,4,6,8,10 --out $V
python misc/gen_multiplication_data.py --n 1000 --lengths 2,4,6,8,10 --exclude $V --out data/multiplication/multiplication_1000.jsonl
python misc/gen_multiplication_data.py --n 1000 --lengths 2,10       --exclude $V --out data/multiplication/multiplication_1000_L2-10.jsonl
python misc/gen_multiplication_data.py --n 1000 --lengths 2,4,10     --exclude $V --out data/multiplication/multiplication_1000_L2-4-10.jsonl
python misc/gen_multiplication_data.py --n 1000 --lengths 2,4,6,10   --exclude $V --out data/multiplication/multiplication_1000_L2-4-6-10.jsonl
```

`--exclude` skips pairs that appear in the given JSONL.

### Training

`sgs.solver_simple` and `sgs.sgs_onpolicy_grpo` (and `sgs.sgs_grpo`) take `--task multiplication`. The prompts, verifier and loader are `Multiplication` in `sgs/tasks.py`. The Solver prompt and the answer parser come verbatim from `misc/multiplication_benchmark.py`. Per-digit metrics are logged as `len_{digits}`. For slurm, set `TASK_NAME=multiplication` with `slurm_scripts/train/{grpo,sgs_onpolicy_grpo}.sbatch`. The array tasks 0-3 map to `L2-10`, `L2-4-10`, `L2-4-6-10` and `all`.
