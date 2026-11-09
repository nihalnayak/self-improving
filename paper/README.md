# paper/

ICML 2026 paper, built from the official style kit
(https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip).

| file | what |
|---|---|
| `main.tex` | paper entry point (blind submission: `\usepackage{icml2026}`) |
| `macros.tex` | notation from SGS (arXiv:2604.20209): `\solver`, `\conj`, `\guide`, `\B`, `\Bsynth`, ... |
| `sections/algorithms.tex` | SGS, SGS-GRPO (off-policy), SGS-GRPO (on-policy), GRPO loss |
| `references.bib` | bibliography |
| `icml2026.sty`, `icml2026.bst`, `algorithm.sty`, `algorithmic.sty`, `fancyhdr.sty` | ICML kit |
| `example_paper.*` | ICML's example paper, kept for reference |

Build locally: `latexmk -pdf main.tex`. On Overleaf, set **Menu → Main document** to `main.tex`.

## Syncing with Overleaf

`paper/` is mirrored to an Overleaf project through Overleaf's git bridge
(needs a plan with git integration). The cluster's git has no `git subtree`,
so `sync_overleaf.sh` does the same thing with built-in git commands.

### One-time setup
1. On Overleaf: **New Project → Blank Project**. Copy the git URL from
   **Menu → Sync → Git** (`https://git.overleaf.com/<project-id>`).
2. Create a git token: Overleaf **Account Settings → Git integration**.
3. From the repo root:
   ```bash
   git remote add overleaf https://git.overleaf.com/<project-id>
   paper/sync_overleaf.sh init   # merge the blank project in, keeping our files
   paper/sync_overleaf.sh push   # upload paper/ to Overleaf
   ```
   When git asks for credentials, the username is `git` and the password is the token.
   After `init`, check `git status`. The blank project's own files (e.g. its
   `main.tex`) conflict with ours and are resolved in favour of ours; any other
   default file it adds (e.g. `sample.bib`) lands in `paper/`, and you can delete it.

### Day to day
```bash
paper/sync_overleaf.sh pull   # bring Overleaf edits into paper/ (merge commit)
# ... edit paper/ locally, git commit ...
paper/sync_overleaf.sh push   # send committed paper/ to Overleaf
```
- `push` only sends **committed** changes, and refuses to run if Overleaf has
  edits you have not pulled yet. Pull first, then push.
- Overleaf does not accept force pushes; never rewrite the pushed history.
- LaTeX build products are ignored via `paper/.gitignore`.
