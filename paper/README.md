# ICML submission — Credit Where It's Due — If It's True

Built on the **official ICML 2026 style kit** (`media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip`). The kit's `.sty`/`.bst` files are unmodified and kept at the project root, as shipped.

**When the ICML 2027 kit is released:**
1. Replace `icml2026.sty` and `icml2026.bst` with the 2027 files.
2. Update the three `icml2026` names in `main.tex` (`\usepackage{icml2026}` and `\bibliographystyle{icml2026}`).
3. Re-check the call for papers for any rule changes.

## ICML rules (from the 2026 kit — verify against the 2027 call for papers)

| Rule | Requirement |
| --- | --- |
| Page limit | Main body **8 pages**, excluding references and appendices. Camera-ready may add 1 page |
| Single file | Main body, references and appendix in **one PDF**; total size under 10 MB |
| Blind review | `\usepackage{icml2026}` (no option) hides authors automatically. **No acknowledgements** in the submission |
| Font | 10 pt Times, **Type-1 fonts only** |
| Captions | Figure captions **below** the figure; table captions **above** the table |
| Abstract | One paragraph, ideally 4–6 sentences |
| Impact Statement | **Required**: unnumbered, before the references, not counted toward the page limit |

**Modes** (set by the `\usepackage{icml2026}` line in `main.tex`):
- **Submission:** `\usepackage{icml2026}` (current).
- **Preprint:** `\usepackage[preprint]{icml2026}`.
- **Camera-ready:** `\usepackage[accepted]{icml2026}`. Also fill in authors, and uncomment the acknowledgements.

**Before submitting:**
- Hide the TODO notes: in `main.tex`, swap the two `todonotes` lines.

## Build

- **Overleaf:** upload the whole `paper/` folder; the compiler is pdfLaTeX (the default).
- **Locally:** `latexmk -pdf main.tex`.
- **TeXpresso (live preview):** TeXpresso runs neither BibTeX nor a second pass; it reads `main.aux` and `main.bbl` from a normal build.
  - Run `latexmk -pdf main.tex` once, and again after any change to `references.bib` or new `\cite`/`\label`.
  - **Don't run `latexmk -c`**: it deletes `main.aux`, and TeXpresso then shows unknown citations.
  - **It does not watch files on disk.** The VS Code extension sends only typed edits; after files change on disk (e.g. edited by a script or another tool), run **TeXpresso: Refresh**, or **External Compile and Refresh** after new `\label`/`\cite`. Restart it (Stop, then Start Document) if a new `\input` file is not picked up.

Checked locally: 0 errors, 0 overfull boxes, all fonts Type 1, US Letter.

## Layout

| Path | Content |
| --- | --- |
| `main.tex` | Preamble (kit verbatim), title, abstract, section inputs, Impact Statement, appendix |
| `sections/01-introduction.tex` … `07-conclusion.tex` | Body, following `../PAPER_PLAN.md` |
| `sections/appendix.tex` | Proofs, notation, experimental details |
| `tables/` | Tables (`\input`) |
| `figures/` | Figures copied from `../figures/`. Re-copy after reruns |
| `references.bib` | Bibliography. **Verify every entry before submission** |
| `icml2026.sty`, `icml2026.bst`, `fancyhdr.sty`, `algorithm.sty`, `algorithmic.sty` | Official kit files. Do not edit |

## Open items

- Numbers: fake-jams results are the 10-seed values (`../logs/fake_jams_*.md`); bandit results are 32 runs, 600 episodes (`../logs/bandit_run2.log`).
- Prose: the TODO notes mark each paragraph to write (see `../PAPER_PLAN.md`).
- Baselines: accuracy, reputation (sceptical receivers), shared return and the TVC ablations are run in fake jams; the bandit has all credit designs but not the TVC bonus-off and fine-off ablations.
