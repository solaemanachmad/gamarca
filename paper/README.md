# GAMA-RCA manuscript (Elsevier elsarticle, English)

Build: `latexmk -pdf main.tex` (or pdflatex -> bibtex -> pdflatex x2). `elsarticle.cls` and `elsarticle-num.bst` are included,
so the folder compiles on any TeX Live / Overleaf installation (needs lmodern, rotating, float, placeins, siunitx, cleveref).

Files
- `main.tex` front matter, declarations, includes the sections `sec1_intro.tex` ... `sec8_conclusion.tex`, `secA_appendix.tex`
- `refs.bib` references; entries marked `% TO VERIFY` must be checked (venue, pages, DOI) before submission
- `figs/` vector figures (PDF) + PNG previews; regenerate with `python make_figs.py` (reads `build/per_case.csv` and
  `build/report.json`, exported by `scripts/analyze_v5.py` from the run zips; no model is re-run)

Placeholders (orange `[Pending: ...]` and `XX` in the PDF), to fill when the runs finish
1. Table 4 / Table 6: Evidence->LLM and Belief->LLM with gpt-oss-120b; hypothesis H4.
2. Table 5: leave-one-module-out runs without M1, M3, M4, M5, M6 (Holm correction over six modules).
3. Section 6.5 and the abstract: AIOps2025 zero-shot results.
4. CRediT roles, the Elsevier generative-AI declaration, the target journal (`\journal{}`), author list and affiliations.
