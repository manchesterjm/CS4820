# CS 4820/5820 Citation Standard

**Established:** HW02 (2025-10-26)
**Applies to:** All remaining homeworks and term paper submissions

## Standard Setup (Matches Graded Group Proposal)

This is the EXACT setup used in the graded term paper proposal that was approved by Professor Atyabi.

### LaTeX Preamble

```latex
\documentclass[letterpaper]{article}
\usepackage{aaai24}
\usepackage{times}
\usepackage{helvet}
\usepackage{courier}
\usepackage[hyphens]{url}
\usepackage{graphicx}
\urlstyle{rm}
\def\UrlFont{\rm}
\usepackage{natbib}  % ← Required for bibliography
\usepackage{caption}
```

### End of Document

```latex
\bibliography{references}  % ← Just this, NO \bibliographystyle command!

\end{document}
```

### Bibliography File (references.bib)

Use BibTeX format:

```bibtex
@book{russell2020,
  title={Artificial Intelligence: A Modern Approach},
  author={Russell, Stuart J and Norvig, Peter},
  year={2020},
  edition={4th},
  publisher={Pearson}
}

@misc{atyabi2025csp,
  author={Atyabi, Adham},
  title={Constraint Satisfaction Problems},
  howpublished={Lecture 5, CS 4820/5820: Artificial Intelligence, University of Colorado Colorado Springs},
  year={2025},
  note={Fall 2025}
}
```

## Key Points

1. **DO NOT** add `\bibliographystyle{}` command in your .tex file
2. **DO NOT** modify the `aaai24.sty` file
3. The style file automatically sets `\bibliographystyle{aaai24}` (line 223)
4. This produces **numbered citations** `[1]`, `[2]`, `[3]` in order of appearance

## Citation Style Details

- **In text:** Numbered citations `[1]`, `[2]`, `[3]`
- **References section:** Numbered list in order of first appearance
- **Format:** AAAI 2024 conference style (official AI conference format)

## Overleaf Compilation

**IMPORTANT:** Must compile 2-3 times for bibliography to appear:
1. First compile: Finds citations
2. Second compile: Processes bibliography
3. Third compile: Inserts references into document

## Files to Upload to Overleaf

For each assignment:
- `assignment_writeup.tex` (main document)
- `aaai24.sty` (AAAI style file - use unmodified version)
- `references.bib` (bibliography database)

## Common References

### Textbook
```bibtex
@book{russell2020,
  title={Artificial Intelligence: A Modern Approach},
  author={Russell, Stuart J and Norvig, Peter},
  year={2020},
  edition={4th},
  publisher={Pearson}
}
```

### Professor's Lectures
```bibtex
@misc{atyabi2025csp,
  author={Atyabi, Adham},
  title={Constraint Satisfaction Problems},
  howpublished={Lecture 5, CS 4820/5820: Artificial Intelligence, University of Colorado Colorado Springs},
  year={2025},
  note={Fall 2025}
}

@misc{atyabi2025optimization,
  author={Atyabi, Adham},
  title={Search Optimization Part I-III},
  howpublished={Lecture 7, CS 4820/5820: Artificial Intelligence, University of Colorado Colorado Springs},
  year={2025},
  note={Fall 2025}
}
```

## Remaining Assignments

This standard applies to:
- ✅ HW02 (current)
- ⬜ HW03
- ⬜ HW04
- ⬜ Term Paper Draft
- ⬜ Term Paper Final Submission

## Success Record

- **Term Paper Proposal:** Used this exact setup, graded successfully
- **HW01:** 12.5/12.5 (no bibliography needed)
- **HW02:** Using this standard

---

**Do not deviate from this standard without explicit instructor approval.**
