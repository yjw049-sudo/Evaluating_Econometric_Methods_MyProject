# Presentation

This folder contains a LaTeX Beamer presentation for the project.

Compile with XeLaTeX because the slides contain Chinese text:

```powershell
cd presentation
xelatex main.tex
xelatex main.tex
```

If you use `latexmk`, run it from this folder:

```powershell
cd presentation
latexmk main.tex
```

The `.latexmkrc` file in this folder forces `latexmk` to use XeLaTeX. This is
important because `main.tex` uses `fontspec`, which does not work with
`pdflatex`.

The slides reference figures using relative paths, for example:

- `../Final/Cluster_Frequency.png`
- `../Final/Word_Frequency.png`
- `../output/1920-1949-v1/figures/cluster_share_bar.png`

Before presenting, update `\author{Your Name}` in `main.tex`.
