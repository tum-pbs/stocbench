# Page format

The whole page is one markdown file, `content.md`, rendered in source order by `render.js`. Styling lives in `index.html`, chart data in `sb-charts.js`. No build step.

## Structure

- Front matter (the `---` block) fills the hero: `eyebrow`, `title`, `authors`, `affiliation`, `links`.
- A `>` blockquote renders as a lede paragraph. The abstract is a blockquote placed before the first `##`, which keeps it in the hero.
- `##` starts a full-width band, `###` a heading inside it. Backgrounds alternate automatically; the last band is dark.
- `**bold**`, `[links](url)` and `$math$` work everywhere.

## Lists

- `1.` numbered, `a.` lettered, `-` bullets.
- A `-` bullet directly under a numbered item nests inside it.
- An item starting with `**a bold lead**` renders the lead as a headline. A bullet list where every item starts bold becomes a card row.

## Figures

Image syntax; the alt text is the caption.

- `![Caption](assets/fig.png)`: image
- `![Caption](assets/clip.mp4 poster=frame.png)`: looping video
- `![Caption](chart:mean)`: interactive chart. Panels: `mean`, `std`, `energy`, `enstrophy`, `rollout`, `ratio`, `det_mean`, `det_std`, `det_rollout`, `det_ratio`, `det_rmse`. `chart:baselines` and `chart:legend` embed the clickable method legend (large and compact).
- `![Shared caption](left.png, right.png)`: side-by-side panels in one figure
- Several `![...](...)` on one line: side-by-side figures, each with its own caption
- Chart proportions: `data-chart-h` on `<html>` in `index.html` is the chart height in viewBox units (the width is 480); the page uses 250.

## Columns

A `|` at the start of a line opens a new column; a blank line ends the row. Content before the first `|` is the first column. `|[width]` sets the width, taking any CSS grid value (prefer relative units like `1.4fr`). Cells hold any content; columns stack on narrow screens.

## Buttons

A paragraph containing only links becomes a button row; a `**[bold](url)**` link becomes a solid button. Icons are inferred from the label (arXiv, GitHub, Pip, Dataset).

## Preview

Content is fetched at runtime, so serve over HTTP instead of opening the file directly: `py -m http.server`, then visit http://localhost:8000.

## Publish

`.\publish.ps1 ["commit message"]` snapshots the current state as a single commit that replaces the entire history and force-pushes `main`; the push triggers the GitHub Pages deploy (live after about a minute).
