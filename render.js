/* Renders content.md: standard markdown, in source order. Front matter plus any
   content before the first "##" (e.g. the ">" abstract lede) = hero, each
   "##" is a band (backgrounds alternate via CSS), "###" a heading inside it, figures
   via ![caption](src). */
(async function () {
  const OL = /^(\d+|[a-z])\.\s+(.*)/, UL = /^-\s+(.*)/;
  const md = await (await fetch('content.md')).text();
  const { meta, body } = frontMatter(md);
  const { pre, sections } = split(body);

  document.getElementById('page').innerHTML =
    hero(meta, pre) +
    sections.map(band).join('');
  document.title = meta.title || document.title;
  document.querySelectorAll('video').forEach(v => { v.muted = true; v.play().catch(() => {}); });

  function frontMatter(text) {
    const m = text.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
    const meta = {};
    if (m) for (const line of m[1].split(/\r?\n/)) {
      const i = line.indexOf(':');
      if (i > 0) meta[line.slice(0, i).trim()] = line.slice(i + 1).trim();
    }
    return { meta, body: m ? text.slice(m[0].length) : text };
  }

  function split(text) {
    const pre = [], sections = [];
    let cur = null;
    for (const line of text.split(/\r?\n/)) {
      const m = line.match(/^##\s+(.*)/);
      if (m) sections.push(cur = { title: m[1].trim(), lines: [] });
      else (cur ? cur.lines : pre).push(line);
    }
    return { pre, sections };
  }

  function band(s) {
    return `<section class="band"><div class="wrap"><h2>${fmt(s.title)}</h2>${blocks(s.lines)}</div></section>`;
  }

  function hero(meta, lines) {
    return `<header class="hero wrap">
      <div class="eyebrow">${fmt(meta.eyebrow)}</div>
      <h1>${fmt(meta.title)}</h1>
      <p class="authors">${fmt(meta.authors)}</p>
      <p class="affiliation">${fmt(meta.affiliation)}</p>
      ${buttons(meta.links || '')}
      ${blocks(lines)}
    </header>`;
  }

  /* ---------- blocks, in source order ---------- */

  function blocks(lines) {
    const out = [];
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      let m;
      if (!line) continue;
      let end = i;
      while (end < lines.length && lines[end].trim()) end++;
      if (lines.slice(i, end).some(l => l.trim().startsWith('|'))) { out.push(row(lines.slice(i, end))); i = end; continue; }
      if ((m = line.match(/^###\s+(.*)/))) out.push(`<h3>${fmt(m[1])}</h3>`);
      else if (line.startsWith('>')) {
        let text = line.replace(/^>\s?/, '');
        while (i + 1 < lines.length && lines[i + 1].trim().startsWith('>'))
          text += ' ' + lines[++i].trim().replace(/^>\s?/, '');
        out.push(`<blockquote>${fmt(text)}</blockquote>`);
      }
      else if (line.startsWith('![')) out.push(figures(line));
      else if (OL.test(line) || UL.test(line)) { const [html, di] = list(lines, i); out.push(html); i = di; }
      else {
        let text = line;
        while (i + 1 < lines.length && lines[i + 1].trim() && !special(lines[i + 1].trim()))
          text += ' ' + lines[++i].trim();
        out.push(linksOnly(text) ? buttons(text) : `<p>${fmt(text)}</p>`);
      }
    }
    return out.join('\n');
  }

  function special(line) { return /^(###\s|!\[|-\s|\||>|(\d+|[a-z])\.\s)/.test(line); }

  /* ---------- columns: within a contiguous run, "|" at line start opens a new
     column ("|[width]" to size it); a blank line ends the row ---------- */

  function row(run) {
    const cells = [{ w: '1fr', lines: [] }];
    for (const l of run) {
      const m = l.trim().match(/^\|(?:\[([^\]]*)\])?\s*(.*)$/);
      if (m) cells.push({ w: m[1] || '1fr', lines: m[2] ? [m[2]] : [] });
      else cells[cells.length - 1].lines.push(l);
    }
    if (!cells[0].lines.length) cells.shift();
    return `<div class="row" style="grid-template-columns:${esc(cells.map(c => c.w).join(' '))}">`
      + cells.map(c => `<div>${blocks(c.lines)}</div>`).join('') + '</div>';
  }

  /* ---------- lists: 1. numbered, a. lettered, - bullets;
     a "-" bullet inside a numbered list nests under the current item ---------- */

  function list(lines, i) {
    const kind = UL.test(lines[i].trim()) ? 'ul' : /^\d/.test(lines[i].trim()) ? 'num' : 'alpha';
    const marker = kind === 'ul' ? UL : OL;
    const items = [];
    let start = 1;
    for (; i < lines.length; i++) {
      const raw = lines[i], t = raw.trim();
      let m;
      if ((m = t.match(marker))) {
        items.push({ text: m[2] ?? m[1], subs: [] });
        if (items.length === 1 && kind !== 'ul') start = kind === 'num' ? +m[1] : m[1].charCodeAt(0) - 96;
      }
      else if (kind !== 'ul' && items.length && (m = t.match(UL))) items[items.length - 1].subs.push(m[1]);
      else if (t && /^\s/.test(raw) && items.length) {
        const it = items[items.length - 1];
        if (it.subs.length) it.subs[it.subs.length - 1] += ' ' + t; else it.text += ' ' + t;
      }
      else if (!t && (lines[i + 1] || '').trim().match(marker)) continue;
      else break;
    }
    const li = items.map(it =>
      `<li>${lead(it.text)}${it.subs.length ? `<ul>${it.subs.map(t => `<li>${fmt(t)}</li>`).join('')}</ul>` : ''}</li>`).join('');
    const html = kind === 'ul'
      ? `<ul${items.every(it => it.text.startsWith('**')) ? ' class="cards"' : ''}>${li}</ul>`
      : `<ol${kind === 'alpha' ? ' class="alpha"' : ''}${start > 1 ? ` start="${start}" style="counter-reset:item ${start - 1}"` : ''}>${li}</ol>`;
    return [html, i - 1];
  }

  // A bold lead ("**Title** text") is marked so CSS can style it as a headline/term.
  function lead(text) {
    return text.startsWith('**') ? fmt(text).replace('<strong>', '<strong class="lead">') : fmt(text);
  }

  /* ---------- figures: ![caption](src) — img, video, chart:<panel>; "src1, src2" = panels ---------- */

  function figures(line) {
    const figs = [...line.matchAll(/!\[([^\]]*)\]\(([^)]+)\)/g)]
      .map(([, alt, src]) => figure(alt.trim(), src.trim()));
    return figs.length > 1
      ? `<div class="figrow" style="grid-template-columns:repeat(${figs.length},1fr)">${figs.join('')}</div>`
      : figs[0];
  }

  function figure(alt, src) {
    const panels = src.split(',').map(s => media(s.trim()));
    const kind = /chart:(baselines|legend)/.test(src) ? 'legend' : src.includes('chart:') ? 'chart' : 'media';
    const body = panels.length > 1
      ? `<div class="panels" style="grid-template-columns:repeat(${panels.length},1fr)">${panels.join('')}</div>`
      : panels[0];
    return `<figure class="${kind}">${body}${alt ? `<figcaption>${fmt(alt)}</figcaption>` : ''}</figure>`;
  }

  function media(spec) {
    const [path, ...opts] = spec.split(/\s+/);
    if (path === 'chart:baselines') return '<sb-baselines></sb-baselines>';
    if (path === 'chart:legend') return '<sb-legend></sb-legend>';
    if (path.startsWith('chart:')) return `<sb-chart panel="${esc(path.slice(6))}"></sb-chart>`;
    if (/\.(mp4|webm|mov)$/i.test(path)) {
      const poster = (opts.find(o => o.startsWith('poster=')) || '').slice(7);
      return `<video src="${esc(path)}"${poster ? ` poster="${esc(poster)}"` : ''} autoplay loop muted playsinline></video>`;
    }
    return `<img src="${esc(path)}" alt="">`;
  }

  /* ---------- a paragraph of links renders as buttons; bold link = solid button ---------- */

  function linksOnly(text) {
    return text.includes('](') && text.replace(/\*\*|\[[^\]]*\]\([^)]*\)|\s/g, '') === '';
  }

  function buttons(spec) {
    const btns = [...spec.matchAll(/(\*\*)?\[([^\]]+)\]\(([^)]+)\)(\*\*)?/g)].map(([, b, label, href]) =>
      `<a class="btn${b ? ' dark' : ''}" href="${esc(href)}">${icon(label, !!b)}${esc(label)}</a>`);
    return btns.length ? `<div class="btns">${btns.join('')}</div>` : '';
  }

  function icon(label, dark) {
    const l = label.toLowerCase();
    const slug = l.includes('arxiv') ? 'arxiv'
      : l.includes('github') || l.includes('code') ? 'github'
      : l.includes('pip') || l.includes('pypi') ? 'pypi'
      : l.includes('dataset') || l.includes('hugging') ? 'hf' : '';
    if (!slug) return '';
    const url = slug === 'hf'
      ? 'https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg'
      : `https://cdn.simpleicons.org/${slug}${slug === 'github' && dark ? '/e8e8e4' : ''}`;
    return `<img src="${url}" alt="">`;
  }

  /* ---------- inline formatting ---------- */

  function esc(s) { return (s || '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

  function fmt(s) {
    return esc(s)
      .replace(/\$([^$]+)\$/g, '<ka-tex tex="$1"></ka-tex>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
  }
})();
