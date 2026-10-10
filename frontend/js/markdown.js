// A small, safe Markdown renderer for answers. Everything is HTML-escaped first; only the
// constructs below are turned into markup. Citations ([1], [1, 3]) become buttons the app wires up.
// Math written as \( … \), \[ … \], $ … $ or $$ … $$ is set as readable text (Greek letters,
// subscripts, superscripts) rather than shown as raw LaTeX; there is no TeX engine here.

const escapeHtml = (value) =>
  String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

const GREEK = {
  alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", varepsilon: "ε", zeta: "ζ", eta: "η", theta: "θ",
  iota: "ι", kappa: "κ", lambda: "λ", mu: "μ", nu: "ν", xi: "ξ", pi: "π", rho: "ρ", sigma: "σ", tau: "τ",
  upsilon: "υ", phi: "φ", varphi: "φ", chi: "χ", psi: "ψ", omega: "ω",
  Gamma: "Γ", Delta: "Δ", Theta: "Θ", Lambda: "Λ", Xi: "Ξ", Pi: "Π", Sigma: "Σ", Phi: "Φ", Psi: "Ψ", Omega: "Ω",
};
const SYMBOLS = {
  cdot: "·", times: "×", div: "÷", pm: "±", leq: "≤", le: "≤", geq: "≥", ge: "≥", neq: "≠", ne: "≠", approx: "≈",
  sim: "∼", infty: "∞", sum: "∑", prod: "∏", int: "∫", partial: "∂", nabla: "∇", in: "∈", notin: "∉",
  subset: "⊂", cup: "∪", cap: "∩", forall: "∀", exists: "∃", to: "→", rightarrow: "→", leftarrow: "←",
  Rightarrow: "⇒", mapsto: "↦", ldots: "…", cdots: "⋯", dots: "…", top: "ᵀ", circ: "∘", odot: "⊙", oplus: "⊕",
  otimes: "⊗", log: "log", exp: "exp", max: "max", min: "min", arg: "arg", softmax: "softmax", tanh: "tanh",
};

// Turn a TeX fragment into escaped, readable HTML.
function texToHtml(tex) {
  let s = String(tex).trim();
  // \frac{a}{b} → (a)/(b); \sqrt{x} → √(x); \text{…}, \mathrm{…}, \mathbf{…} → the text
  for (let i = 0; i < 4; i++) {
    s = s
      .replace(/\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, "($1)/($2)")
      .replace(/\\sqrt\s*\{([^{}]*)\}/g, "√($1)")
      .replace(/\\(?:text|mathrm|mathbf|mathit|operatorname|mathcal|boldsymbol|hat|bar|vec|tilde)\s*\{([^{}]*)\}/g, "$1");
  }
  s = s.replace(/\\(left|right|big|Big|bigl|bigr|,|;|!|quad|qquad)\b|\\[,;!]/g, " ");
  s = s.replace(/\\([A-Za-z]+)/g, (match, name) => GREEK[name] ?? SYMBOLS[name] ?? name);
  // Escape, then subscripts/superscripts (braced or single character)
  let out = escapeHtml(s);
  out = out
    .replace(/_\{([^{}]*)\}/g, "<sub>$1</sub>")
    .replace(/\^\{([^{}]*)\}/g, "<sup>$1</sup>")
    .replace(/_([A-Za-z0-9α-ωΑ-Ω])/g, "<sub>$1</sub>")
    .replace(/\^([A-Za-z0-9α-ωΑ-Ω*'])/g, "<sup>$1</sup>")
    .replace(/[{}]/g, "")
    .replace(/\s{2,}/g, " ");
  return out.trim();
}

const CITATION = /\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]/g;

function inline(raw, { citations }) {
  const slots = [];
  const keep = (html) => `\u0000${slots.push(html) - 1}\u0000`;
  let s = String(raw);
  s = s.replace(/`([^`\n]+)`/g, (_, code) => keep(`<code>${escapeHtml(code)}</code>`));
  s = s.replace(/\\\((.+?)\\\)/g, (_, tex) => keep(`<span class="math">${texToHtml(tex)}</span>`));
  // $…$ as math only when it reads like math: no space just inside, and not a plain amount like $40
  s = s.replace(/\$(?!\s)([^$\n]+?)(?<!\s)\$(?!\d)/g, (match, tex) =>
    /^[\d.,\s]+$/.test(tex) ? match : keep(`<span class="math">${texToHtml(tex)}</span>`)
  );
  s = s.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, (_, label, url) =>
    keep(`<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)}</a>`)
  );
  s = escapeHtml(s);
  s = s
    .replace(/\*\*(?=\S)([\s\S]*?\S)\*\*/g, "<strong>$1</strong>")
    .replace(/__(?=\S)([\s\S]*?\S)__/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*(?=\S)([^*\n]*?\S)\*(?!\*)/g, "$1<em>$2</em>")
    .replace(/(^|[^\w])_(?=\S)([^_\n]*?\S)_(?!\w)/g, "$1<em>$2</em>");
  s = s.replace(/(^|[\s(])(https?:\/\/[^\s<]+[^\s<.,;:)!?'"])/g, '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>');
  if (citations) {
    s = s.replace(CITATION, (_, list) =>
      list
        .split(",")
        .map((n) => n.trim())
        .map((n) => `<button type="button" class="citation" data-source="${n}" aria-label="Source ${n}">${n}</button>`)
        .join("")
    );
  }
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => slots[Number(i)]);
}

const isTableRow = (line) => /^\s*\|.*\|\s*$/.test(line);
const isTableRule = (line) => /^\s*\|[\s|:-]+\|\s*$/.test(line);
const bullet = /^\s*[-*•]\s+(.*)$/;
const numbered = /^\s*(\d{1,3})[.)]\s+(.*)$/;

export function renderMarkdown(text, { citations = true } = {}) {
  const lines = String(text ?? "").replace(/\r\n?/g, "\n").split("\n");
  const out = [];
  let paragraph = [];
  const flush = () => {
    if (paragraph.length) out.push(`<p>${paragraph.map((line) => inline(line, { citations })).join("<br>")}</p>`);
    paragraph = [];
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();
    if (!trimmed) {
      flush();
      continue;
    }
    if (/^```/.test(trimmed)) {
      flush();
      const code = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i].trim())) code.push(lines[i++]);
      out.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }
    const display = /^(?:\\\[|\$\$)([\s\S]*?)(?:\\\]|\$\$)$/.exec(trimmed);
    if (display) {
      flush();
      out.push(`<div class="math-block">${texToHtml(display[1])}</div>`);
      continue;
    }
    const heading = /^(#{1,4})\s+(.*)$/.exec(trimmed);
    if (heading) {
      flush();
      const level = Math.min(heading[1].length + 2, 5);
      out.push(`<h${level}>${inline(heading[2], { citations })}</h${level}>`);
      continue;
    }
    if (isTableRow(line)) {
      flush();
      const rows = [];
      while (i < lines.length && isTableRow(lines[i])) rows.push(lines[i++]);
      i--;
      const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((cell) => inline(cell.trim(), { citations }));
      const body = rows.filter((row) => !isTableRule(row));
      const hasHead = rows.length > 1 && isTableRule(rows[1]);
      const head = hasHead ? `<thead><tr>${cells(body[0]).map((c) => `<th>${c}</th>`).join("")}</tr></thead>` : "";
      const bodyRows = (hasHead ? body.slice(1) : body).map((row) => `<tr>${cells(row).map((c) => `<td>${c}</td>`).join("")}</tr>`).join("");
      out.push(`<div class="md-table"><table>${head}<tbody>${bodyRows}</tbody></table></div>`);
      continue;
    }
    if (bullet.test(line) || numbered.test(line)) {
      flush();
      const ordered = numbered.test(line);
      const pattern = ordered ? numbered : bullet;
      const items = [];
      while (i < lines.length && (pattern.test(lines[i]) || (/^\s{2,}\S/.test(lines[i]) && items.length))) {
        const match = pattern.exec(lines[i]);
        if (match) items.push(ordered ? match[2] : match[1]);
        else items[items.length - 1] += ` ${lines[i].trim()}`;
        i++;
      }
      i--;
      const tag = ordered ? "ol" : "ul";
      out.push(`<${tag}>${items.map((item) => `<li>${inline(item, { citations })}</li>`).join("")}</${tag}>`);
      continue;
    }
    if (/^>\s?/.test(trimmed)) {
      flush();
      const quote = [];
      while (i < lines.length && /^>\s?/.test(lines[i].trim())) quote.push(lines[i++].trim().replace(/^>\s?/, ""));
      i--;
      out.push(`<blockquote>${quote.map((q) => inline(q, { citations })).join("<br>")}</blockquote>`);
      continue;
    }
    paragraph.push(trimmed);
  }
  flush();
  return out.join("");
}

// Plain text for copying: Markdown and TeX markers removed, citations kept as [n].
export function plainText(text) {
  return String(text ?? "")
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/__(.+?)__/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\\\((.+?)\\\)/g, (_, tex) => tex.replace(/\\([A-Za-z]+)/g, (m, name) => GREEK[name] ?? SYMBOLS[name] ?? name))
    .replace(/^#{1,4}\s+/gm, "");
}

export { escapeHtml };
