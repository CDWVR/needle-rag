// Small, dependency-free chart builders for the Analytics page. Each returns HTML; wireTips()
// adds the hover cards. Colors: status roles for outcomes, the reference blue/orange pair for the
// two latency lines, brand ink for single-series bars. Values and labels stay in text colors.
import { escapeHtml } from "./markdown.js";

export const OUTCOMES = [
  { key: "answered", label: "Answered with sources", color: "#0ca30c" },
  { key: "no_coverage", label: "Not in the documents", color: "#fab219" },
  { key: "check_failed", label: "Withheld by the check", color: "#ec835a" },
  { key: "error", label: "Failed", color: "#d03b3b" },
];

const pct = (part, total) => (total ? Math.round((part / total) * 1000) / 10 : 0);
const tip = (lines) => `data-viz-tip="${escapeHtml(lines.join("|"))}"`;
const niceMax = (value) => {
  if (value <= 4) return Math.max(1, value);
  const power = 10 ** Math.floor(Math.log10(value));
  return Math.ceil(value / (power / 2)) * (power / 2);
};
// Whole-number gridlines: the first of 2–5 equal steps that divides the top evenly.
const ticksFor = (max) => {
  if (max <= 4) return Array.from({ length: max + 1 }, (_, i) => i);
  const steps = [2, 3, 4, 5].find((n) => max % n === 0) || 1;
  return Array.from({ length: steps + 1 }, (_, i) => (max / steps) * i);
};

// Part-to-whole: one stacked bar with a 2px gap between parts, plus a labelled legend.
export function outcomeBar(outcomes) {
  const total = OUTCOMES.reduce((sum, o) => sum + (outcomes?.[o.key] || 0), 0);
  if (!total) return `<p class="empty-note">No questions in this range yet.</p>`;
  const parts = OUTCOMES.filter((o) => outcomes[o.key]);
  return `
    <div class="viz-stack" role="img" aria-label="${parts.map((o) => `${o.label}: ${outcomes[o.key]}`).join(", ")}">
      ${parts.map((o, i) => `<i class="viz-seg" style="--w:${(outcomes[o.key] / total) * 100}%;--c:${o.color};--i:${i}" ${tip([o.label, `${outcomes[o.key]} of ${total} (${pct(outcomes[o.key], total)}%)`])}></i>`).join("")}
    </div>
    <ul class="viz-legend">${OUTCOMES.map(
      (o) => `<li class="${outcomes[o.key] ? "" : "zero"}"><i style="background:${o.color}"></i><span>${o.label}</span><strong>${outcomes[o.key] || 0}</strong><small>${pct(outcomes[o.key] || 0, total)}%</small></li>`
    ).join("")}</ul>`;
}

// Vertical columns on a shared axis; one series.
export function columnChart({ values, labels, tips, ariaLabel, every = 1, unit = "" }) {
  const top = Math.max(...values, 0);
  if (!top) return `<p class="empty-note">Nothing to show for this range yet.</p>`;
  const max = niceMax(top);
  return `
    <div class="viz-wrap">
      <div class="viz-plot" role="img" aria-label="${escapeHtml(ariaLabel)}">
        ${ticksFor(max).map((t) => `<i class="grid" style="bottom:${(t / max) * 100}%"><span>${Math.round(t)}${unit}</span></i>`).join("")}
        <div class="viz-cols">${values
          .map((v, i) => `<div class="viz-col" style="--i:${i}" ${tip(tips[i])}><i style="height:${(v / max) * 100}%"></i></div>`)
          .join("")}</div>
        <div class="viz-tip" hidden></div>
      </div>
      <div class="viz-x">${labels
        .map((label, i) => (i % every === 0 ? `<span style="left:${((i + 0.5) / labels.length) * 100}%">${escapeHtml(label)}</span>` : ""))
        .join("")}</div>
    </div>`;
}

// Two lines on one axis (same unit), with point markers and a per-day hover column.
export function lineChart({ points, series, ariaLabel, format }) {
  if (!points.length) return `<p class="empty-note">No answers in this range yet.</p>`;
  const top = Math.max(...points.flatMap((p) => series.map((s) => p[s.key] || 0)), 1);
  const max = niceMax(Math.ceil(top / 1000)) * 1000;
  const x = (i) => (points.length === 1 ? 50 : (i / (points.length - 1)) * 100);
  const y = (v) => 100 - (v / max) * 100;
  const path = (key) =>
    points
      .map((p, i) => [x(i), p[key]])
      .filter(([, v]) => v != null)
      .map(([px, v], i) => `${i ? "L" : "M"}${px.toFixed(2)},${y(v).toFixed(2)}`)
      .join(" ");
  const step = Math.max(1, Math.ceil(points.length / 6));
  return `
    <div class="viz-wrap">
      <ul class="viz-legend inline">${series.map((s) => `<li><i style="background:${s.color}"></i><span>${s.label}</span></li>`).join("")}</ul>
      <div class="viz-plot line" role="img" aria-label="${escapeHtml(ariaLabel)}">
        ${ticksFor(max / 1000).map((t) => `<i class="grid" style="bottom:${((t * 1000) / max) * 100}%"><span>${Math.round(t * 10) / 10}s</span></i>`).join("")}
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">${series
          .map((s) => `<path d="${path(s.key)}" fill="none" stroke="${s.color}" stroke-width="2" vector-effect="non-scaling-stroke" stroke-linejoin="round" stroke-linecap="round" class="viz-line"/>`)
          .join("")}</svg>
        ${points
          .map((p, i) =>
            series
              .filter((s) => p[s.key] != null)
              .map((s) => `<i class="viz-dot" style="left:${x(i)}%;top:${y(p[s.key])}%;--c:${s.color}"></i>`)
              .join("")
          )
          .join("")}
        <div class="viz-hover">${points
          .map((p, i) => `<div style="left:${x(i)}%;width:${100 / points.length}%" ${tip([p.label, ...series.map((s) => `${s.label}: ${format(p[s.key])}`), `${p.n} answer${p.n === 1 ? "" : "s"}`])}></div>`)
          .join("")}</div>
        <div class="viz-tip" hidden></div>
      </div>
      <div class="viz-x">${points
        .map((p, i) => ((points.length - 1 - i) % step === 0 ? `<span style="left:${x(i)}%">${escapeHtml(p.label)}</span>` : ""))
        .join("")}</div>
    </div>`;
}

// Ranked horizontal bars with the value at the end.
export function rankBars(items, { unit }) {
  if (!items.length) return `<p class="empty-note">No answers cited a document in this range.</p>`;
  const top = Math.max(...items.map((item) => item.value), 1);
  return `<ul class="viz-rank">${items
    .map(
      (item, i) => `<li ${tip([item.label, `${item.value} ${unit}${item.value === 1 ? "" : "s"}`])} style="--i:${i}">
        <span class="name">${escapeHtml(item.label)}</span>
        <span class="track"><i style="width:${(item.value / top) * 100}%"></i></span>
        <strong>${item.value}</strong>
      </li>`
    )
    .join("")}</ul>`;
}

// One hover card per chart, kept inside the plot.
export function wireTips(root) {
  root.querySelectorAll(".viz-plot, .viz-stack, .viz-rank").forEach((plot) => {
    let card = plot.querySelector(":scope > .viz-tip");
    if (!card) {
      card = document.createElement("div");
      card.className = "viz-tip";
      card.hidden = true;
      plot.appendChild(card);
    }
    plot.querySelectorAll("[data-viz-tip]").forEach((target) => {
      const show = () => {
        const [title, ...lines] = target.dataset.vizTip.split("|");
        card.innerHTML = `<strong>${escapeHtml(title)}</strong>${lines.map((line) => `<span>${escapeHtml(line)}</span>`).join("")}`;
        card.hidden = false;
        const box = target.getBoundingClientRect();
        const area = plot.getBoundingClientRect();
        const half = card.offsetWidth / 2;
        const center = box.left - area.left + box.width / 2;
        card.style.left = `${Math.min(Math.max(center, half + 4), area.width - half - 4)}px`;
        card.style.top = plot.classList.contains("viz-rank") ? `${box.top - area.top - card.offsetHeight - 8}px` : "";
        plot.querySelectorAll(".on").forEach((node) => node.classList.remove("on"));
        target.classList.add("on");
      };
      const hide = () => {
        card.hidden = true;
        target.classList.remove("on");
      };
      target.addEventListener("mouseenter", show);
      target.addEventListener("mouseleave", hide);
      target.addEventListener("focus", show);
      target.addEventListener("blur", hide);
      if (!target.hasAttribute("tabindex")) target.setAttribute("tabindex", "0");
    });
  });
}
