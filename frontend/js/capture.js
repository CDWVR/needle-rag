// A screenshot of the page exactly as the viewer sees it, drawn by the browser itself: the DOM is
// cloned and frozen (no animation, current scroll positions and form values), wrapped in an SVG
// <foreignObject> with the page's stylesheet and fonts inlined, and painted onto a canvas. No
// third-party code, and nothing leaves the page until the person chooses to send it.

let fontFaces = null; // @font-face rules with the font files inlined, built once per page load

const blobToDataUrl = (blob) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });

function stylesheetUrl() {
  return document.querySelector('link[rel="stylesheet"][href*="style.css"]')?.href || "/static/style.css";
}

// An image cannot fetch anything, so web fonts must travel inside it. Only the Latin subsets are
// used on this page; the others would only add weight.
async function inlineFonts(css) {
  if (fontFaces !== null) return fontFaces;
  let faces = "";
  try {
    const imports = [...css.matchAll(/@import\s+url\(['"]?([^'")]+)['"]?\)/g)]
      .map((match) => match[1])
      .filter((url) => url.startsWith("https://fonts.googleapis.com/"));
    for (const url of imports) {
      const sheet = await (await fetch(url)).text();
      for (const face of sheet.match(/@font-face\s*{[^}]*}/g) || []) {
        if (!/unicode-range:[^;]*U\+0000-00FF/i.test(face)) continue;
        const source = face.match(/url\((https:\/\/fonts\.gstatic\.com\/[^)]+)\)/);
        if (!source) continue;
        const data = await blobToDataUrl(await (await fetch(source[1])).blob());
        faces += face.replace(source[1], data) + "\n";
      }
    }
    fontFaces = faces;
  } catch {
    return faces; // offline or blocked: the screenshot falls back to system fonts; try again next time
  }
  return fontFaces;
}

/** Warm the font cache ahead of time (e.g. when the pointer nears the bug button). */
export async function prepareCapture() {
  if (fontFaces !== null) return;
  try {
    const css = await (await fetch(stylesheetUrl())).text();
    await inlineFonts(css);
  } catch {
    /* capture will retry */
  }
}

// Carry over what cloneNode leaves behind: scroll offsets, typed values, ticked boxes, canvases.
function freeze(original, copy) {
  if (!(original instanceof Element)) return;
  if ((original.scrollTop || original.scrollLeft) && original !== document.documentElement) {
    copy.style.overflow = "hidden";
    for (const child of copy.children) {
      // `translate` composes with any transform the child already has and leaves layout alone.
      child.style.translate = `${-original.scrollLeft}px ${-original.scrollTop}px`;
    }
  }
  if (original instanceof HTMLInputElement) {
    if (original.type === "password") copy.setAttribute("value", "");
    else if (original.type === "checkbox" || original.type === "radio") copy.toggleAttribute("checked", original.checked);
    else copy.setAttribute("value", original.value);
  } else if (original instanceof HTMLTextAreaElement) {
    copy.textContent = original.value;
  } else if (original instanceof HTMLSelectElement) {
    [...copy.options].forEach((option, index) => option.toggleAttribute("selected", index === original.selectedIndex));
  } else if (original instanceof HTMLCanvasElement) {
    try {
      const image = document.createElement("img");
      image.src = original.toDataURL();
      image.style.cssText = original.style.cssText;
      image.width = original.width;
      image.height = original.height;
      copy.replaceWith(image);
    } catch {
      /* a tainted canvas stays blank */
    }
  }
}

const loadImage = (src) =>
  new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("The page could not be drawn."));
    image.src = src;
  });

/**
 * Capture the visible viewport. Elements matching `skip` (the reporter's own UI) are left out.
 * Resolves to { blob, url, width, height }; the caller revokes `url` when done.
 */
export async function capturePage({ skip = "[data-capture-skip]", maxWidth = 2400, quality = 0.86 } = {}) {
  const width = document.documentElement.clientWidth;
  const height = window.innerHeight;
  const css = await (await fetch(stylesheetUrl())).text();
  const fonts = await inlineFonts(css);

  const source = document.documentElement;
  const clone = source.cloneNode(true);
  const originals = [source, ...source.querySelectorAll("*")];
  const copies = [clone, ...clone.querySelectorAll("*")];
  originals.forEach((node, index) => freeze(node, copies[index]));
  clone.querySelectorAll(`script, link, noscript, template, ${skip}`).forEach((node) => node.remove());

  // Inside an SVG image `:root` is the <svg>, not <html>, so the page's tokens move to `html`.
  // (The Google Fonts URL itself contains semicolons, so the @import is matched through its `url(...)`.)
  const pageCss = css.replace(/@import\s+url\([^)]*\)[^;]*;/g, "").replace(/:root/g, "html");
  const style = document.createElement("style");
  style.textContent = `${fonts}\n${pageCss}\n
    html, body { width: ${width}px !important; height: ${height}px !important; overflow: hidden !important; }
    *, *::before, *::after { animation: none !important; transition: none !important; caret-color: transparent !important; }`;
  (clone.querySelector("head") || clone).appendChild(style);

  const markup = new XMLSerializer().serializeToString(clone);
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"><foreignObject x="0" y="0" width="100%" height="100%">${markup}</foreignObject></svg>`;
  const image = await loadImage(`data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`);

  const scale = Math.min(window.devicePixelRatio || 1, 2, maxWidth / width);
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const context = canvas.getContext("2d");
  context.fillStyle = getComputedStyle(document.body).backgroundColor || "#f2f0e8";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  // toBlob can hand back null (e.g. a background tab); the data-URL path is slower but dependable.
  let blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
  if (!blob) {
    const data = canvas.toDataURL("image/jpeg", quality);
    if (!data.startsWith("data:image/jpeg")) throw new Error("The screenshot could not be saved.");
    const bytes = atob(data.slice(data.indexOf(",") + 1)); // decoded here: the CSP does not allow fetching data: URLs
    blob = new Blob([Uint8Array.from(bytes, (char) => char.charCodeAt(0))], { type: "image/jpeg" });
  }
  return { blob, url: URL.createObjectURL(blob), width, height };
}
