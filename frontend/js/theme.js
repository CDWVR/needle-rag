// Sets the colour theme before the first paint, so a dark-mode visitor never sees a light flash.
// The choice (system, light or dark) lives in this browser only; "system" follows the OS setting.
(function () {
  var KEY = "needle-theme";
  var media = window.matchMedia("(prefers-color-scheme: dark)");
  var chosen = "system";
  try {
    var saved = localStorage.getItem(KEY);
    if (saved === "light" || saved === "dark") chosen = saved;
  } catch (err) {
    /* storage blocked: follow the system */
  }
  function apply() {
    var root = document.documentElement;
    root.dataset.theme = chosen === "system" ? (media.matches ? "dark" : "light") : chosen;
    root.dataset.themePref = chosen;
  }
  function set(pref) {
    chosen = pref === "light" || pref === "dark" ? pref : "system";
    try {
      if (chosen === "system") localStorage.removeItem(KEY);
      else localStorage.setItem(KEY, chosen);
    } catch (err) {
      /* storage blocked: the choice lasts for this page only */
    }
    apply();
  }
  apply();
  if (media.addEventListener) media.addEventListener("change", apply);
  window.needleTheme = {
    set: set,
    get: function () {
      return chosen;
    },
  };
})();
