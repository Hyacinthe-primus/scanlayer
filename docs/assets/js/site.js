// scanlayer docs, shared site behavior
(function () {
  "use strict";

  document.addEventListener("DOMContentLoaded", function () {
    initThemeToggle();
    initMobileNav();
    initTabs();
    initCopyButtons();
    initHighlighting();
    initTocScrollSpy();
    markActiveSidebarLink();
    initDebugCycle();
  });

  var THEME_KEY = "scanlayer-docs-theme";

  function initThemeToggle() {
    var root = document.documentElement;
    var btn = document.querySelector(".theme-toggle");
    if (!btn) return;

    function applyState(theme) {
      btn.setAttribute("aria-pressed", String(theme === "dark"));
      btn.setAttribute(
        "aria-label",
        theme === "dark" ? "Switch to light mode" : "Switch to dark mode"
      );
    }
    applyState(root.getAttribute("data-theme") || "light");

    btn.addEventListener("click", function () {
      var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* storage unavailable */ }
      applyState(next);
    });
  }

  function initMobileNav() {
    var menuBtn = document.querySelector(".menu-btn");
    var sidebar = document.querySelector(".sidebar");
    var backdrop = document.querySelector(".backdrop");
    if (!menuBtn || !sidebar) return;

    function close() {
      sidebar.classList.remove("open");
      if (backdrop) backdrop.classList.remove("show");
      menuBtn.setAttribute("aria-expanded", "false");
    }
    function toggle() {
      var willOpen = !sidebar.classList.contains("open");
      sidebar.classList.toggle("open", willOpen);
      if (backdrop) backdrop.classList.toggle("show", willOpen);
      menuBtn.setAttribute("aria-expanded", String(willOpen));
    }
    menuBtn.addEventListener("click", toggle);
    if (backdrop) backdrop.addEventListener("click", close);
    sidebar.querySelectorAll("a").forEach(function (a) {
      a.addEventListener("click", close);
    });
  }

  function initTabs() {
    document.querySelectorAll(".tabs").forEach(function (group) {
      var buttons = group.querySelectorAll(".tab-buttons button");
      var panels = group.querySelectorAll(".tab-panel");
      buttons.forEach(function (btn) {
        btn.addEventListener("click", function () {
          buttons.forEach(function (b) { b.classList.remove("active"); b.setAttribute("aria-selected", "false"); });
          panels.forEach(function (p) { p.classList.remove("active"); });
          btn.classList.add("active");
          btn.setAttribute("aria-selected", "true");
          var target = group.querySelector('[data-panel="' + btn.dataset.tab + '"]');
          if (target) target.classList.add("active");
        });
      });
    });
  }

  function initCopyButtons() {
    document.querySelectorAll(".code-block").forEach(function (block) {
      var btn = block.querySelector(".copy-btn");
      var codeEl = block.querySelector("pre code");
      if (!btn || !codeEl) return;
      btn.addEventListener("click", function () {
        var text = codeEl.innerText;
        navigator.clipboard && navigator.clipboard.writeText(text).then(function () {
          var original = btn.innerHTML;
          btn.innerHTML = "copied";
          setTimeout(function () { btn.innerHTML = original; }, 1400);
        }).catch(function () {
          btn.innerHTML = "select + ⌘/ctrl-C";
          setTimeout(function () { btn.innerHTML = "copy"; }, 1600);
        });
      });
    });
  }

  function initHighlighting() {
    if (window.hljs) {
      document.querySelectorAll("pre code").forEach(function (el) {
        window.hljs.highlightElement(el);
      });
    }
  }

  function initTocScrollSpy() {
    var tocLinks = document.querySelectorAll(".toc a");
    if (!tocLinks.length) return;
    var targets = [];
    tocLinks.forEach(function (link) {
      var id = link.getAttribute("href").replace("#", "");
      var el = document.getElementById(id);
      if (el) targets.push({ link: link, el: el });
    });
    if (!targets.length) return;

    function onScroll() {
      var scrollPos = window.scrollY + 120;
      var current = targets[0];
      targets.forEach(function (t) {
        if (t.el.offsetTop <= scrollPos) current = t;
      });
      tocLinks.forEach(function (l) { l.classList.remove("active"); });
      current.link.classList.add("active");
    }
    document.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }

  function markActiveSidebarLink() {
    var current = window.location.pathname.split("/").pop() || "index.html";
    document.querySelectorAll(".sidebar a[href]").forEach(function (a) {
      var href = a.getAttribute("href").split("/").pop();
      if (href === current) a.classList.add("active");
    });
  }

  function initDebugCycle() {
    var page = document.querySelector(".scan-page[data-debug-cycle]");
    if (!page) return;
    var label = document.getElementById("stage-label");
    var original = page.querySelector(".cycle-img.original");
    var debugged = page.querySelector(".cycle-img.debugged");
    if (!original || !debugged) return;

    var DEBUG_LABEL = "debug-image.py, confidence overlay";
    var showingOriginal = false;
    var second = 1000;

    setInterval(function () {
      showingOriginal = !showingOriginal;
      original.style.opacity = showingOriginal ? "1" : "0";
      debugged.style.opacity = showingOriginal ? "0" : "1";
      if (label && label.dataset.cycleOriginal) {
        label.textContent = showingOriginal ? label.dataset.cycleOriginal : DEBUG_LABEL;
      }
    }, 3 * second);
  }
})();
