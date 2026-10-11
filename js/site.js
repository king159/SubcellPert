(() => {
  "use strict";

  const header = document.querySelector("[data-header]");
  const menuButton = document.querySelector("[data-menu-toggle]");
  const navigation = document.querySelector("[data-nav]");
  const mobileNavigation = window.matchMedia("(max-width: 900px)");

  const setHeaderState = () => {
    header?.classList.toggle("is-scrolled", window.scrollY > 12);
  };

  setHeaderState();
  window.addEventListener("scroll", setHeaderState, { passive: true });

  const closeMenu = ({ returnFocus = false } = {}) => {
    if (!menuButton || !navigation) return;
    const wasOpen = menuButton.getAttribute("aria-expanded") === "true";
    menuButton.setAttribute("aria-expanded", "false");
    menuButton.setAttribute("aria-label", "Open navigation");
    navigation.classList.remove("is-open");
    document.body.classList.remove("menu-open");
    navigation.inert = mobileNavigation.matches;
    if (mobileNavigation.matches) navigation.setAttribute("aria-hidden", "true");
    else navigation.removeAttribute("aria-hidden");
    if (returnFocus && wasOpen) menuButton.focus();
  };

  menuButton?.addEventListener("click", () => {
    const opening = menuButton.getAttribute("aria-expanded") !== "true";
    menuButton.setAttribute("aria-expanded", String(opening));
    menuButton.setAttribute("aria-label", opening ? "Close navigation" : "Open navigation");
    navigation?.classList.toggle("is-open", opening);
    document.body.classList.toggle("menu-open", opening);
    if (navigation) {
      navigation.inert = !opening;
      navigation.setAttribute("aria-hidden", String(!opening));
      if (opening) window.requestAnimationFrame(() => navigation.querySelector("a")?.focus());
    }
  });

  navigation?.querySelectorAll("a").forEach((link) => link.addEventListener("click", closeMenu));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeMenu({ returnFocus: true });
      return;
    }
    if (
      event.key !== "Tab" ||
      !mobileNavigation.matches ||
      menuButton?.getAttribute("aria-expanded") !== "true"
    ) return;
    const focusable = [menuButton, ...(navigation?.querySelectorAll("a[href]") || [])].filter(Boolean);
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  });
  mobileNavigation.addEventListener("change", () => closeMenu());
  closeMenu();

  const navLinks = [...document.querySelectorAll('.site-nav a[href^="#"]')];
  const navSections = navLinks
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);

  if ("IntersectionObserver" in window) {
    const activeSectionObserver = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (!visible) return;
        navLinks.forEach((link) => {
          const active = link.getAttribute("href") === `#${visible.target.id}`;
          link.classList.toggle("is-active", active);
          if (active) link.setAttribute("aria-current", "location");
          else link.removeAttribute("aria-current");
        });
      },
      { rootMargin: "-20% 0px -68%", threshold: [0, 0.1, 0.5] },
    );
    navSections.forEach((section) => activeSectionObserver.observe(section));
  }

  const seededRandom = (seed) => {
    let state = seed >>> 0;
    return () => {
      state += 0x6d2b79f5;
      let value = state;
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
  };

  const dotColors = ["#6957db", "#18a999", "#ef6f91", "#d9a224"];
  const morph = document.querySelector("[data-rna-morph]");
  if (morph) {
    const random = seededRandom(20260908);
    const sourceDots = morph.querySelector("[data-morph-source-dots]");
    const targetDots = morph.querySelector("[data-morph-target-dots]");
    const axes = morph.querySelector("[data-morph-axes]");
    const sourceCell = morph.querySelector("[data-morph-source-cell]");
    const targetCell = morph.querySelector("[data-morph-target-cell]");
    const cellLabels = morph.querySelector("[data-morph-cell-labels]");
    const arrow = morph.querySelector("[data-morph-arrow]");
    const arrowPath = arrow?.querySelector("path");
    const countLabel = morph.querySelector("[data-count-label]");
    const spaceLabel = morph.querySelector("[data-space-label]");
    const motionPreference = window.matchMedia("(prefers-reduced-motion: reduce)");
    const particles = [];

    const pointInEllipse = (cx, cy, rx, ry) => {
      const angle = random() * Math.PI * 2;
      const radius = Math.sqrt(.12 + random() * .76);
      return {
        x: cx + Math.cos(angle) * rx * radius,
        y: cy + Math.sin(angle) * ry * radius,
      };
    };

    const sourcePatterns = [
      { cx: 166, cy: 150, rx: 45, ry: 45 },
      { cx: 266, cy: 224, rx: 46, ry: 56 },
      { cx: 168, cy: 252, rx: 40, ry: 40 },
      { cx: 274, cy: 136, rx: 40, ry: 36 },
    ];
    const targetPatterns = [
      { cx: 624, cy: 158, rx: 45, ry: 46 },
      { cx: 724, cy: 226, rx: 47, ry: 57 },
      { cx: 622, cy: 255, rx: 40, ry: 39 },
      { cx: 738, cy: 139, rx: 40, ry: 36 },
    ];

    const pointInTargetCell = () => {
      const angle = random() * Math.PI * 2;
      const radius = Math.sqrt(random());
      return {
        x: 680 + Math.cos(angle) * 112 * radius,
        y: 200 + Math.sin(angle) * 124 * radius,
      };
    };

    const createParticles = (counts, barXs, patterns, group, randomizeTarget = false) => {
      counts.forEach((count, gene) => {
        const { cx, cy, rx, ry } = patterns[gene];
        for (let index = 0; index < count; index += 1) {
          const start = {
            x: barXs[gene] + (index % 5) * 7,
            y: 306 - Math.floor(index / 5) * 14,
          };
          const target = randomizeTarget ? pointInTargetCell() : pointInEllipse(cx, cy, rx, ry);
          const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          dot.setAttribute("class", "rna-dot");
          dot.setAttribute("cx", start.x.toFixed(2));
          dot.setAttribute("cy", start.y.toFixed(2));
          dot.setAttribute("r", (2.6 + random() * 1.2).toFixed(2));
          dot.setAttribute("fill", dotColors[gene]);
          dot.style.animationDelay = `${(-random() * 4).toFixed(2)}s`;
          group.append(dot);
          particles.push({
            dot,
            start,
            target,
            delay: random() * 850,
            drift: { x: (random() - 0.5) * 34, y: (random() - 0.5) * 42 },
          });
        }
      });
    };

    createParticles([24, 30, 18, 24], [118, 184, 250, 316], sourcePatterns, sourceDots);
    createParticles([16, 38, 26, 20], [568, 634, 700, 766], targetPatterns, targetDots, true);

    const ease = (value) => value * value * (3 - 2 * value);
    const clamp = (value) => Math.max(0, Math.min(1, value));
    const animationDuration = 6800;
    const render = (elapsed) => {
      const arrowProgress = ease(clamp((elapsed - 450) / 1000));
      const progress = ease(clamp((elapsed - 1350) / 3400));
      axes?.setAttribute("opacity", String(1 - ease(clamp((elapsed - 1100) / 1200))));
      sourceCell?.setAttribute("opacity", String(progress));
      targetCell?.setAttribute("opacity", String(progress));
      cellLabels?.setAttribute("opacity", String(progress));
      arrow?.setAttribute("opacity", String(.55 + arrowProgress * .45));
      if (arrowPath) arrowPath.style.strokeDashoffset = String(32 * (1 - arrowProgress));
      if (countLabel) countLabel.style.opacity = String(1 - progress * .55);
      if (spaceLabel) spaceLabel.style.opacity = String(.45 + progress * .55);
      particles.forEach(({ dot, start, target, delay, drift }) => {
        const t = ease(clamp((elapsed - 1400 - delay) / 3400));
        const arc = Math.sin(Math.PI * t);
        dot.setAttribute("cx", (start.x + (target.x - start.x) * t + arc * drift.x).toFixed(2));
        dot.setAttribute("cy", (start.y + (target.y - start.y) * t - arc * 24 + arc * drift.y).toFixed(2));
      });
    };
    let frame;
    const play = () => {
      cancelAnimationFrame(frame);
      if (motionPreference.matches) { render(animationDuration); return; }
      const start = performance.now();
      render(0);
      const tick = (now) => {
        const phase = (now - start) % (animationDuration * 2);
        render(phase <= animationDuration ? phase : animationDuration * 2 - phase);
        frame = requestAnimationFrame(tick);
      };
      frame = requestAnimationFrame(tick);
    };
    motionPreference.addEventListener("change", play);
    play();
  }

  const compactNumber = new Intl.NumberFormat("en", {
    notation: "compact",
    maximumFractionDigits: 2,
  });
  const counterElements = [...document.querySelectorAll("[data-count]")];

  const finishCounter = (element) => {
    const value = Number(element.dataset.count);
    element.textContent = element.dataset.format === "compact" ? compactNumber.format(value) : value.toLocaleString("en");
  };

  counterElements.forEach(finishCounter);

  const filterButtons = [...document.querySelectorAll("[data-filter]")];
  const datasetCards = [...document.querySelectorAll("[data-track]")];
  const datasetGroupHeadings = [...document.querySelectorAll("[data-track-heading]")];
  filterButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const filter = button.dataset.filter;
      filterButtons.forEach((candidate) => {
        const active = candidate === button;
        candidate.classList.toggle("is-active", active);
        candidate.setAttribute("aria-pressed", String(active));
      });
      datasetCards.forEach((card) => {
        const hidden = filter !== "all" && card.dataset.track !== filter;
        card.classList.toggle("is-filtered", hidden);
        card.setAttribute("aria-hidden", String(hidden));
      });
      datasetGroupHeadings.forEach((heading) => {
        const hidden = filter !== "all" && heading.dataset.trackHeading !== filter;
        heading.hidden = hidden;
        heading.setAttribute("aria-hidden", String(hidden));
      });
    });
  });

  const figureDialog = document.querySelector("[data-figure-dialog]");
  const openFigureButton = document.querySelector("[data-open-figure]");
  const closeFigureButton = document.querySelector("[data-close-figure]");
  const openFigure = () => {
    if (!figureDialog) return;
    if (typeof figureDialog.showModal === "function") figureDialog.showModal();
    else figureDialog.setAttribute("open", "");
  };
  const closeFigure = () => {
    if (!figureDialog) return;
    if (typeof figureDialog.close === "function") figureDialog.close();
    else figureDialog.removeAttribute("open");
  };
  openFigureButton?.addEventListener("click", openFigure);
  closeFigureButton?.addEventListener("click", closeFigure);
  figureDialog?.addEventListener("click", (event) => {
    if (event.target === figureDialog) closeFigure();
  });

  const toast = document.querySelector("[data-toast]");
  let toastTimer;
  const showToast = (message) => {
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add("is-visible");
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => toast.classList.remove("is-visible"), 1800);
  };

  const copyText = async (text) => {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return;
    }
    const input = document.createElement("textarea");
    input.value = text;
    input.setAttribute("readonly", "");
    input.style.position = "fixed";
    input.style.opacity = "0";
    document.body.append(input);
    input.select();
    const copied = document.execCommand("copy");
    input.remove();
    if (!copied) throw new Error("Copy command failed");
  };

  document.querySelector("[data-copy-title]")?.addEventListener("click", async () => {
    try {
      await copyText("SubcellPert: Benchmarking Subcellular Resolution Perturbation Prediction");
      showToast("Title copied to clipboard");
    } catch {
      showToast("Copy unavailable — select the title above");
    }
  });

  const resultData = window.SUBCELLPERT_RESULTS;
  const datasetSelect = document.querySelector("[data-result-dataset]");
  const panelSelect = document.querySelector("[data-result-panel]");
  if (resultData && datasetSelect && panelSelect) {
    const metricOrder = ["oracle_w1", "joint_w1", "coverage", "mse", "pcc_delta"];
    const metricInfo = {
      oracle_w1: {
        title: "Oracle spatial W1", direction: "Lower is better",
        description: "Intracellular localization error given exact target counts. All target-present genes must have geometry.",
      },
      joint_w1: {
        title: "Joint spatial W1 · recovered genes", direction: "Lower error · check coverage",
        description: "Conditional error on the genes each method recovers. Gene support differs across methods; compare coverage and abundance before interpreting a spatial gain.",
      },
      coverage: {
        title: "Joint target-gene coverage", direction: "Higher is better",
        description: "Fraction of target-present genes with predicted geometry. Missing genes are not assigned zero spatial error.",
      },
      mse: {
        title: "Joint abundance MSE", direction: "Lower is better",
        description: "Mean squared error between predicted and observed mean log1p gene counts, including zero-expression genes.",
      },
      pcc_delta: {
        title: "Joint PCC-delta", direction: "Higher is better",
        description: "Pearson correlation of source-relative abundance changes. Correlated response directions can still have poorly calibrated magnitudes.",
      },
    };
    const metricButtons = [...document.querySelectorAll("[data-result-metric]")];
    const rowsElement = document.querySelector("[data-score-rows]");
    const barsElement = document.querySelector("[data-score-bars]");
    let selectedMetric = "oracle_w1";
    const number = (value, digits = 3) => Number(value.toPrecision(digits)).toString();
    const format = (score, metric) => {
      if (score.mean === null) return { mean: "—", sd: "", spoken: "Not available" };
      const mean = metric === "coverage"
        ? (score.mean === 1 ? "100%" : score.mean > .99995 ? ">99.99%" : `${(score.mean * 100).toFixed(2)}%`)
        : number(score.mean);
      const sd = score.sd === null ? "" : metric === "coverage"
        ? `± ${number(score.sd * 100, 2)} pp` : `± ${number(score.sd, 2)}`;
      return { mean, sd, spoken: `${mean} ${sd}`.trim() };
    };
    const element = (tag, className, text) => {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined) node.textContent = text;
      return node;
    };
    const renderResults = () => {
      const dataset = resultData.datasets.find((candidate) => candidate.id === datasetSelect.value);
      const rows = dataset.panels[panelSelect.value];
      const panelLabel = panelSelect.value === "all_gene" ? "All genes" : "Top-100 DEGs";
      const splitCount = rows[0].splits;
      document.querySelector("[data-result-context]").textContent =
        `${dataset.label} · ${dataset.axis} · ${panelLabel} · ${splitCount} data ${splitCount === 1 ? "split" : "splits"}`;
      document.querySelector("[data-score-caption]").textContent =
        `${dataset.label} · ${panelLabel} · scores for all eight baselines`;
      const info = metricInfo[selectedMetric];
      document.querySelector("[data-chart-title]").textContent = info.title;
      document.querySelector("[data-chart-direction]").textContent = info.direction;
      document.querySelector("[data-chart-description]").textContent = info.description;
      metricButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.resultMetric === selectedMetric)));

      const scores = rows.map((row) => row[selectedMetric]).filter((score) => score.mean !== null);
      const min = Math.min(0, ...scores.map((score) => score.mean - (score.sd || 0)));
      const max = Math.max(selectedMetric === "coverage" ? 1 : 0, ...scores.map((score) => score.mean + (score.sd || 0)));
      const extent = max - min || 1;
      const position = (value) => ((value - min) / extent) * 100;
      const zero = position(0);
      const spatialJoint = selectedMetric === "joint_w1";
      barsElement.replaceChildren();
      rowsElement.replaceChildren();
      rows.forEach((row, index) => {
        const model = resultData.models.find((candidate) => candidate.id === row.model);
        const score = row[selectedMetric];
        const value = format(score, selectedMetric);
        const coverage = format(row.coverage, "coverage").mean;
        const barRow = element("div", `score-bar-row${index < 3 ? " is-classical" : ""}`);
        const label = element("span", "score-bar-label", model.label);
        if (spatialJoint) label.append(element("small", "", `${coverage} gene coverage`));
        barRow.append(label);
        const track = element("div", "score-bar-track");
        track.setAttribute("aria-hidden", "true");
        track.style.setProperty("--zero", `${zero}%`);
        if (score.mean !== null) {
          const bar = element("span", "score-bar-fill");
          const end = position(score.mean);
          bar.style.left = `${Math.min(zero, end)}%`;
          bar.style.width = `${Math.abs(end - zero)}%`;
          track.append(bar);
          if (score.sd !== null) {
            const whisker = element("span", "score-bar-whisker");
            whisker.style.left = `${position(score.mean - score.sd)}%`;
            whisker.style.width = `${score.sd * 2 / extent * 100}%`;
            track.append(whisker);
          }
        } else track.append(element("span", "score-bar-missing", "Undefined"));
        barRow.append(track);
        const scoreValue = element("span", "score-bar-value", value.mean);
        if (value.sd) scoreValue.append(element("small", "", value.sd));
        if (score.mean === null) scoreValue.setAttribute("aria-label", value.spoken);
        barRow.append(scoreValue);
        barsElement.append(barRow);

        const tableRow = element("tr");
        const heading = element("th", "", model.label);
        heading.scope = "row";
        tableRow.append(heading);
        metricOrder.forEach((metric) => {
          const formatted = format(row[metric], metric);
          const cell = element("td", metric === selectedMetric ? "is-selected" : "", formatted.mean);
          if (formatted.sd) cell.append(element("span", "score-sd", formatted.sd));
          if (row[metric].mean === null) cell.setAttribute("aria-label", formatted.spoken);
          tableRow.append(cell);
        });
        rowsElement.append(tableRow);
      });
      document.querySelector("[data-chart-caption]").textContent =
        scores.length ? `Bars start at zero. Scale: ${number(min)}–${number(max)}${selectedMetric === "coverage" ? " (fraction)" : ""}.` : "No scores are available for this view.";
    };
    datasetSelect.addEventListener("change", renderResults);
    panelSelect.addEventListener("change", renderResults);
    metricButtons.forEach((button) => button.addEventListener("click", () => {
      selectedMetric = button.dataset.resultMetric;
      renderResults();
    }));
    renderResults();
    document.querySelector("[data-result-controls]").hidden = false;
    document.querySelector("[data-metric-picker]").hidden = false;
    document.querySelector("[data-score-chart]").hidden = false;
  }

  const year = document.querySelector("[data-year]");
  if (year) year.textContent = String(new Date().getFullYear());
})();
