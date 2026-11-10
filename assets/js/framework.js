/* Copied from abstractframework.ai assets/js/main.js (gh-pages e6e5471) so abstractcore.ai shares its behaviour. Edit there first. */
document.addEventListener('DOMContentLoaded', () => {
  const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ── Nav scroll effect ── */
  const nav = document.querySelector('.nav');
  let lastScroll = 0;
  window.addEventListener('scroll', () => {
    const y = window.scrollY;
    nav.classList.toggle('scrolled', y > 50);
    lastScroll = y;
  }, { passive: true });

  /* ── Smooth scroll for anchor links ── */
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      const href = a.getAttribute('href');
      if (href === '#') { e.preventDefault(); window.scrollTo({ top: 0, behavior: reduceMotion ? 'auto' : 'smooth' }); return; }
      const el = document.querySelector(href);
      if (el) {
        e.preventDefault();
        history.replaceState(null, '', href);
        const offset = 80;
        const y = el.getBoundingClientRect().top + window.scrollY - offset;
        window.scrollTo({ top: y, behavior: reduceMotion ? 'auto' : 'smooth' });
      }
    });
  });

  /* ── Scroll reveal with stagger ── */
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry, idx) => {
      if (entry.isIntersecting) {
        const delay = entry.target.dataset.delay || 0;
        setTimeout(() => {
          entry.target.classList.add('visible');
        }, parseInt(delay));
        revealObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

  document.querySelectorAll('.reveal').forEach((el, i) => {
    if (!el.dataset.delay && el.parentElement) {
      const siblings = el.parentElement.querySelectorAll(':scope > .reveal');
      const idx = Array.from(siblings).indexOf(el);
      if (idx > 0) el.dataset.delay = idx * 80;
    }
    revealObserver.observe(el);
  });

  /* ── Animated counter for stats ── */
  const counterObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const el = entry.target;
        const target = el.dataset.count;
        if (!target) return;
        if (reduceMotion) { el.textContent = target; counterObserver.unobserve(el); return; }
        const isPercent = target.includes('%');
        const num = parseInt(target);
        const suffix = target.replace(/[\d]/g, '');
        let current = 0;
        const step = Math.max(1, Math.floor(num / 40));
        const timer = setInterval(() => {
          current += step;
          if (current >= num) {
            current = num;
            clearInterval(timer);
          }
          el.textContent = current + suffix;
        }, 30);
        counterObserver.unobserve(el);
      }
    });
  }, { threshold: 0.5 });

  document.querySelectorAll('.stat-num[data-count]').forEach(el => {
    counterObserver.observe(el);
  });

  /* ── Terminal look for shell blocks ──
     A block is shown as a terminal unless it reads as source code (an import, a definition,
     an assignment or a method call at the start of a line). data-lang="shell|code" overrides. */
  const CODE_LINE = /^\s*(from\s+[\w.]+\s+import\b|import\s+[\w.]|def\s|class\s|@\w|const\s|let\s|[A-Za-z_][\w.]*\s*=[^=]|[A-Za-z_]\w*(\.\w+)+\(|[{[])/m;
  document.querySelectorAll('.code-block').forEach(block => {
    const lang = block.dataset.lang;
    const text = (block.querySelector('code') || block).textContent || '';
    const shell = lang ? lang === 'shell' : !CODE_LINE.test(text);
    block.classList.toggle('term', shell);
  });

  /* ── Copy buttons: an icon, a brief "Copied" state, announced to screen readers ── */
  const ICON_COPY = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>';
  const ICON_DONE = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"/></svg>';
  const liveRegion = document.createElement('div');
  liveRegion.className = 'sr-only';
  liveRegion.setAttribute('aria-live', 'polite');
  document.body.appendChild(liveRegion);
  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    return new Promise((resolve, reject) => {
      const ta = document.createElement('textarea');
      ta.value = text; ta.setAttribute('readonly', ''); ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta); ta.select();
      try { document.execCommand('copy') ? resolve() : reject(new Error('copy refused')); }
      catch (err) { reject(err); } finally { ta.remove(); }
    });
  }
  function addCopyButton(block, getText) {
    if (block.querySelector(':scope > .copy-btn')) return;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'copy-btn';
    btn.innerHTML = ICON_COPY + '<span class="copy-tip" aria-hidden="true">Copied</span>';
    btn.setAttribute('aria-label', 'Copy');
    let timer = null;
    btn.addEventListener('click', () => {
      copyText(getText()).then(() => {
        btn.classList.add('copied');
        btn.innerHTML = ICON_DONE + '<span class="copy-tip" aria-hidden="true">Copied</span>';
        btn.setAttribute('aria-label', 'Copied');
        liveRegion.textContent = 'Copied to the clipboard';
        clearTimeout(timer);
        timer = setTimeout(() => {
          btn.classList.remove('copied');
          btn.innerHTML = ICON_COPY + '<span class="copy-tip" aria-hidden="true">Copied</span>';
          btn.setAttribute('aria-label', 'Copy');
          liveRegion.textContent = '';
        }, 1600);
      }).catch(() => { liveRegion.textContent = 'Copy failed: select the text and copy it'; });
    });
    if (getComputedStyle(block).position === 'static') block.style.position = 'relative';
    block.appendChild(btn);
  }
  document.querySelectorAll('.code-block, .hero-code').forEach(block => {
    addCopyButton(block, () => {
      const code = block.querySelector('code') || block.querySelector('pre') || block;
      return (block.dataset.copy || code.textContent).trim();
    });
  });

  /* ── Quick start: OS segmented control over one terminal line ── */
  document.querySelectorAll('.qs').forEach(qs => {
    const tabs = Array.from(qs.querySelectorAll('.qs-tab'));
    const cmd = qs.querySelector('.qs-cmd');
    const prompt = qs.querySelector('.qs-prompt');
    const note = qs.querySelector('.qs-note-text');
    const term = qs.querySelector('.qs-term');
    function select(tab, focus) {
      tabs.forEach(t => {
        const on = t === tab;
        t.setAttribute('aria-selected', on ? 'true' : 'false');
        t.tabIndex = on ? 0 : -1;
      });
      cmd.textContent = tab.dataset.cmd;
      if (prompt) prompt.textContent = tab.dataset.prompt || '$';
      if (note && tab.dataset.note) note.innerHTML = tab.dataset.note;
      term.setAttribute('aria-labelledby', tab.id);
      qs.style.setProperty('--qs-i', tabs.indexOf(tab));
      /* dependent content: every [data-follows="qs"] block shows only its [data-when] variant for this OS */
      qs.querySelectorAll('[data-follows="qs"] [data-when]').forEach(el => {
        el.hidden = !el.dataset.when.split(/\s+/).includes(tab.dataset.os);
      });
      if (focus) tab.focus();
    }
    tabs.forEach((tab, i) => {
      tab.addEventListener('click', () => select(tab, false));
      tab.addEventListener('keydown', e => {
        let j = null;
        if (e.key === 'ArrowRight') j = (i + 1) % tabs.length;
        else if (e.key === 'ArrowLeft') j = (i - 1 + tabs.length) % tabs.length;
        else if (e.key === 'Home') j = 0;
        else if (e.key === 'End') j = tabs.length - 1;
        if (j !== null) { e.preventDefault(); select(tabs[j], true); }
      });
    });
    const ua = navigator.userAgent || '';
    const os = /Windows/i.test(ua) ? 'win' : (/Android/i.test(ua) ? 'mac' : (/Linux|X11|CrOS/i.test(ua) ? 'linux' : 'mac'));
    select(tabs.find(t => t.dataset.os === os) || tabs[0], false);
    addCopyButton(term, () => cmd.textContent.trim());
  });

  /* ── Mobile menu toggle ── */
  const toggle = document.querySelector('.mobile-toggle');
  const links = document.querySelector('.nav-links');
  if (toggle && links) {
    toggle.setAttribute('aria-expanded', 'false');
    toggle.addEventListener('click', () => {
      links.classList.toggle('open');
      toggle.setAttribute('aria-expanded', links.classList.contains('open'));
    });
    links.querySelectorAll('a').forEach(a => {
      a.addEventListener('click', () => { links.classList.remove('open'); toggle.setAttribute('aria-expanded', 'false'); });
    });
  }

  /* ── Image lightbox (mouse and keyboard; focus moves in and back) ── */
  document.querySelectorAll('.showcase-img img, .gallery-item img, .get-card .shot img, .figure img, .j-body img, .mm-art img').forEach(img => {
    img.style.cursor = 'zoom-in';
    img.tabIndex = 0;
    img.setAttribute('role', 'button');
    img.setAttribute('aria-label', 'Enlarge: ' + (img.alt || 'image'));
    function open() {
      const overlay = document.createElement('div');
      overlay.className = 'lightbox-overlay';
      overlay.tabIndex = -1;
      const big = document.createElement('img');
      big.src = img.currentSrc || img.src;
      big.alt = img.alt || '';
      overlay.setAttribute('role', 'dialog');
      overlay.setAttribute('aria-modal', 'true');
      overlay.setAttribute('aria-label', img.alt || 'Image');
      const closeBtn = document.createElement('button');
      closeBtn.type = 'button';
      closeBtn.className = 'lightbox-close';
      closeBtn.setAttribute('aria-label', 'Close');
      closeBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" aria-hidden="true"><path d="M18 6L6 18M6 6l12 12"/></svg>';
      overlay.appendChild(big);
      overlay.appendChild(closeBtn);
      function close() { overlay.remove(); document.removeEventListener('keydown', esc); img.focus(); }
      function esc(e) {
        if (e.key === 'Escape') { e.preventDefault(); close(); }
        else if (e.key === 'Tab') { e.preventDefault(); closeBtn.focus(); }  /* focus stays inside the dialog */
      }
      overlay.addEventListener('click', close);
      document.addEventListener('keydown', esc);
      document.body.appendChild(overlay);
      closeBtn.focus();
    }
    img.addEventListener('click', open);
    img.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); open(); } });
  });

  /* ── Tab system (ARIA tabs, arrow keys) ── */
  document.querySelectorAll('.code-tabs').forEach(tabs => {
    const nav = tabs.querySelector('.tab-nav');
    const buttons = Array.from(tabs.querySelectorAll('.tab-btn'));
    const panels = tabs.querySelectorAll('.tab-panel');
    if (nav) nav.setAttribute('role', 'tablist');
    function activate(btn, focus) {
      const target = btn.dataset.tab;
      buttons.forEach(b => {
        const on = b === btn;
        b.classList.toggle('active', on);
        b.setAttribute('aria-selected', on ? 'true' : 'false');
        b.tabIndex = on ? 0 : -1;
      });
      panels.forEach(p => p.classList.toggle('active', p.id === 'tab-' + target));
      if (focus) btn.focus();
    }
    buttons.forEach((btn, i) => {
      btn.setAttribute('role', 'tab');
      btn.setAttribute('aria-controls', 'tab-' + btn.dataset.tab);
      btn.id = btn.id || ('tabbtn-' + btn.dataset.tab);
      const panel = tabs.querySelector('#tab-' + btn.dataset.tab);
      if (panel) { panel.setAttribute('role', 'tabpanel'); panel.setAttribute('aria-labelledby', btn.id); }
      btn.setAttribute('aria-selected', btn.classList.contains('active') ? 'true' : 'false');
      btn.tabIndex = btn.classList.contains('active') ? 0 : -1;
      btn.addEventListener('click', () => activate(btn, false));
      btn.addEventListener('keydown', e => {
        if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
          e.preventDefault();
          const next = buttons[(i + (e.key === 'ArrowRight' ? 1 : -1) + buttons.length) % buttons.length];
          activate(next, true);
        }
      });
    });
    /* Pre-select a tab from the page's platform when asked to */
    if (tabs.dataset.autoplatform !== undefined) {
      const ua = navigator.userAgent || '';
      const want = /Windows/i.test(ua) ? tabs.dataset.win
        : (/Android/i.test(ua) ? null : (/Linux|X11|CrOS/i.test(ua) ? tabs.dataset.linux : null));
      const btn = want && buttons.find(b => b.dataset.tab === want);
      if (btn) activate(btn, false);
    }
  });

  /* ── Pause every video that leaves the screen or is not the visible slide ── */
  /* Video playback (autoplay in view, pause out of view, reduced motion) is handled by carousel3.js. */

  /* ── Carousel ── */
  document.querySelectorAll('.carousel').forEach(carousel => {
    const track = carousel.querySelector('.carousel-track');
    const slides = carousel.querySelectorAll('.carousel-slide');
    const prevBtn = carousel.querySelector('.carousel-prev');
    const nextBtn = carousel.querySelector('.carousel-next');
    const dotsContainer = carousel.querySelector('.carousel-dots');
    let current = 0;
    const total = slides.length;
    slides.forEach((_, i) => {
      const dot = document.createElement('button');
      dot.className = 'carousel-dot' + (i === 0 ? ' active' : '');
      dot.setAttribute('aria-label', 'Go to slide ' + (i + 1));
      dot.addEventListener('click', () => goTo(i));
      dotsContainer.appendChild(dot);
    });
    const dots = dotsContainer.querySelectorAll('.carousel-dot');
    function goTo(index) {
      slides[current].querySelectorAll('video').forEach(v => v.pause());
      current = ((index % total) + total) % total;
      track.style.transform = 'translateX(-' + (current * 100) + '%)';
      dots.forEach((d, i) => d.classList.toggle('active', i === current));
    }
    prevBtn.addEventListener('click', () => goTo(current - 1));
    nextBtn.addEventListener('click', () => goTo(current + 1));
    let autoplay = reduceMotion ? null : setInterval(() => goTo(current + 1), 6000);
    carousel.addEventListener('mouseenter', () => clearInterval(autoplay));
    carousel.addEventListener('mouseleave', () => {
      if (!reduceMotion) autoplay = setInterval(() => goTo(current + 1), 6000);
    });
    carousel.addEventListener('focusin', () => clearInterval(autoplay));
    let startX = 0;
    carousel.addEventListener('touchstart', e => { startX = e.touches[0].clientX; }, { passive: true });
    carousel.addEventListener('touchend', e => {
      const dx = e.changedTouches[0].clientX - startX;
      if (Math.abs(dx) > 50) goTo(current + (dx > 0 ? -1 : 1));
    }, { passive: true });
  });

  /* ── Parallax on hero bg ── */
  const heroBg = document.querySelector('.hero-bg-img');
  if (heroBg && !reduceMotion) {
    window.addEventListener('scroll', () => {
      const y = window.scrollY;
      if (y < window.innerHeight) {
        heroBg.style.transform = `translateY(${y * 0.3}px) scale(1.1)`;
      }
    }, { passive: true });
  }

  /* ── Interactive isometric cube architecture ── */
  const cubeGrid = document.getElementById('cubeGrid');
  if (cubeGrid) {
    const CUBE_SIZE = 48, COL_STEP = 90, ROW_STEP = 110;
    /* Rows mirror the three-layer positioning (library, durable runtime, control plane) with the
       applications on top and the reusable toolkits underneath; the colour of a cube is its type. */
    const cubesData = [
      { id:'flow',      type:'app',        col:-2.55,row:0, label:'Flow',      name:'AbstractFlow',      typeName:'Application',       href:'flow.html',      desc:'The visual workflow editor: draw a graph that mixes agent steps with deterministic nodes; the gateway installs it as a versioned .flow bundle that runs on AbstractRuntime.' },
      { id:'code',      type:'app',        col:-1.7, row:0, label:'Code',      name:'AbstractCode',      typeName:'Application',       href:'code.html',      desc:'A coding agent that runs durably on the gateway, with a terminal client and a browser client: tool approvals, workspace files, streamed replies and automations.' },
      { id:'observer',  type:'app',        col:-0.85,row:0, label:'Observer',  name:'AbstractObserver',  typeName:'Application',       href:'observer.html',  desc:'Watch runs live, replay the ledger step by step, and create and manage automations.' },
      { id:'assistant', type:'app',        col:0,    row:0, label:'Assistant', name:'AbstractAssistant', typeName:'Application',       href:'assistant.html', desc:'Your assistant one click away on the desktop (macOS, Linux, Windows), and over the same conversations in a browser (private preview) and on iPhone and Apple Watch (pre-release): a palette, voice, and your gateway sessions and automations.' },
      { id:'continuum', type:'app',        col:0.85, row:0, label:'Continuum', name:'AbstractContinuum', typeName:'Application',       href:'continuum.html', desc:'A board-first console for continuous development and deployment work.' },
      { id:'entity',    type:'app',        col:1.7,  row:0, label:'Entity',    name:'AbstractEntity',    typeName:'Application',       href:'entity.html',    desc:'Persistent, self-evolving agents with an identity, a purpose, a history and experiences: entities.' },
      { id:'note',      type:'app',        col:2.55, row:0, label:'Note',      name:'AbstractNote',      typeName:'Application (experimental)', href:'note.html', desc:'Experimental: a notebook on your gateway that files your notes, links them into a knowledge graph in AbstractMemory and answers by effort. Installed beside the framework, not with it.' },
      { id:'gateway',   type:'control',    col:0,    row:1, label:'Gateway',   name:'AbstractGateway',   typeName:'Control Plane',     href:'gateway.html',   desc:'The controller: starts, resumes and cancels durable runs over HTTP and SSE, runs automations, manages users and network access, and serves the apps, the web console and the abstractgateway-console terminal console.' },
      { id:'runtime',   type:'foundation', col:-2,   row:2, label:'Runtime',   name:'AbstractRuntime',   typeName:'Foundation',        href:'runtime.html',   desc:'Where agentic operations run: durable runs, effects and waits, checkpoint and resume, with an append-only ledger.' },
      { id:'agent',     type:'compose',    col:-1,   row:2, label:'Agent',     name:'AbstractAgent',     typeName:'Composition',       href:'agent.html',     desc:'Agent patterns (ReAct, CodeAct, MemAct) run durably by AbstractRuntime, with tool approval.' },
      { id:'memory',    type:'knowledge',  col:1,    row:2, label:'Memory',    name:'AbstractMemory',    typeName:'Knowledge',         href:'memory.html',    desc:'Durable, append-only agent memory: temporal triples, and a memory system that forms, recalls and consolidates records from use.' },
      { id:'semantics', type:'knowledge',  col:2,    row:2, label:'Semantics', name:'AbstractSemantics', typeName:'Knowledge',         href:'semantics.html', desc:'The shared vocabulary: predicates, entity types and memory relations, with JSON Schema helpers.' },
      { id:'skill',     type:'compose',    col:0,    row:2, label:'Skill',     name:'AbstractSkill',     typeName:'Composition',       href:'https://github.com/lpalbou/AbstractSkill', desc:'Agent Skills (SKILL.md): parsing, validation, the curated skill shelf and the trust gate the gateway applies before a skill reaches a run.' },
      { id:'core',      type:'foundation', col:-2.5, row:3, label:'Core',      name:'AbstractCore',      typeName:'Foundation',        href:'core.html',      desc:'One Python API over ten provider types, local and cloud: tools, structured output, media input, capability plugins, decision models, an OpenAI-compatible server with a web console, and abstractcore-console in a terminal.' },
      { id:'voice',     type:'plugin',     col:-1.5, row:3, label:'Voice',     name:'AbstractVoice',     typeName:'Capability Plugin', href:'voice.html',     desc:'Text-to-speech, speech-to-text and voice cloning, local or remote.' },
      { id:'music',     type:'plugin',     col:-0.5, row:3, label:'Music',     name:'AbstractMusic',     typeName:'Capability Plugin', href:'music.html',     desc:'Text-to-music and text-to-audio: ACE-Step and Stable Audio locally, ACE Music and ElevenLabs remotely.' },
      { id:'vision',    type:'plugin',     col:0.5,  row:3, label:'Vision',    name:'AbstractVision',    typeName:'Capability Plugin', href:'vision.html',    desc:'Image generation, editing and upscaling, text-to-video and image-to-video, through MLX-Gen, Diffusers, stable-diffusion.cpp or OpenAI-compatible services.' },
      { id:'scene3d',   type:'plugin',     col:1.5,  row:3, label:'3D',        name:'Abstract3D',        typeName:'Capability Plugin', href:'3d.html',        desc:'Image-to-3D and text-to-3D with a validated local TripoSR backend and GLB output.' },
      { id:'camera',    type:'plugin',     col:2.5,  row:3, label:'Camera',    name:'AbstractCamera',    typeName:'Capability Plugin', href:'camera.html',    desc:'Camera control: tethered bodies, webcams and smart telescopes behind one manager, with camera tools for agents.' },
      { id:'tui',       type:'toolkit',    col:-1.5, row:4, label:'TUI',       name:'AbstractTUI',       typeName:'Toolkit',           href:'tui.html', desc:'The reactive Rust terminal-UI engine behind the AbstractCode terminal client and both terminal consoles. Build your own terminal apps on it (crates.io abstracttui).' },
      { id:'uic',       type:'toolkit',    col:-0.5, row:4, label:'UIC',       name:'AbstractUIC',       typeName:'Toolkit',           href:'uic.html', desc:'React components, web components and a gateway session proxy used by the browser apps. Build your own web apps on it (npm @abstractframework/ui-kit).' },
      { id:'avatar',    type:'toolkit',    col:0.5,  row:4, label:'Avatar',    name:'AbstractAvatar',    typeName:'Toolkit',           href:'avatar.html', desc:'Procedural activity avatars from one Rust generator: 44 designs and seven states for terminal, browser and native apps (crates.io and PyPI abstractavatar).' },
      { id:'mlxgen',    type:'toolkit',    col:1.5,  row:4, label:'MLX-Gen',   name:'MLX-Gen',           typeName:'Toolkit',           href:'https://github.com/lpalbou/mlx-gen', desc:'Generative image and video model runtimes for MLX: the Apple silicon engine behind AbstractVision.' },
    ];
    const typeColors = { app:'#34d399', control:'#22d3ee', compose:'#a78bfa', foundation:'#818cf8', plugin:'#f472b6', knowledge:'#fbbf24', toolkit:'#a1a1b5' };

    cubesData.forEach(c => {
      const cx = c.col * COL_STEP, cy = c.row * ROW_STEP;
      const wrap = document.createElement('div');
      wrap.className = 'cube-wrap';
      wrap.dataset.id = c.id;
      wrap.dataset.layer = c.type;
      wrap.dataset.row = c.row;
      wrap.style.cssText = '--size:'+CUBE_SIZE+'px;--tx:'+cx+'px;--ty:'+cy+'px;transform:translate('+cx+'px,'+cy+'px)';
      wrap.innerHTML = '<div class="cube"><div class="face top"></div><div class="face left"></div><div class="face right"></div></div>'
        + '<div class="cube-tooltip"><div class="tip-layer" style="color:'+typeColors[c.type]+'">'+c.typeName+'</div>'
        + '<div class="tip-name">'+c.name+'</div><div class="tip-desc">'+c.desc+'</div></div>'
        + '<div class="cube-label">'+c.label+'</div>';
      wrap.setAttribute('role', 'listitem');
      wrap.tabIndex = 0;
      wrap.setAttribute('aria-label', c.name + ' (' + c.typeName + '): ' + c.desc);
      wrap.addEventListener('click', function(){ window.location.href = c.href; });
      wrap.addEventListener('keydown', function(e){ if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); window.location.href = c.href; } });
      cubeGrid.appendChild(wrap);
    });

    var layerLabels = [
      { row:0, text:'APPLICATIONS', color:'#34d399' },
      { row:1, text:'CONTROL PLANE', color:'#22d3ee' },
      { row:2, text:'DURABLE RUNTIME', color:'#818cf8' },
      { row:3, text:'LIBRARY', color:'#f472b6' },
      { row:4, text:'TOOLKITS', color:'#a1a1b5' },
    ];
    layerLabels.forEach(function(l) {
      var el = document.createElement('div');
      el.className = 'layer-label';
      el.dataset.row = l.row;
      el.style.cssText = 'left:'+(3.4*COL_STEP)+'px;top:'+(l.row*ROW_STEP+14)+'px;color:'+l.color;
      el.textContent = l.text;
      cubeGrid.appendChild(el);
    });

    var cubeWraps = cubeGrid.querySelectorAll('.cube-wrap[data-id]');
    var cubeLabels = cubeGrid.querySelectorAll('.layer-label');
    function highlightCube(id) {
      var activeRow = null;
      cubeWraps.forEach(function(w){ if(w.dataset.id===id) activeRow=w.dataset.row; });
      cubeWraps.forEach(function(w){
        w.classList.toggle('active', w.dataset.id===id);
        w.classList.toggle('dimmed', id && w.dataset.id!==id);
      });
      cubeLabels.forEach(function(l){ l.classList.toggle('highlight', l.dataset.row===activeRow); });
    }
    function clearCubeHighlight() {
      cubeWraps.forEach(function(w){ w.classList.remove('active','dimmed'); });
      cubeLabels.forEach(function(l){ l.classList.remove('highlight'); });
    }
    cubeWraps.forEach(function(w){
      w.addEventListener('mouseenter', function(){ highlightCube(w.dataset.id); });
      w.addEventListener('mouseleave', clearCubeHighlight);
      w.addEventListener('focus', function(){ highlightCube(w.dataset.id); });
      w.addEventListener('blur', clearCubeHighlight);
    });
  }

  /* ── Section navigator: <div class="pkg-nav" data-section-nav> is filled from this page's own
     <section id> + .section-title pairs (label: data-nav-label, else the title, kept short), and the
     current section is highlighted while scrolling. ── */
  document.querySelectorAll('[data-section-nav]').forEach(bar => {
    const items = [];
    document.querySelectorAll('main section[id], body > section[id]').forEach(sec => {
      if (sec.closest('[data-section-nav]')) return;
      const title = sec.querySelector('.section-title, .doc-section h2, h2');
      let label = sec.dataset.navLabel || (title ? title.textContent : '');
      label = label.replace(/\s+/g, ' ').trim();
      if (!sec.dataset.navLabel) {
        label = label.split(/\s[—–]\s|:\s|\.\s/)[0].replace(/\.$/, '');
        if (label.length > 34) {
          const tag = sec.querySelector('.section-label');
          label = tag ? tag.textContent.trim() : label.slice(0, 32).replace(/\s+\S*$/, '') + '…';
        }
      }
      if (label && !items.some(it => it.sec === sec)) items.push({ sec, label });
    });
    if (items.length < 2) { bar.hidden = true; return; }
    const inner = document.createElement('div');
    inner.className = 'pkg-nav-inner';
    const links = items.map(({ sec, label }) => {
      const a = document.createElement('a');
      a.className = 'pkg-nav-link';
      a.href = '#' + sec.id;
      a.textContent = label;
      a.addEventListener('click', e => {
        e.preventDefault();
        history.replaceState(null, '', '#' + sec.id);
        const y = sec.getBoundingClientRect().top + window.scrollY - 110;
        window.scrollTo({ top: y, behavior: reduceMotion ? 'auto' : 'smooth' });
      });
      inner.appendChild(a);
      return a;
    });
    bar.textContent = '';
    bar.appendChild(inner);
    let current = -1;
    function update() {
      const probe = 130;
      let idx = 0;
      items.forEach((it, i) => { if (it.sec.getBoundingClientRect().top <= probe) idx = i; });
      if (idx === current) return;
      current = idx;
      links.forEach((a, i) => {
        a.classList.toggle('active', i === idx);
        if (i === idx) a.setAttribute('aria-current', 'location'); else a.removeAttribute('aria-current');
      });
      const a = links[idx];
      if (a && bar.scrollWidth > bar.clientWidth) {
        const left = a.offsetLeft - bar.clientWidth / 2 + a.offsetWidth / 2;
        bar.scrollTo({ left, behavior: reduceMotion ? 'auto' : 'smooth' });
      }
    }
    window.addEventListener('scroll', update, { passive: true });
    update();
  });

  /* ── Active nav link highlighting ── */
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');
  const navObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        navLinks.forEach(link => {
          link.classList.toggle('active', link.getAttribute('href') === '#' + entry.target.id);
        });
      }
    });
  }, { threshold: 0.3, rootMargin: '-80px 0px -50% 0px' });
  sections.forEach(s => navObserver.observe(s));

});
