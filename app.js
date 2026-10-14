/* ═══════════════════════════════════════════════
   DEPRISK · APPLICATION LOGIC
   ═══════════════════════════════════════════════ */

// ── Slider live bind ──────────────────────────
const sliders = [
  { id: 's-screen',  out: 'v-screen',  fmt: v => v + ' hrs' },
  { id: 's-index',   out: 'v-index',   fmt: v => v },
  { id: 's-sleep',   out: 'v-sleep',   fmt: v => v + ' hrs' },
  { id: 's-quality', out: 'v-quality', fmt: v => v + ' / 5' },
  { id: 's-jetlag',  out: 'v-jetlag',  fmt: v => v + ' hrs' },
];
sliders.forEach(s => {
  const el = document.getElementById(s.id);
  const out = document.getElementById(s.out);
  const refresh = () => { out.textContent = s.fmt(parseFloat(el.value).toFixed(1)); };
  el.addEventListener('input', refresh);
  refresh();
});

// Gender radio styling
document.querySelectorAll('input[name="gender"]').forEach(r => {
  r.addEventListener('change', () => {
    document.querySelectorAll('input[name="gender"]').forEach(x => {
      x.parentElement.querySelector('.gender-btn').style.borderColor = '';
    });
  });
});

// ── Logistic Regression Model ─────────────────
// Coefficients calibrated to match dataset patterns:
// - 16.4% overall prevalence (BDI ≥ 14)
// - ~3× gender difference (F 24.4% vs M 8.6%)
// - Sleep quality strongest correlate (r = 0.38)
// - Screen time, jetlag as additional predictors
const W = {
  intercept:    -1.8,
  screenTime:    0.13,   // hrs/day
  screenIndex:   0.09,   // 1-10 composite
  sleepHours:   -0.30,   // more sleep = lower risk
  sleepQuality: -0.52,   // higher quality = lower risk
  jetlag:        0.22,   // more jetlag = higher risk
  gender:        1.10,   // female = 1 (calibrated to 3× ratio)
};

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function computeRisk(params) {
  const z = W.intercept
    + W.screenTime   * params.screenTime
    + W.screenIndex  * params.screenIndex
    + W.sleepHours   * params.sleepHours
    + W.sleepQuality * params.sleepQuality
    + W.jetlag       * params.jetlag
    + W.gender       * params.gender;
  return sigmoid(z);
}

// ── Factor contributions (0-1 risk contribution) ─
function factorContribs(params) {
  // Each factor's weighted contribution to z (unsigned)
  const contribs = [
    { name: 'Sleep Quality',    val: Math.abs(W.sleepQuality * params.sleepQuality), direction: params.sleepQuality < 3 ? 'risk' : 'protect' },
    { name: 'Sleep Duration',   val: Math.abs(W.sleepHours   * params.sleepHours),   direction: params.sleepHours  < 7 ? 'risk' : 'protect' },
    { name: 'Screen Time',      val: Math.abs(W.screenTime   * params.screenTime),   direction: params.screenTime  > 4 ? 'risk' : 'protect' },
    { name: 'Social Jetlag',    val: Math.abs(W.jetlag       * params.jetlag),        direction: params.jetlag      > 1.5 ? 'risk' : 'protect' },
    { name: 'Screen Intensity', val: Math.abs(W.screenIndex  * params.screenIndex),  direction: params.screenIndex > 5 ? 'risk' : 'protect' },
    { name: 'Gender Factor',    val: Math.abs(W.gender       * params.gender),        direction: params.gender === 1 ? 'risk' : 'neutral' },
  ];
  const max = Math.max(...contribs.map(c => c.val));
  return contribs.map(c => ({ ...c, norm: max > 0 ? c.val / max : 0 }));
}

// ── Recommendations engine ────────────────────
function getRecs(params, risk) {
  const recs = [];

  if (params.screenTime > 4)
    recs.push({ icon: '📵', text: 'Cut daily screen time below 4 hrs. Set a hard daily limit with screen time controls.' });
  if (params.screenTime > 6)
    recs.push({ icon: '🚫', text: 'Heavy screen use (>6 hrs) significantly raises risk. Try a 7-day digital detox challenge.' });

  if (params.sleepHours < 8)
    recs.push({ icon: '🛏️', text: `You're getting ${params.sleepHours} hrs — aim for 8–9 hrs. Go to bed 30 min earlier each week until you reach it.` });

  if (params.sleepQuality < 3)
    recs.push({ icon: '🌙', text: 'Poor sleep quality detected. Avoid screens 1 hour before bed and keep your room cool and dark.' });

  if (params.jetlag > 1.5)
    recs.push({ icon: '⏰', text: `Social jetlag of ${params.jetlag} hrs disrupts your circadian rhythm. Try keeping wake-up time within 1 hr on weekends.` });

  if (params.screenIndex > 6)
    recs.push({ icon: '📵', text: 'High screen intensity (passive scrolling, social media) is more damaging than active use. Swap 1 hr of scrolling for a physical activity.' });

  if (risk > 0.5)
    recs.push({ icon: '🗣️', text: 'Your risk level is elevated. Consider speaking with a school counsellor or trusted adult about how you\'re feeling.' });

  if (params.sleepQuality >= 4 && params.sleepHours >= 8)
    recs.push({ icon: '✅', text: 'Great sleep habits! Maintain your consistent sleep schedule — this is your biggest protective factor.' });

  if (params.screenTime <= 2)
    recs.push({ icon: '🌿', text: 'Low screen time is excellent. Keep that boundary and maintain offline hobbies and social activities.' });

  if (recs.length === 0)
    recs.push({ icon: '✨', text: 'Your habits look balanced. Maintain your current screen and sleep patterns and check in monthly.' });

  return recs.slice(0, 4);
}

// ── BDI range estimate ────────────────────────
function getBDIRange(risk) {
  if (risk < 0.15) return { range: '0 – 8',  label: 'Minimal depressive symptoms',      color: '#22c55e' };
  if (risk < 0.30) return { range: '9 – 13', label: 'Borderline — monitor closely',      color: '#86efac' };
  if (risk < 0.50) return { range: '14 – 19', label: 'Mild depressive symptoms (BDI ≥ 14)', color: '#f59e0b' };
  if (risk < 0.70) return { range: '20 – 28', label: 'Moderate depressive symptoms',     color: '#f97316' };
  return                  { range: '29+',     label: 'Severe depressive symptoms',        color: '#ef4444' };
}

// ── Main predict function ─────────────────────
function runPredict() {
  const params = {
    screenTime:   parseFloat(document.getElementById('s-screen').value),
    screenIndex:  parseFloat(document.getElementById('s-index').value),
    sleepHours:   parseFloat(document.getElementById('s-sleep').value),
    sleepQuality: parseFloat(document.getElementById('s-quality').value),
    jetlag:       parseFloat(document.getElementById('s-jetlag').value),
    gender:       parseInt(document.querySelector('input[name="gender"]:checked').value),
  };

  const risk = computeRisk(params);
  const pct  = Math.round(risk * 100);

  // Category
  let cat, catClass;
  if      (pct < 25)  { cat = '🟢 Low Risk';      catClass = 'cat-low'; }
  else if (pct < 55)  { cat = '🟡 Moderate Risk';  catClass = 'cat-mod'; }
  else                { cat = '🔴 Elevated Risk';   catClass = 'cat-high'; }

  // Show result panel
  document.getElementById('idleState').classList.add('hidden');
  const out = document.getElementById('resultOutput');
  out.classList.remove('hidden');

  // Gauge
  document.getElementById('gaugePct').textContent = pct + '%';
  const catEl = document.getElementById('gaugeCat');
  catEl.textContent = cat;
  catEl.className = 'gauge-cat ' + catClass;

  // Animate gauge (track arc length ≈ 251)
  const fill   = document.getElementById('gaugeFill');
  const needle = document.getElementById('gaugeNeedle');
  const offset = 251 - (251 * risk);
  const angle  = -90 + (risk * 180); // -90 = far left, +90 = far right
  setTimeout(() => {
    fill.style.strokeDashoffset = offset;
    needle.style.transform = `rotate(${angle}deg)`;
  }, 50);

  // BDI range
  const bdi = getBDIRange(risk);
  document.getElementById('bdiEstVal').textContent  = bdi.range;
  document.getElementById('bdiEstVal').style.color  = bdi.color;
  document.getElementById('bdiEstSub').textContent  = bdi.label;

  // Factor breakdown
  const factors = factorContribs(params);
  const bList = document.getElementById('breakdown');
  bList.innerHTML = factors.map(f => {
    const color = f.direction === 'protect' ? '#22c55e' : f.direction === 'risk' ? '#ef4444' : '#94a3b8';
    const dir   = f.direction === 'protect' ? '↓ protective' : f.direction === 'risk' ? '↑ risk factor' : '— neutral';
    return `
      <div class="bf-row">
        <label>
          <span>${f.name}</span>
          <strong style="color:${color}">${dir}</strong>
        </label>
        <div class="bf-bar-bg">
          <div class="bf-bar" style="width:0%;background:${color}" data-w="${Math.round(f.norm*100)}"></div>
        </div>
      </div>`;
  }).join('');

  // Animate breakdown bars
  setTimeout(() => {
    document.querySelectorAll('.bf-bar').forEach(b => {
      b.style.width = b.dataset.w + '%';
    });
  }, 100);

  // Recommendations
  const recs = getRecs(params, risk);
  document.getElementById('recList').innerHTML = recs.map(r =>
    `<div class="rec-item"><span class="rec-icon">${r.icon}</span><span>${r.text}</span></div>`
  ).join('');

  // Scroll to result
  document.getElementById('resultPanel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Reset function ────────────────────────────
function resetForm() {
  document.getElementById('idleState').classList.remove('hidden');
  document.getElementById('resultOutput').classList.add('hidden');
  // Reset gauge
  const fill   = document.getElementById('gaugeFill');
  const needle = document.getElementById('gaugeNeedle');
  fill.style.strokeDashoffset = 251;
  needle.style.transform = 'rotate(-90deg)';
}

// ── Correlation bars (static data) ───────────
(function buildCorrBars() {
  const data = [
    { label: 'Sleep Quality Index', r: 0.38, color: '#7c5cfc' },
    { label: 'Avg Sleep Duration',  r: 0.30, color: '#4f8ef7' },
    { label: 'Social Jetlag',       r: 0.25, color: '#f59e0b' },
    { label: 'Weekend Mid-Sleep',   r: 0.20, color: '#f97316' },
    { label: 'Leisure Screen Time', r: 0.18, color: '#22c55e' },
    { label: 'Screen Intensity',    r: 0.14, color: '#14b8a6' },
  ];
  const MAX_PCT = 100; // r=0.38 = 100%
  const container = document.getElementById('corrBars');
  container.innerHTML = data.map(d => {
    const pct = Math.round((d.r / 0.38) * 100);
    return `
      <div class="corr-row">
        <div class="corr-label">${d.label}</div>
        <div class="corr-bar-bg">
          <div class="corr-bar" style="width:0%; background:${d.color}" data-w="${pct}"></div>
        </div>
        <div class="corr-val" style="color:${d.color}">r = ${d.r}</div>
      </div>`;
  }).join('');

  // Animate on scroll
  const corrSection = document.getElementById('insights');
  let animated = false;
  const obs = new IntersectionObserver(entries => {
    if (entries[0].isIntersecting && !animated) {
      animated = true;
      setTimeout(() => {
        document.querySelectorAll('.corr-bar').forEach(b => {
          b.style.transition = 'width 1s ease';
          b.style.width = b.dataset.w + '%';
        });
      }, 200);
      obs.disconnect();
    }
  }, { threshold: 0.3 });
  obs.observe(corrSection);
})();

// ── Active nav on scroll ──────────────────────
const sections  = ['predictor','insights','about'];
const navLinks  = document.querySelectorAll('.nav-a');
const sectionEls = sections.map(id => document.getElementById(id));

const navObs = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      navLinks.forEach(a => {
        a.classList.toggle('active', a.getAttribute('href') === '#' + entry.target.id);
      });
    }
  });
}, { threshold: 0.4 });
sectionEls.forEach(el => el && navObs.observe(el));
