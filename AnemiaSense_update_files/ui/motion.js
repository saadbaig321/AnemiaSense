// Native Streamlit output shares the page scroll root. No iframe/parent access.
export default function() {
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const root = document.querySelector('[data-testid="stMainBlockContainer"]');
  if (!root || reduced.matches || !('IntersectionObserver' in window)) return;
  const seen = new WeakSet();
  const owned = new Set();
  const selector = '.as-section, .as-about, .as-workflow, .as-module-card, .metric-guide, .kpi-card, .info-card, .story-card, [data-testid="stMetric"], [data-testid="stPlotlyChart"], [data-testid="stVegaLiteChart"]';
  const observer = new IntersectionObserver(entries => {
    entries.forEach(e => {if(e.isIntersecting){e.target.classList.add('as-revealed');observer.unobserve(e.target);}});
  }, {threshold:.08, rootMargin:'0px 0px -16px 0px'});
  const scan = () => root.querySelectorAll(selector).forEach(el => {
    if(seen.has(el)) return;
    seen.add(el); owned.add(el); el.classList.add('as-reveal-ready'); observer.observe(el);
  });
  scan();
  const mutations = new MutationObserver(scan);
  mutations.observe(root,{childList:true,subtree:true});
  const revealAll = () => {if(reduced.matches) owned.forEach(el=>el.classList.add('as-revealed'));};
  reduced.addEventListener('change',revealAll);
  return () => {observer.disconnect();mutations.disconnect();reduced.removeEventListener('change',revealAll);owned.forEach(el=>el.classList.remove('as-reveal-ready','as-revealed'));};
}
