export default function({parentElement, data}) {
  const root = parentElement.querySelector('.as-metrics');
  root.replaceChildren();
  const items = Array.isArray(data) ? data : (data?.items || []);
  root.classList.toggle('compact', Boolean(data?.compact));
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const frames = new Set();
  const pending = new Set();
  let disposed = false;
  const make = (tag, cls, text) => {
    const el = document.createElement(tag);
    el.className = cls;
    if (text !== undefined) el.textContent = text;
    return el;
  };
  const format = (n, item) => Number(n).toLocaleString('en-US', {
    minimumFractionDigits: item.decimals || 0, maximumFractionDigits: item.decimals || 0,
  }) + (item.suffix || '');
  const reveal = card => {
    if (!pending.has(card)) return;
    pending.delete(card);
    card.classList.add('is-visible');
    const { item, number } = card.metric;
    if (item.value == null || reduced.matches) return;
    const start = performance.now();
    const tick = now => {
      if (disposed || !card.isConnected) return;
      const p = Math.min((now - start) / 1250, 1);
      number.textContent = format(item.value * (1 - Math.pow(1 - p, 3)), item);
      if (p < 1 && !reduced.matches) schedule(tick);
      else number.textContent = format(item.value, item);
    };
    schedule(tick);
  };
  const schedule = fn => {
    const id = requestAnimationFrame(t => { frames.delete(id); fn(t); });
    frames.add(id);
  };
  const observer = 'IntersectionObserver' in window ? new IntersectionObserver(entries => {
    for (const e of entries) if (e.isIntersecting) {
      reveal(e.target); observer.unobserve(e.target);
    }
  }, {threshold: .14, rootMargin:'0px 0px -24px 0px'}) : null;
  for (const [i, item] of items.entries()) {
    const card = make('article','as-metric' + (item.featured ? ' featured' : ''));
    card.setAttribute('role','listitem');
    card.style.setProperty('--delay', ((i % 3) * 90) + 'ms');
    const top = make('div','as-metric-top');
    top.append(make('span','',item.label),make('span','as-metric-index',String(i + 1).padStart(2,'0')));
    const final = item.value == null ? 'Unavailable' : format(item.value, item);
    const number = make('span','as-metric-number',final);
    number.setAttribute('aria-hidden','true');
    card.append(top, number, make('span','sr-only',final),make('p','as-metric-note',item.note));
    if (item.suffix === '%' && item.value != null) {
      const track = make('div','as-metric-track');
      const fill = make('span','');
      fill.style.setProperty('--fill', Math.max(0,Math.min(100,item.value))+'%');
      track.append(fill); track.setAttribute('aria-hidden','true'); card.append(track);
    }
    root.append(card); card.metric = {item,number}; pending.add(card);
    if (observer && !reduced.matches) {card.classList.add('motion-ready');observer.observe(card);}
    else reveal(card);
  }
  const change = () => {
    if (reduced.matches) {
      pending.forEach(reveal);
      root.querySelectorAll('.as-metric').forEach(card => {
        card.metric.number.textContent = card.metric.item.value == null ? 'Unavailable' : format(card.metric.item.value,card.metric.item);
      });
    }
  };
  reduced.addEventListener('change',change);
  return () => {disposed=true;observer?.disconnect();frames.forEach(cancelAnimationFrame);reduced.removeEventListener('change',change);};
}
