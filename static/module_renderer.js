// Minimal Module Renderer: renders moduleSpec into a target container with micro-animations
export function renderModule(moduleSpec, targetEl) {
  if (!moduleSpec || !targetEl) return;
  targetEl.innerHTML = '';
  const wrap = document.createElement('div');
  wrap.style.padding = '16px';
  wrap.style.background = '#fff';
  wrap.style.borderRadius = '16px';
  wrap.style.boxShadow = '0 10px 30px rgba(0,0,0,0.08)';

  const title = document.createElement('div');
  title.textContent = moduleSpec.title || 'Module';
  title.style.fontWeight = '600';
  title.style.marginBottom = '8px';
  wrap.appendChild(title);

  const instr = document.createElement('div');
  instr.textContent = moduleSpec.instructions || '';
  instr.style.opacity = '.8';
  instr.style.marginBottom = '10px';
  wrap.appendChild(instr);

  const area = document.createElement('div');
  area.style.minHeight = '120px';
  area.style.display = 'grid';
  area.style.placeItems = 'center';
  area.style.background = 'linear-gradient(120deg, #f9f5ff, #eef7ff)';
  area.style.borderRadius = '12px';
  area.style.position = 'relative';
  wrap.appendChild(area);

  // floating orbs
  for (let i = 0; i < 6; i++) {
    const o = document.createElement('div');
    o.style.position = 'absolute';
    o.style.width = '10px';
    o.style.height = '10px';
    o.style.borderRadius = '50%';
    o.style.left = `${Math.random() * 90 + 5}%`;
    o.style.top = `${Math.random() * 80 + 10}%`;
    o.style.background = 'radial-gradient(circle at 30% 30%, #caa9ff, #6a0dad)';
    o.style.animation = `float${i} 3s ease-in-out infinite`;
    const style = document.createElement('style');
    style.textContent = `@keyframes float${i} { 0%,100%{ transform: translateY(0) } 50%{ transform: translateY(-8px) } }`;
    document.head.appendChild(style);
    area.appendChild(o);
  }

  // render interactive elements
  (moduleSpec.interactive_elements || []).forEach(el => {
    if (el.type === 'button') {
      const b = document.createElement('button');
      b.textContent = el.label || 'Action';
      b.style.padding = '10px 14px';
      b.style.borderRadius = '10px';
      b.style.border = 'none';
      b.style.background = '#6a0dad';
      b.style.color = '#fff';
      b.style.cursor = 'pointer';
      b.addEventListener('click', () => {
        b.style.transform = 'scale(0.98)';
        setTimeout(() => (b.style.transform = 'scale(1)'), 120);
      });
      wrap.appendChild(b);
    }
    if (el.type === 'mood-slider') {
      const label = document.createElement('div');
      label.textContent = el.label || 'Mood';
      const r = document.createElement('input');
      r.type = 'range';
      r.min = '0';
      r.max = '10';
      r.value = '5';
      wrap.appendChild(label);
      wrap.appendChild(r);
    }
  });

  if (moduleSpec.reward && moduleSpec.reward.type === 'poem') {
    const box = document.createElement('div');
    box.style.marginTop = '12px';
    box.style.whiteSpace = 'pre-wrap';
    box.style.opacity = '.9';
    box.textContent = moduleSpec.reward.text || '';
    wrap.appendChild(box);
  }

  targetEl.appendChild(wrap);
}

