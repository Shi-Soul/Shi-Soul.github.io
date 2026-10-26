'use strict';

const form = document.querySelector('#filters');
const body = document.querySelector('#catalog-body');
const selected = new Set();
const expanded = new Set();
const dimensions = ['end', 'arms', 'base', 'environment', 'group', 'tasks', 'difficulty'];
const presets = {
  single: {end: '夹爪', arms: '单臂'},
  dual: {end: '夹爪', arms: '双臂'},
  dex: {end: '灵巧手'},
  mobile: {base: '轮式移动'},
  humanoid: {base: '双足'},
  real: {environment: '实机'}
};
let benchmarks = [];
let visible = [];

function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, character => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[character]));
}

function link(url, label) {
  return `<a href="${escapeHTML(url)}" target="_blank" rel="noopener noreferrer">${escapeHTML(label)} ↗</a>`;
}

function dateText(date) {
  return date.date === '' ? '未注明' : date.date;
}

function dateHTML(date) {
  const value = date.date === '' ? '未注明' : `<time datetime="${escapeHTML(date.date)}">${escapeHTML(date.date)}</time>`;
  return `${value}<span class="date-kind">${escapeHTML(date.kind)}</span>`;
}

function profileText(benchmark) {
  return benchmark.profiles.map(profile => `${profile.arms} · ${profile.end} · ${profile.base}`).join('；');
}

function fields() {
  return Object.fromEntries(new FormData(form));
}

function matches(benchmark, filter) {
  const profileMatch = benchmark.profiles.some(profile => ['end', 'arms', 'base'].every(key => filter[key] === '' || profile[key] === filter[key]));
  const facetMatch = ['environment', 'tasks', 'difficulty'].every(key => filter[key] === '' || benchmark[key].includes(filter[key]));
  const groupMatch = filter.group === '' || benchmark.group === filter.group;
  const searchable = [benchmark.name, benchmark.setup, benchmark.scale, benchmark.observations, benchmark.actions, benchmark.metrics, benchmark.protocol, benchmark.caution, profileText(benchmark), ...benchmark.tasks, ...benchmark.difficulty].join(' ').toLocaleLowerCase();
  const words = filter.q.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
  return profileMatch && facetMatch && groupMatch && words.every(word => searchable.includes(word));
}

function sortRows(rows, mode) {
  return rows.toSorted((a, b) => {
    if (mode === 'name') return a.name.localeCompare(b.name, 'en');
    const dateOrder = b[mode].date.localeCompare(a[mode].date);
    return dateOrder || a.name.localeCompare(b.name, 'en');
  });
}

function details(benchmark) {
  const entries = [
    ['任务规模', benchmark.scale], ['平台与配置', benchmark.setup],
    ['观测输入', benchmark.observations], ['动作接口', benchmark.actions],
    ['原文指标', benchmark.metrics], ['比较条件', benchmark.protocol]
  ];
  return `<dl class="detail-grid">${entries.map(([label, value]) => `<div><dt>${label}</dt><dd>${escapeHTML(value)}</dd></div>`).join('')}
    <div class="wide"><dt>日期与版本</dt><dd>首次公开：${dateHTML(benchmark.published)}${link(benchmark.published.url, '首发依据')}<br>来源更新：${dateHTML(benchmark.updated)}${link(benchmark.updated.url, '更新依据')}<br>${escapeHTML(benchmark.dateNote)}</dd></div>
    </dl><p class="boundary">${escapeHTML(benchmark.caution)}</p>
    <p class="evidence-line">${link(benchmark.source.url, '技术来源')} · ${escapeHTML(benchmark.source.locator)}<br>发现线索：${escapeHTML(benchmark.origin)} · 本页核对：2026-10-26</p>`;
}

function rowHTML(benchmark) {
  const isExpanded = expanded.has(benchmark.id);
  return `<tr class="catalog-row" id="benchmark-${benchmark.id}">
    <td><input type="checkbox" data-select="${benchmark.id}" aria-label="比较 ${escapeHTML(benchmark.name)}" ${selected.has(benchmark.id) ? 'checked' : ''}></td>
    <td><a class="benchmark-name" href="#benchmark-${benchmark.id}" data-detail="${benchmark.id}" aria-expanded="${isExpanded}" aria-controls="detail-${benchmark.id}">${escapeHTML(benchmark.name)}</a>
      <div class="row-subtitle">${escapeHTML(benchmark.group)}</div>
      <dl class="dates"><dt>首发</dt><dd>${dateHTML(benchmark.published)}</dd><dt>更新</dt><dd>${dateHTML(benchmark.updated)}</dd></dl>
      <button type="button" class="details-button" data-detail="${benchmark.id}" aria-expanded="${isExpanded}" aria-controls="detail-${benchmark.id}">${isExpanded ? '收起' : '详情与协议'}</button>${link(benchmark.source.url, '来源')}</td>
    <td data-label="本体">${benchmark.profiles.map(profile => `<div class="config">${escapeHTML(profile.arms)} · ${escapeHTML(profile.end)}<span class="base">${escapeHTML(profile.base)}</span></div>`).join('')}</td>
    <td data-label="任务">${escapeHTML(benchmark.tasks.join(' · '))}<div class="tags">${benchmark.difficulty.map(value => `<span class="tag">${escapeHTML(value)}</span>`).join('')}</div></td>
    <td>${benchmark.environment.map(value => `<span class="env">${escapeHTML(value)}</span>`).join(' ')}</td>
    <td data-label="评测重点">${escapeHTML(benchmark.scale)}<p class="note">${escapeHTML(benchmark.metrics)}</p></td>
  </tr><tr class="detail-row" id="detail-${benchmark.id}" ${isExpanded ? '' : 'hidden'}><td colspan="6">${details(benchmark)}</td></tr>`;
}

function updateURL() {
  const url = new URL(window.location.href);
  url.search = '';
  for (const [key, value] of Object.entries(fields())) {
    if (value !== '' && !(key === 'sort' && value === 'published')) url.searchParams.set(key, value);
  }
  if (selected.size > 0) url.searchParams.set('compare', [...selected].join(','));
  history.replaceState(null, '', url);
}

function syncSelection() {
  body.querySelectorAll('[data-select]').forEach(input => {
    input.checked = selected.has(input.dataset.select);
    input.disabled = selected.size === 3 && !input.checked;
  });
  document.querySelector('#selected-count').textContent = `${selected.size} / 3`;
  const chosen = benchmarks.filter(benchmark => selected.has(benchmark.id));
  document.querySelector('#compare-message').textContent = chosen.length === 0 ? '尚未选择条目。' : `已选 ${chosen.length} 项。比较保留在这里，不受目录筛选影响。`;
  document.querySelector('#clear-comparison').disabled = chosen.length === 0;
  const columns = [
    ['首次公开', b => `${dateText(b.published)} · ${b.published.kind}`],
    ['来源更新', b => `${dateText(b.updated)} · ${b.updated.kind}`],
    ['日期说明', b => b.dateNote], ['本体', profileText],
    ['环境／形式', b => `${b.environment.join(' + ')} · ${b.group}`],
    ['任务与规模', b => `${b.tasks.join('、')}。${b.scale}`],
    ['难度来源', b => b.difficulty.join('、')], ['平台', b => b.setup],
    ['观测', b => b.observations], ['动作', b => b.actions],
    ['指标', b => b.metrics], ['比较条件', b => b.protocol], ['解释边界', b => b.caution]
  ];
  document.querySelector('#comparison-content').innerHTML = chosen.length === 0 ? '' : `<div class="table-wrap"><table class="comparison-table"><caption class="sr-only">所选基准的实验条件比较</caption><thead><tr><th scope="col">比较维度</th>${chosen.map(b => `<th scope="col">${escapeHTML(b.name)}<button type="button" class="compare-remove" data-remove="${b.id}" aria-label="从比较中移除 ${escapeHTML(b.name)}">移除</button></th>`).join('')}</tr></thead><tbody>${columns.map(([label, read]) => `<tr><th scope="row">${label}</th>${chosen.map(b => `<td>${escapeHTML(read(b))}</td>`).join('')}</tr>`).join('')}<tr><th scope="row">来源</th>${chosen.map(b => `<td>${link(b.source.url, '技术资料')}<br>${link(b.published.url, '首发记录')}<br>${link(b.updated.url, '更新记录')}</td>`).join('')}</tr></tbody></table></div>`;
}

function render() {
  const filter = fields();
  visible = sortRows(benchmarks.filter(benchmark => matches(benchmark, filter)), filter.sort);
  body.innerHTML = visible.map(rowHTML).join('');
  document.querySelector('#result-count').textContent = `显示 ${visible.length} / ${benchmarks.length} 项`;
  document.querySelector('#empty').hidden = visible.length !== 0;
  document.querySelector('.catalog-table').hidden = visible.length === 0;
  document.querySelector('#export').disabled = visible.length === 0;
  document.querySelectorAll('[data-preset]').forEach(button => {
    const preset = presets[button.dataset.preset];
    const active = filter.q === '' && dimensions.every(key => filter[key] === (Object.hasOwn(preset, key) ? preset[key] : ''));
    button.setAttribute('aria-pressed', String(active));
  });
  syncSelection();
  updateURL();
}

function resetFilters() {
  for (const name of ['q', ...dimensions]) form.elements.namedItem(name).value = '';
  form.elements.namedItem('sort').value = 'published';
}

function openBenchmark(id) {
  if (!benchmarks.some(benchmark => benchmark.id === id)) return;
  if (!visible.some(benchmark => benchmark.id === id)) resetFilters();
  expanded.add(id);
  render();
  const target = document.querySelector(`#benchmark-${id}`);
  target.scrollIntoView({block: 'start'});
  target.querySelector('.benchmark-name').focus({preventScroll: true});
}

function exportCSV() {
  const columns = [
    ['名称', b => b.name], ['首次公开', b => dateText(b.published)], ['首发类型', b => b.published.kind],
    ['来源更新', b => dateText(b.updated)], ['更新类型', b => b.updated.kind], ['日期说明', b => b.dateNote],
    ['本体', profileText], ['环境', b => b.environment.join(' + ')], ['形式', b => b.group],
    ['任务', b => b.tasks.join('；')], ['难度来源', b => b.difficulty.join('；')], ['规模', b => b.scale],
    ['平台', b => b.setup], ['观测', b => b.observations], ['动作', b => b.actions],
    ['指标', b => b.metrics], ['比较条件', b => b.protocol], ['解释边界', b => b.caution],
    ['技术来源', b => b.source.url], ['章节', b => b.source.locator],
    ['首发来源', b => b.published.url], ['更新来源', b => b.updated.url], ['本页核对', () => '2026-10-26']
  ];
  const quote = value => `"${String(value).replaceAll('"', '""')}"`;
  const csv = [columns.map(([label]) => quote(label)).join(','), ...visible.map(b => columns.map(([, read]) => quote(read(b))).join(','))].join('\r\n');
  const url = URL.createObjectURL(new Blob(['\uFEFF' + csv], {type: 'text/csv;charset=utf-8;'}));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'robot-evaluation-2026-10-26.csv';
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  document.querySelector('#theme').textContent = `主题：${{auto: '自动', light: '浅色', dark: '深色'}[theme]}`;
}

setTheme('auto');
document.querySelector('#theme').addEventListener('click', () => {
  const themes = ['auto', 'light', 'dark'];
  setTheme(themes[(themes.indexOf(document.documentElement.dataset.theme) + 1) % themes.length]);
});
form.addEventListener('submit', event => event.preventDefault());
form.addEventListener('input', render);
form.addEventListener('reset', event => { event.preventDefault(); resetFilters(); render(); });
document.querySelector('#clear-empty').addEventListener('click', () => { resetFilters(); render(); });
document.querySelector('#export').addEventListener('click', exportCSV);
document.querySelector('#clear-comparison').addEventListener('click', () => { selected.clear(); syncSelection(); updateURL(); });
body.addEventListener('change', event => {
  const input = event.target.closest('[data-select]');
  if (!input) return;
  if (input.checked && selected.size < 3) selected.add(input.dataset.select);
  else selected.delete(input.dataset.select);
  syncSelection();
  updateURL();
});
document.addEventListener('click', event => {
  const detail = event.target.closest('[data-detail]');
  const presetButton = event.target.closest('[data-preset]');
  const open = event.target.closest('[data-open]');
  const remove = event.target.closest('[data-remove]');
  if (detail) {
    event.preventDefault();
    const id = detail.dataset.detail;
    if (expanded.has(id)) expanded.delete(id); else expanded.add(id);
    document.querySelector(`#detail-${id}`).hidden = !expanded.has(id);
    body.querySelectorAll(`[data-detail="${id}"]`).forEach(control => {
      control.setAttribute('aria-expanded', String(expanded.has(id)));
      if (control.tagName === 'BUTTON') control.textContent = expanded.has(id) ? '收起' : '详情与协议';
    });
  } else if (presetButton) {
    resetFilters();
    for (const [key, value] of Object.entries(presets[presetButton.dataset.preset])) form.elements.namedItem(key).value = value;
    render();
  } else if (open) {
    event.preventDefault();
    history.replaceState(null, '', `#benchmark-${open.dataset.open}`);
    openBenchmark(open.dataset.open);
  } else if (remove) {
    selected.delete(remove.dataset.remove);
    syncSelection();
    updateURL();
  }
});

async function initialize() {
  const response = await fetch('benchmarks.json');
  if (!response.ok) throw new Error(`基准数据读取失败（HTTP ${response.status}）。`);
  const data = await response.json();
  benchmarks = data.benchmarks;
  document.querySelector('#total-count').textContent = benchmarks.length;
  for (const name of dimensions) {
    const values = benchmarks.flatMap(benchmark => ['end', 'arms', 'base'].includes(name) ? benchmark.profiles.map(profile => profile[name]) : benchmark[name]);
    const select = form.elements.namedItem(name);
    for (const value of new Set(values)) select.add(new Option(value, value));
  }
  const params = new URL(window.location.href).searchParams;
  for (const name of ['q', 'sort', ...dimensions]) {
    if (!params.has(name)) continue;
    const control = form.elements.namedItem(name);
    const value = params.get(name);
    if (name === 'q' || [...control.options].some(option => option.value === value)) control.value = value;
  }
  if (params.has('compare')) {
    for (const id of params.get('compare').split(',')) {
      if (selected.size < 3 && benchmarks.some(benchmark => benchmark.id === id)) selected.add(id);
    }
  }
  render();
  const id = window.location.hash.replace('#benchmark-', '');
  if (window.location.hash.startsWith('#benchmark-')) openBenchmark(id);
}

initialize().catch(error => {
  const notice = document.querySelector('#load-error');
  notice.hidden = false;
  notice.textContent = `${error.message} 请重新载入页面，或使用页末的完整数据下载链接。`;
  document.querySelector('#result-count').textContent = '基准目录未载入。';
  document.querySelector('.catalog-table').hidden = true;
  [...form.elements].forEach(control => { control.disabled = true; });
});
