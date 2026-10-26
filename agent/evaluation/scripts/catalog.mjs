import {readFile, writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';

const root = new URL('../', import.meta.url);
const data = JSON.parse(await readFile(new URL('benchmarks.json', root), 'utf8'));
const ids = new Set();
const allowedDifficulty = new Set(['基础技能', '精细接触', '多步协作', '泛化与恢复']);
const allowedEnvironment = new Set(['仿真', '实机', '离线']);
const allowedGroups = new Set(['闭环控制', '综合代理', '迁移研究', '离线诊断']);
const csvCell = value => `"${String(value).replaceAll('"', '""')}"`;
const csv = rows => rows.map(row => row.map(csvCell).join(',')).join('\n') + '\n';
const ledger = [['source_id','bib_key','title','authors','year','family_or_topic','artifact_type','version_or_release_date','url','accessed_date','primary_source','claims_supported','evidence_location','disclosure_status','notes','published_date','published_kind','published_url','updated_date','updated_kind','updated_url']];
const claims = [['claim_id','section','claim','source_ids','evidence_location','status','notes']];
const references = [];

function checkURL(value) {
  assert.equal(new URL(value).protocol, 'https:', value);
}

function checkDate(value, mayBeEmpty) {
  if (mayBeEmpty && value === '') return;
  assert.match(value, /^\d{4}-\d{2}-\d{2}$/);
  assert.equal(new Date(value).toISOString().slice(0, 10), value);
  assert.ok(value <= data.updated, `日期晚于检索截止：${value}`);
}

checkDate(data.updated, false);
for (const b of data.benchmarks) {
  assert.match(b.id, /^[a-z0-9-]+$/);
  assert.ok(!ids.has(b.id), `重复条目：${b.id}`);
  ids.add(b.id);
  for (const key of ['name','setup','scale','observations','actions','metrics','protocol','caution','dateNote','origin']) {
    assert.equal(typeof b[key], 'string', `${b.id}.${key}`);
    assert.ok(b[key].trim().length > 0, `${b.id}.${key}`);
    assert.ok(!/TODO|TBD|主要|大概|本质上|通常|一般|差不多|大约/.test(b[key]), `${b.id}.${key}`);
  }
  assert.ok(allowedGroups.has(b.group));
  assert.ok(b.tasks.length > 0 && b.profiles.length > 0);
  assert.ok(b.difficulty.length > 0 && b.difficulty.every(value => allowedDifficulty.has(value)));
  assert.ok(b.environment.length > 0 && b.environment.every(value => allowedEnvironment.has(value)));
  for (const profile of b.profiles) for (const key of ['end','arms','base']) assert.ok(profile[key].length > 0);
  checkDate(b.published.date, false);
  checkDate(b.updated.date, true);
  assert.ok(b.updated.date === '' || b.updated.date >= b.published.date);
  assert.equal(b.year, Number(b.published.date.slice(0, 4)));
  for (const source of [b.source, b.published, b.updated]) checkURL(source.url);
  for (const key of ['title','authors','locator']) assert.ok(b.source[key].trim().length > 0);
  ledger.push([b.id,b.id,b.source.title,b.source.authors,b.year,b.name,'primary paper / official documentation',b.updated.kind,b.source.url,data.updated,'yes','结构、任务、观测、动作、指标与协议',b.source.locator,'reported',b.dateNote,b.published.date,b.published.kind,b.published.url,b.updated.date,b.updated.kind,b.updated.url]);
  claims.push([`${b.id}-protocol`,'catalog',`${b.name}：${b.scale}；${b.setup}；${b.metrics}`,b.id,b.source.locator,'reported',b.caution]);
  claims.push([`${b.id}-classification`,'catalog',`${b.name}：难度来源 ${b.difficulty.join('、')}；结构 ${b.profiles.map(p => [p.end,p.arms,p.base].join('/')).join('；')}`,b.id,b.source.locator,'inferred','结构为所引配置的归类；难度不是原作者统一量表。']);
  claims.push([`${b.id}-dates`,'dates',`${b.published.kind}：${b.published.date}；${b.updated.kind}：${b.updated.date || '未注明'}`,b.id,`${b.published.url} ; ${b.updated.url}`,b.updated.date === '' ? 'not-disclosed' : 'reported',b.dateNote]);
  const bibValue = value => value.replaceAll('\\', '').replaceAll('{', '').replaceAll('}', '').replaceAll('&', '\\&').replaceAll('%', '\\%').replaceAll('_', '\\_');
  references.push(`@misc{${b.id},\n  title = {${bibValue(b.source.title)}},\n  author = {{${bibValue(b.source.authors)}}},\n  year = {${b.year}},\n  url = {${b.source.url}},\n  note = {Accessed ${data.updated}; ${bibValue(b.source.locator)}}\n}`);
}
claims.push(['design-combinations','design','按目标技能、扰动条件和实机验证组合基准。','libero;libero-pro2;robotwin;robodojo;adroit;dexart;handson;bidex100;dexh2r;objectnav;homerobot;robocasa;behavior;locomujoco;humanoidbench;capbench;roboquest;robotworld','各条目协议与解释边界','inferred','编辑提出的实验设计建议，不是原论文规定的组合。']);
claims.push(['reporting-protocol','protocol','同时报告完成、进度、时间、恢复和物理质量；固定接口与信息权限。','calvin;behavior;robotworld;libero-recover;bench2dex','对应原文的 Evaluation / Metrics / Protocol 章节','inferred','页面追加报告规范；官方分数须另行保留。']);
await writeFile(new URL('research/source-ledger.csv', root), csv(ledger));
await writeFile(new URL('research/claim-evidence-matrix.csv', root), csv(claims));
await writeFile(new URL('research/references.bib', root), references.join('\n\n') + '\n');
console.log(`已验证 ${ids.size} 个条目，生成 ${ledger.length - 1} 条来源记录和 ${claims.length - 1} 条论断记录。`);
