/*
 * Interaction test for the paged 点读 reader, run against the REAL inline
 * script extracted from server_intel.py.
 *
 * The browser surface was checked by hand; this locks down the behaviour that
 * is easy to get subtly wrong and impossible to eyeball:
 *   · 点读 is its own tab (合成 / 点读), and pagination only runs once the pane
 *     is displayed (a hidden stage measures as zero height);
 *   · turning a page slides HORIZONTALLY — two pages on the stage mid-turn,
 *     the outgoing one pushed off to the side, and nothing scrolls;
 *   · clicking sentence N seeks to starts[N];
 *   · with 自动连读 OFF, playback must STOP at the end of that sentence — the
 *     chapter is a single audio file, so without an explicit clock check it
 *     would read on to the end of the book (the bug this pins);
 *   · with it ON, playback advances sentence by sentence;
 *   · the last sentence plays out the file;
 *   · the sentence being read is underlined (class `cur`);
 *   · the progress bar follows the SENTENCE NUMBER, not the clock.
 *
 * Requires node (nothing else):   node test_reader_js.js
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const SRC = path.join(__dirname, 'server_intel.py');
const py = fs.readFileSync(SRC, 'utf-8');
const m = py.match(/WEB_HTML = """([\s\S]*?)\n"""\n/);
if (!m) { console.error('could not extract WEB_HTML'); process.exit(1); }
const html = m[1];
const script = html.split('<script>')[1].split('</script>')[0];

// 20 sentences so the layout really has to paginate.
const SENTS = Array.from({ length: 20 }, (_, i) => `第${i + 1}句测试文本。`);
const STARTS = SENTS.map((_, i) => Number((i * 1.5).toFixed(3)));
// The deep link below keeps the retired `&key=...` parameter on purpose:
// bookmarks and downloaded archives from older builds must keep working.
const TASK = 'uitest0001', KEY = 'uikey0123456789';
const SETTLE_MS = 450;                    // longer than the 400 ms slide fallback

let failures = 0;
function check(name, cond, detail = '') {
  console.log(`  ${cond ? 'OK  ' : '<<FAIL'} ${name}${detail ? ' — ' + detail : ''}`);
  if (!cond) failures++;
}
const settle = () => new Promise(r => setTimeout(r, SETTLE_MS));

// ── minimal DOM ──────────────────────────────────────────────────
const CHILD_HEIGHT = 40, PAGE_PAD = 20;   // stub metrics: ~7 sentences/page at 300px

function makeEl(id) {
  const el = {
    id, children: [], style: {}, dataset: {}, value: '', checked: true,
    disabled: false, textContent: '', className: '', files: [], clientHeight: 0,
    title: '', type: '', accept: '',
    scrollIntoView() { el._scrolled = true; },
    setAttribute(k, v) { el[k] = v; },
    removeAttribute(k) { delete el[k]; },
    append(...kids) { kids.forEach(k => { k._parent = el; el.children.push(k); }); },
    appendChild(k) { k._parent = el; el.children.push(k); return k; },
    removeChild(k) { el.children = el.children.filter(c => c !== k); return k; },
    remove() { if (el._parent) el._parent.removeChild(el); },
    addEventListener() {},          // no transitionend in the stub: the 400 ms
    removeEventListener() {},       // fallback in slideTo() is what settles it
    querySelector() { return makeEl('sub'); },
    cloneNode() { const c = makeEl(id); c.className = el.className; c.style = { ...el.style }; return c; },
    classList: {
      _parts() { return String(el.className).split(/\s+/).filter(Boolean); },
      _set(list) { el.className = list.join(' '); },
      contains(c) { return el.classList._parts().includes(c); },
      add(c) { const p = el.classList._parts(); if (!p.includes(c)) p.push(c); el.classList._set(p); },
      remove(c) { el.classList._set(el.classList._parts().filter(x => x !== c)); },
      toggle(c, on) { on ? el.classList.add(c) : el.classList.remove(c); },
    },
  };
  let _html = '';
  Object.defineProperty(el, 'innerHTML', {
    get() { return _html; },
    set(v) { _html = v; if (v === '') el.children.length = 0; },
  });
  // Emulated layout: content height grows with the number of children, so
  // paginate()'s overflow test has something real to measure.
  Object.defineProperty(el, 'scrollHeight', {
    get() { return PAGE_PAD + CHILD_HEIGHT * el.children.length; },
  });
  return el;
}

const byId = {};
const document = {
  activeElement: null,
  body: makeEl('body'),
  getElementById(id) { return (byId[id] = byId[id] || makeEl(id)); },
  createElement(tag) { return makeEl(tag); },
  addEventListener() {},
  fonts: { add() {} },
};

const audio = byId['audio'] = makeEl('audio');
audio.playCalls = 0; audio.pauseCalls = 0; audio.paused = true;
audio.play = function () { audio.playCalls++; audio.paused = false; return Promise.resolve(); };
audio.pause = function () { audio.pauseCalls++; audio.paused = true; };
audio.readyState = 0;

const store = {};
const splitDoc = {
  sents: SENTS, starts: STARTS, durations: SENTS.map(() => 1.5),
  audio_seconds: SENTS.length * 1.5, voice: 'default', speed: 1.0, total: SENTS.length,
};
// 服务端「现在」会返回的点读文档：测试逐段改写它（先是最终文档，后面是半成品）
let SERVER_DOC = splitDoc;
// 服务端会返回的任务进度，与 SERVER_DOC 联动（第 14 节驱动轮询链用）
let PROGRESS = { status: 'running', done: 0, total: SENTS.length, error: null };
let progressCalls = 0;

const sandbox = {
  console, document, audio, setTimeout, clearTimeout, URL, URLSearchParams,
  location: { search: `?task=${TASK}&key=${KEY}` },
  EventSource: function () { this.onmessage = null; this.onerror = null; },
  localStorage: {
    getItem: k => (k in store ? store[k] : null),
    setItem: (k, v) => { store[k] = String(v); },
    removeItem: k => { delete store[k]; },
  },
  // 状态条的兜底轮询在这套测试里不需要跑（状态条有自己的断言场景）
  setInterval: () => 0, clearInterval: () => {},
  FontFace: function (name) { this.family = name; this.load = async () => this; },
  FileReader: function () { this.readAsDataURL = () => { this.onload && this.onload(); }; },
  fetch: async (url) => {
    if (url.includes('/api/backend/status')) {
      return { ok: true, status: 200, json: async () => ({ backends: [], events: [] }),
               text: async () => '' };
    }
    // 进度接口：由测试脚本逐段改写 PROGRESS 来驱动轮询链（第 14 节）
    if (url.includes('/api/tts/progress/')) {
      progressCalls++;
      return { ok: true, status: 200,
               json: async () => ({ ...PROGRESS, pool: [] }), text: async () => '' };
    }
    return {
      ok: true, status: 200,
      json: async () => JSON.parse(JSON.stringify(SERVER_DOC)),
      text: async () => '', headers: { get: () => null }, blob: async () => ({}),
    };
  },
};
sandbox.globalThis = sandbox;
sandbox.window = sandbox;

const probe = `
;globalThis.__api = {
  playSentence, loadReader, stopReading, paginate, renderPage, fmtTime, gotoPage,
  seekPoints, sentenceEnd, spanSeconds, applyAppearance, sentenceCount, switchTab,
  refreshPartial, upgradeToFinal, waitForMore, readyCount, showDoc, audioUrl, pollTask,
  get partialMode() { return partialMode; },
  get curSent() { return curSent; },
  get pageIdx() { return pageIdx; },
  get pages() { return pages; },
  get splitDoc() { return splitDoc; },
  get currentTaskId() { return currentTaskId; },
};`;
vm.createContext(sandbox);
vm.runInContext(script + probe, sandbox);
const api = sandbox.__api;

const stage = document.getElementById('rdStage');
const pageEl = document.getElementById('rdPage');   // register it explicitly:
stage.clientHeight = 300;                 // the stub only knows ids it was asked for
stage.clientWidth = 1000;                 // page width, as a real viewport gives
document.documentElement = { clientWidth: 1000 };
// The markup already nests #rdPage inside #rdStage and gives it class="rd-page";
// the stub reads neither, so mirror both before anything inspects the stage.
pageEl.className = 'rd-page';
stage.appendChild(pageEl);
const autoNext = document.getElementById('autoNext');
const onStage = () => stage.children.filter(c => String(c.className).includes('rd-page'));
const sentSpans = () => [...pageEl.children].filter(c => String(c.className).includes('rd-sent'));
const spanFor = i => sentSpans().find(s => s.textContent.trim() === SENTS[i]);

(async () => {
  await new Promise(r => setTimeout(r, 40));       // let the deep-link load settle
  audio.readyState = 1;                            // metadata available (as in a browser)

  console.log('1. 点读 is its own tab and paginates there');
  check('reader tab is active', byId['paneRead'].classList.contains('active'));
  check('synthesis tab is not', !byId['paneSynth'].classList.contains('active'));
  check('task adopted from URL (a stale &key=... is ignored)',
        api.currentTaskId === TASK);
  check('sentences laid out into several pages', api.pages.length > 1,
        `${api.pages.length} pages for 20 sentences`);
  check('every sentence appears exactly once across the pages',
        api.pages.flat().length === 20 && new Set(api.pages.flat()).size === 20);
  check('one page on the stage at rest', onStage().length === 1, `${onStage().length}`);
  check('page position shown', byId['rdPos'].textContent === `1 / ${api.pages.length}`,
        byId['rdPos'].textContent);
  check('prev disabled on the first page', byId['rdPrev'].disabled === true);
  check('next enabled', byId['rdNext'].disabled === false);

  console.log('\n2. turning a page slides horizontally');
  byId['rdNext'].onclick();
  check('two pages on the stage mid-turn', onStage().length === 2, `${onStage().length}`);
  check('outgoing page is pushed off to the LEFT (not scrolled away)',
        pageEl.style.transform === 'translateX(-100%)', pageEl.style.transform);
  const ghost = onStage().find(c => c !== pageEl);
  check('incoming page settled at centre', !!ghost && ghost.style.transform === 'translateX(0)',
        ghost ? ghost.style.transform : 'no incoming page');
  await settle();
  check('one page left after the turn', onStage().length === 1, `${onStage().length}`);
  check('the resting page is back at centre', pageEl.style.transform === 'translateX(0)',
        pageEl.style.transform);
  check('page index advanced', api.pageIdx === 1, `${api.pageIdx}`);
  check('position label followed', byId['rdPos'].textContent === `2 / ${api.pages.length}`,
        byId['rdPos'].textContent);
  check('prev now enabled', byId['rdPrev'].disabled === false);

  console.log('\n3. turning back slides the other way');
  byId['rdPrev'].onclick();
  check('incoming page arrives from the left', onStage().length === 2);
  check('outgoing page leaves to the RIGHT', pageEl.style.transform === 'translateX(100%)',
        pageEl.style.transform);
  await settle();
  check('back on page 1',
        api.pageIdx === 0 && byId['rdPos'].textContent === `1 / ${api.pages.length}`,
        `page ${api.pageIdx + 1}`);

  console.log('\n4. the reader does not scroll');
  check('no vertical scroll offset is ever applied',
        !pageEl.scrollTop && !stage.scrollTop,
        `page ${pageEl.scrollTop}, stage ${stage.scrollTop}`);

  console.log('\n5. clicking a sentence seeks to starts[N]');
  for (const i of [0, 3, 6]) {
    api.gotoPage(api.pages.findIndex(pg => pg.includes(i)));   // instant jump
    const span = spanFor(i);
    check(`sentence ${i + 1} rendered on its page`, !!span);
    if (!span) continue;
    check(`sentence ${i + 1} span is clickable`, typeof span.onclick === 'function');
    audio.currentTime = -1; audio.playCalls = 0;
    span.onclick({ stopPropagation() {} });
    check(`sentence ${i + 1} → currentTime ${STARTS[i]}`,
          Math.abs(audio.currentTime - STARTS[i]) < 1e-9 && audio.playCalls === 1,
          `got ${audio.currentTime}, play()×${audio.playCalls}`);
  }

  console.log('\n6. the sentence being read is underlined (class `cur`)');
  api.gotoPage(0);
  api.playSentence(2);
  const cur = spanFor(2);
  check('playing sentence carries `cur`', cur && cur.classList.contains('cur'),
        cur ? cur.className : 'span not found');
  check('exactly one sentence is underlined',
        sentSpans().filter(s => s.classList.contains('cur')).length === 1);
  api.stopReading();
  check('no underline after stopping',
        sentSpans().filter(s => s.classList.contains('cur')).length === 0);

  console.log('\n7. 自动连读 OFF → playback stops at the END of the sentence');
  autoNext.checked = false;
  api.playSentence(2);
  check('started at sentence 3', api.curSent === 2 && audio.paused === false);
  audio.currentTime = api.sentenceEnd(2) - 0.5;     // still inside sentence 3
  audio.ontimeupdate();
  check('keeps playing inside the sentence', audio.paused === false && api.curSent === 2);
  audio.currentTime = api.sentenceEnd(2);           // reached the boundary
  audio.ontimeupdate();
  check('paused at the sentence boundary', audio.pauseCalls === 1 && audio.paused === true,
        `pause()×${audio.pauseCalls}`);
  check('reader stopped (not advanced into sentence 4)', api.curSent === -1,
        `curSent ${api.curSent}`);

  console.log('\n8. 自动连读 ON → advances, and the page follows');
  autoNext.checked = true;
  api.gotoPage(0);
  api.playSentence(2);
  audio.currentTime = api.sentenceEnd(2);
  audio.ontimeupdate();
  check('advanced to sentence 4', api.curSent === 3, `curSent ${api.curSent}`);
  check('playing sentence is the one it advanced to',
        api.curSent === 3 && Math.abs(audio.currentTime - STARTS[3]) < 1e-9,
        `curSent ${api.curSent}, currentTime ${audio.currentTime}`);
  check('underlined sentence followed', !!spanFor(3) && spanFor(3).classList.contains('cur'));
  check('page followed the reading', api.pages[api.pageIdx].includes(3),
        `page ${api.pageIdx + 1} holds [${api.pages[api.pageIdx]}]`);

  // 自动连读跨句边界时不能再定位一次：timeupdate 只有 250 ms 粒度，此时播放
  // 位置通常已经超前几十毫秒，再赋 currentTime 会触发 Range 重取 + 解码器重置
  // （听感＝每句开头卡一下），还会倒回去把几个字重复一遍。
  audio.currentTime = api.sentenceEnd(3) + 0.2;
  const beforeAdvance = audio.currentTime;
  audio.ontimeupdate();
  check('自动连读跨边界不重新定位（位置已对齐就续播）',
        api.curSent === 4 && Math.abs(audio.currentTime - beforeAdvance) < 1e-9,
        `curSent=${api.curSent}, ${beforeAdvance} -> ${audio.currentTime}`);
  // 真正点击远处的句子时仍然要跳过去
  audio.currentTime = 0;
  api.playSentence(8);
  check('点击别处仍然精确定位',
        Math.abs(audio.currentTime - STARTS[8]) < 1e-9, String(audio.currentTime));

  console.log('\n9. the last sentence plays out the file');
  const last = SENTS.length - 1;
  api.gotoPage(api.pages.length - 1);
  api.playSentence(last);
  check('no boundary for the last sentence', api.sentenceEnd(last) === null);
  audio.currentTime = api.splitDoc.audio_seconds - 0.1;
  audio.pauseCalls = 0;
  audio.ontimeupdate();
  check('not cut short', audio.pauseCalls === 0 && api.curSent === last);
  audio.onended();
  check('stops when the file ends', api.curSent === -1);

  console.log('\n10. the progress bar follows the sentence number');
  api.playSentence(0);
  check('sentence 1 of 20 → 5%', byId['rdProgFill'].style.width === '5.00%',
        byId['rdProgFill'].style.width);
  api.playSentence(9);
  check('sentence 10 of 20 → 50%', byId['rdProgFill'].style.width === '50.00%',
        byId['rdProgFill'].style.width);
  api.stopReading();
  api.gotoPage(0);
  check('idle: bar tracks the first sentence of the visible page (5%)',
        byId['rdProgFill'].style.width === '5.00%', byId['rdProgFill'].style.width);
  api.gotoPage(1);
  check('moving to the next page moves the bar with it',
        byId['rdProgFill'].style.width === '40.00%', byId['rdProgFill'].style.width);

  console.log('\n11. appearance applies to the stage (so every page inherits)');
  byId['rdFontSize'].oninput({ target: { value: '26' } });
  check('font size applied', stage.style.fontSize === '26px', stage.style.fontSize);
  check('font size stored', store['bsai.rd.size'] === '26', store['bsai.rd.size']);
  check('re-paginated after the size change', api.pages.length >= 1);
  const swatch = byId['rdColors'].children[1];
  swatch.onclick();
  check('colour applied', stage.style.color === swatch.dataset.color,
        `${stage.style.color} vs ${swatch.dataset.color}`);
  check('colour stored', store['bsai.rd.color'] === swatch.dataset.color);
  check('chosen swatch marked selected', swatch.classList.contains('sel'));

  console.log('\n12. pages still hold the whole book, in order');
  const flat = api.pages.flat();
  check('pages are contiguous and ordered', flat.every((v, i) => v === i),
        JSON.stringify(flat.slice(0, 25)));
  check('no sentence dropped after re-pagination', flat.length === 20, `${flat.length}`);

  console.log('\n13. 合成中提前点读（用已完成句子的前缀音频）');
  // 服务端给的前缀文档：4 句就绪 / 共 20 句，starts 全长，未就绪的尾部钉在前缀末尾
  const READY = 4;
  const partialStarts = SENTS.map((_, i) => Number((Math.min(i, READY) * 1.5).toFixed(3)));
  const partialDoc = (extra = {}) => Object.assign({
    sents: SENTS, starts: partialStarts, durations: SENTS.map(() => 1.5),
    audio_seconds: READY * 1.5, voice: 'default', speed: 1.0, total: SENTS.length,
    partial: true, ready: READY, done: READY,
  }, extra);
  const pageWith = i => api.pages.findIndex(pg => pg.includes(i));

  SERVER_DOC = partialDoc();
  PROGRESS = { status: 'running', done: READY, total: SENTS.length, error: null };
  await api.loadReader(TASK, KEY);
  check('partial document adopted',
        api.partialMode === true && api.readyCount() === READY,
        `partial=${api.partialMode}, ready=${api.readyCount()}`);
  check('the prefix audio is the current source',
        String(audio.src).includes('/api/tts/partial/')
        && String(audio.src).includes('v=4'), String(audio.src));

  api.gotoPage(pageWith(READY + 2));
  const pendingSpan = spanFor(READY + 2);
  check('a not-yet-synthesized sentence is struck through',
        !!pendingSpan && pendingSpan.classList.contains('pending'),
        pendingSpan ? pendingSpan.className : 'span not found');
  check('and it is not clickable',
        !!pendingSpan && typeof pendingSpan.onclick !== 'function');
  api.gotoPage(pageWith(1));
  const readySpan = spanFor(1);
  check('a finished sentence stays clickable',
        !!readySpan && !readySpan.classList.contains('pending')
        && typeof readySpan.onclick === 'function');
  audio.currentTime = -1;
  readySpan.onclick({ stopPropagation() {} });
  check('clicking it seeks to its start (prefix samples match the final audio)',
        Math.abs(audio.currentTime - partialStarts[1]) < 1e-9, String(audio.currentTime));

  // 读到「已就绪末尾」→ 挂起等合成，而不是一路读到文件末尾
  autoNext.checked = true;
  api.playSentence(READY - 1);
  audio.currentTime = api.sentenceEnd(READY - 1);
  audio.pauseCalls = 0;
  audio.ontimeupdate();
  check('at the ready edge it pauses and waits for the synthesizer',
        audio.paused === true && api.curSent === READY - 1,
        `paused=${audio.paused}, curSent=${api.curSent}`);

  // 合成追上来：ready 4 → 6，换更长的前缀并接着读
  SERVER_DOC = partialDoc({
    starts: SENTS.map((_, i) => Number((Math.min(i, READY + 2) * 1.5).toFixed(3))),
    audio_seconds: (READY + 2) * 1.5, ready: READY + 2, done: READY + 2,
  });
  PROGRESS = { status: 'running', done: READY + 2, total: SENTS.length, error: null };
  await api.refreshPartial();
  check('the longer prefix is adopted', api.readyCount() === READY + 2, String(api.readyCount()));
  if (typeof audio.onloadedmetadata === 'function') audio.onloadedmetadata();
  check('reading resumes at the first newly ready sentence',
        api.curSent === READY && audio.paused === false,
        `curSent=${api.curSent}, paused=${audio.paused}`);
  api.gotoPage(pageWith(READY + 1));
  const nowReady = spanFor(READY + 1);
  check('the strikethrough is lifted as sentences land',
        !!nowReady && !nowReady.classList.contains('pending'));

  // 合成完成：就地换成最终整章音频，位置与页码都不重置
  const beforePage = api.pageIdx;
  SERVER_DOC = partialDoc({ starts: STARTS, audio_seconds: SENTS.length * 1.5,
                            partial: false, ready: SENTS.length, done: SENTS.length });
  PROGRESS = { status: 'running', done: SENTS.length, total: SENTS.length, error: null };
  api.upgradeToFinal();
  if (typeof audio.onloadedmetadata === 'function') audio.onloadedmetadata();
  check('completion swaps to the final chapter audio in place',
        api.partialMode === false && String(audio.src).includes('/api/tts/result/'),
        String(audio.src));
  check('every sentence is clickable again', api.readyCount() === SENTS.length);
  check('the page you were on is kept', api.pageIdx === beforePage,
        `${api.pageIdx} vs ${beforePage}`);

  console.log('\n14. 轮询链一直跑到完成（曾经只跑一次就断）');
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  // 回到「合成中」：4/20 句就绪，轮询由 loadReader 自己起
  SERVER_DOC = partialDoc();
  PROGRESS = { status: 'running', done: READY, total: SENTS.length, error: null };
  const calls0 = progressCalls;
  await api.loadReader(TASK);
  check('打开点读就开始轮询', api.partialMode === true && api.readyCount() === READY,
        `partial=${api.partialMode}, ready=${api.readyCount()}`);
  await sleep(1900);
  check('轮询在继续（不是只跑一次就停）', progressCalls - calls0 >= 2,
        `${progressCalls - calls0} 次进度请求`);
  check('进度行跟着服务端走', byId['progressText'].textContent.includes(`${READY}/20`),
        byId['progressText'].textContent);

  // 合成推进到 8 句：进度与点读前缀都要自己跟上
  PROGRESS = { status: 'running', done: 8, total: SENTS.length, error: null };
  SERVER_DOC = partialDoc({
    starts: SENTS.map((_, i) => Number((Math.min(i, 8) * 1.5).toFixed(3))),
    audio_seconds: 12, ready: 8, done: 8,
  });
  await sleep(1900);
  check('进度数字自动更新', byId['progressText'].textContent.includes('8/20'),
        byId['progressText'].textContent);
  check('点读的就绪句数自动跟上', api.readyCount() === 8, String(api.readyCount()));

  // 合成完成：不刷新页面也要变 100% 并换成整章音频
  PROGRESS = { status: 'done', done: SENTS.length, total: SENTS.length, error: null };
  SERVER_DOC = splitDoc;
  await sleep(1900);
  check('完成后进度条 100%', byId['progressFill'].style.width === '100%',
        byId['progressFill'].style.width);
  check('完成后提示「合成完成」', byId['progressText'].textContent.includes('合成完成'),
        byId['progressText'].textContent);
  check('完成后点读自动升级为整章音频',
        api.partialMode === false && String(audio.src).includes('/api/tts/result/'),
        String(audio.src));
  check('合成按钮解锁', byId['longBtn'].disabled === false);
  const afterDone = progressCalls;
  await sleep(1600);
  check('完成后轮询自己收尾', progressCalls === afterDone,
        `多跑了 ${progressCalls - afterDone} 次`);

  console.log();
  console.log('RESULT:', failures ? `${failures} FAILURES` : 'ALL PASS');
  process.exit(failures ? 1 : 0);
})();
