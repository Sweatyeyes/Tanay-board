/* Журнал прыжков.
 *
 * Как считается прыжок. Табло показывает запись во взлёт, а не факт прыжка,
 * поэтому смотрим на исчезновение: пока взлёт с вашей фамилией висит на табло,
 * он лежит в "ожидании"; когда табло обновилось и этого взлёта на нём больше
 * нет - взлёт ушёл, значит прыжок был. Запись сразу идёт в зачёт, лишнюю
 * (самолёт вернулся с людьми, прыжок отменили) можно удалить руками.
 *
 * Всё хранится только в памяти телефона: репозиторий публичный, журналу там
 * не место. Копию можно выгрузить файлом на вкладке журнала.
 *
 * Модуль общий для table.html и log.html - логика одна, чтобы счёт не разошёлся.
 */
(function (global) {
  'use strict';

  var K_CFG = 'tanay_log_cfg';
  var K_JUMPS = 'tanay_log_jumps';
  var K_PEND = 'tanay_log_pending';

  function read(key, def) {
    try {
      var v = JSON.parse(localStorage.getItem(key) || 'null');
      return v == null ? def : v;
    } catch (e) { return def; }
  }
  function write(key, val) {
    try { localStorage.setItem(key, JSON.stringify(val)); } catch (e) {}
  }

  // ---------- настройки ----------
  // me      - "Фамилия" или "Фамилия Имя", по ним ищем себя во взлёте
  // canopy  - парашют по умолчанию ("JFX 84"), подставляется в новую запись
  // alt     - высота отделения по умолчанию, метры
  // base    - сколько прыжков было до того, как завели журнал
  // baseY   - год, к которому относится baseYear
  // baseYear- сколько из них выполнено в этом году
  function cfg() {
    var c = read(K_CFG, {});
    return {
      me: c.me || '',
      canopy: c.canopy || '',
      alt: c.alt == null ? '' : c.alt,
      base: +c.base || 0,
      baseY: +c.baseY || new Date().getFullYear(),
      baseYear: +c.baseYear || 0
    };
  }
  function setCfg(o) { write(K_CFG, o); }

  // ---------- записи ----------
  // {id, date:'2026-08-05', time:'11:22', load:'7', craft:'Л-410',
  //  canopy:'JFX 84', alt:4000, ex:'ФВ', task:'', auto:1}
  // load хранится, но на экране не показывается: он нужен только чтобы
  // не записать один и тот же взлёт дважды.
  function jumps() {
    var a = read(K_JUMPS, []);
    return Array.isArray(a) ? a : [];
  }
  function setJumps(a) { write(K_JUMPS, a || []); }

  function sortJumps(a) {
    return a.slice().sort(function (x, y) {
      if (x.date !== y.date) return x.date < y.date ? -1 : 1;
      var lx = parseInt(x.load, 10) || 0, ly = parseInt(y.load, 10) || 0;
      if (lx !== ly) return lx - ly;
      return (x.id || 0) - (y.id || 0);
    });
  }

  // Номера не храним: пересчитываем при показе. Иначе после удаления записи
  // из середины пришлось бы перенумеровывать весь журнал.
  function numbered() {
    var c = cfg(), list = sortJumps(jumps()), perYear = {}, out = [];
    for (var i = 0; i < list.length; i++) {
      var j = list[i], y = String(j.date || '').slice(0, 4);
      perYear[y] = (perYear[y] || 0) + 1;
      var offset = (+y === c.baseY) ? c.baseYear : 0;
      out.push({
        j: j,
        n: c.base + i + 1,               // общий номер
        ny: offset + perYear[y],         // номер в своём году
        year: y
      });
    }
    return out;
  }

  function totals() {
    var c = cfg(), list = jumps(), y = String(new Date().getFullYear());
    var thisYear = 0;
    for (var i = 0; i < list.length; i++) {
      if (String(list[i].date || '').slice(0, 4) === y) thisYear++;
    }
    return {
      total: c.base + list.length,
      year: (c.baseY === +y ? c.baseYear : 0) + thisYear,
      logged: list.length
    };
  }

  // ---------- сверка фамилий ----------
  function norm(s) {
    return String(s || '').toLowerCase().replace(/ё/g, 'е').replace(/[^a-zа-я]/g, '');
  }
  function dist(a, b) {
    var m = a.length, n = b.length, prev = [], cur = [], i, j;
    for (j = 0; j <= n; j++) prev[j] = j;
    for (i = 1; i <= m; i++) {
      cur[0] = i;
      for (j = 1; j <= n; j++) {
        cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1,
                          prev[j - 1] + (a.charAt(i - 1) === b.charAt(j - 1) ? 0 : 1));
      }
      prev = cur.slice();
    }
    return prev[n];
  }
  // допуск на ошибку распознавания - как в уведомлениях на табло
  function looksSame(a, b) {
    a = norm(a); b = norm(b);
    if (!a || !b) return false;
    if (a === b) return true;
    if (Math.abs(a.length - b.length) > 2) return false;
    return dist(a, b) <= (a.length <= 5 ? 1 : 2);
  }
  // "Мышкевич" сверяем только по фамилии, "Мышкевич Сергей" - и по имени
  function isMe(rowName, me) {
    var p = String(rowName || '').trim().split(/\s+/);
    var w = String(me || '').trim().split(/\s+/);
    if (!w[0]) return false;
    if (!looksSame(p[0], w[0])) return false;
    if (w.length > 1 && !looksSame(p[1] || '', w[1])) return false;
    return true;
  }
  function isService(r) {
    var n = String((r && r.name) || '');
    return n.indexOf('КВОРУМ') === 0 || n === '(Вып.)';
  }
  function loadNumber(title) {
    var m = String(title || '').match(/(\d+)\s*взл/i);
    return m ? m[1] : '';
  }

  // Дата берётся с самого табло ("01.08 11:22"): часовой пояс телефона
  // может отличаться, а день прыжка должен совпадать с днём на аэродроме.
  function boardDate(clock) {
    var m = String(clock || '').match(/(\d{1,2})\.(\d{1,2})/);
    var now = new Date();
    if (!m) return iso(now);
    var d = new Date(now.getFullYear(), +m[2] - 1, +m[1]);
    // 31 декабря на табло, а на телефоне уже январь - значит год прошлый
    if (now.getMonth() === 0 && +m[2] === 12) d.setFullYear(now.getFullYear() - 1);
    return iso(d);
  }
  function iso(d) {
    return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate());
  }
  function pad(n) { return (n < 10 ? '0' : '') + n; }
  function boardTime(clock) {
    var m = String(clock || '').match(/(\d{1,2}):(\d{2})/);
    return m ? pad(+m[1]) + ':' + m[2] : '';
  }

  // ---------- собственно наблюдение ----------
  // Возвращает число добавленных записей.
  function observe(data) {
    var c = cfg();
    if (!c.me || !data || !Array.isArray(data.loads)) return 0;
    var loads = data.loads;
    // Пустое табло (заставка, обрыв связи) - не повод считать взлёты ушедшими
    if (!loads.length) return 0;

    var date = boardDate(data.clock);
    var time = boardTime(data.clock);
    var pend = read(K_PEND, {});
    var onBoard = {}, added = 0, k;

    for (var i = 0; i < loads.length; i++) {
      var L = loads[i], num = loadNumber(L.title) || String(L.index || i + 1);
      onBoard[date + '|' + num] = 1;

      var rows = L.rows || [], mine = null;
      for (var r = 0; r < rows.length; r++) {
        if (!isService(rows[r]) && isMe(rows[r].name, c.me)) { mine = rows[r]; break; }
      }
      if (!mine) continue;
      k = date + '|' + num;
      pend[k] = { date: date, load: num, craft: L.aircraft || '',
                  ex: mine.cat || '', time: time || (pend[k] && pend[k].time) || '' };
    }

    // взлёт, который был в ожидании, пропал с табло - значит ушёл
    var list = jumps(), changed = false;
    for (k in pend) {
      if (onBoard[k]) continue;
      var p = pend[k];
      if (!hasJump(list, p.date, p.load)) {
        list.push({
          id: Date.now() + Math.floor(Math.random() * 1000),
          date: p.date, time: p.time || '', load: p.load,
          canopy: c.canopy || '', craft: p.craft || '',
          alt: c.alt === '' ? '' : +c.alt, ex: p.ex || '', task: '', auto: 1
        });
        added++; changed = true;
      }
      delete pend[k];
    }
    write(K_PEND, pend);
    if (changed) setJumps(list);
    return added;
  }

  function hasJump(list, date, load) {
    for (var i = 0; i < list.length; i++) {
      if (list[i].date === date && String(list[i].load) === String(load)) return true;
    }
    return false;
  }

  // сколько взлётов сейчас ждут отправления - для подписи на табло
  function pending() {
    var p = read(K_PEND, {}), n = 0;
    for (var k in p) n++;
    return n;
  }

  global.TanayLog = {
    cfg: cfg, setCfg: setCfg,
    jumps: jumps, setJumps: setJumps, sortJumps: sortJumps,
    numbered: numbered, totals: totals,
    observe: observe, pending: pending,
    hasJump: hasJump, iso: iso, boardDate: boardDate, boardTime: boardTime,
    looksSame: looksSame, isMe: isMe, loadNumber: loadNumber
  };
})(window);
