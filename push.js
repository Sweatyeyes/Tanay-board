/* Подписка на пуш-уведомления.
 *
 * Зачем: пока приложение открыто, оно само следит за табло. Как только экран
 * гаснет, iOS замораживает страницу, и уведомления приходить перестают.
 * Пуш будит телефон снаружи - это единственный способ.
 *
 * Что уходит на сервер: адрес подписки, выданный Apple, галочки событий и
 * хэши отслеживаемых фамилий. Сами фамилии не передаются - по хэшу сверить
 * можно, прочитать нельзя.
 *
 * Работает только в установленном приложении: на айфоне пуши доступны
 * с iOS 16.4 и лишь для значка, добавленного на домашний экран.
 */
(function (global) {
  'use strict';

  var API = 'https://tanay-board.myshkevich.workers.dev';
  // Публичный ключ VAPID. Не секрет - он и должен быть виден в приложении.
  var VAPID_PUBLIC = 'BL9BlBCkLxcJDaKpNYzoMvs38fSTMCxtfPWqlIW1wEh414OoEeCqMMju_YoEbKsVkBqUQypEDpyOpSVTzZ8wEEY';

  function b64ToBytes(s) {
    s = (s + '='.repeat((4 - s.length % 4) % 4)).replace(/-/g, '+').replace(/_/g, '/');
    var bin = atob(s), out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  function norm(s) {
    return String(s || '').toLowerCase().replace(/ё/g, 'е').replace(/[^a-zа-я]/g, '');
  }
  // тот же хэш считает воркер для фамилий с табло
  function nameHash(surname) {
    var data = new TextEncoder().encode('tanay|' + norm(surname));
    return crypto.subtle.digest('SHA-256', data).then(function (buf) {
      var h = '';
      new Uint8Array(buf).forEach(function (b) { h += b.toString(16).padStart(2, '0'); });
      return h.slice(0, 16);
    });
  }

  function supported() {
    return 'serviceWorker' in navigator && 'PushManager' in window &&
           'Notification' in window;
  }
  function standalone() {
    return window.matchMedia('(display-mode: standalone)').matches ||
           window.navigator.standalone === true;
  }

  // Отправляет на сервер актуальный список фамилий и галочек.
  // Вызывается и при подписке, и когда список поменяли.
  function sync(watch, events) {
    if (!supported()) return Promise.resolve(false);
    return navigator.serviceWorker.ready.then(function (reg) {
      return reg.pushManager.getSubscription();
    }).then(function (sub) {
      if (!sub) return false;
      var names = (watch || []).map(function (w) {
        return nameHash(String(w).trim().split(/\s+/)[0]);
      });
      return Promise.all(names).then(function (hashes) {
        return fetch(API + '/push/sub', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            endpoint: sub.endpoint, names: hashes, events: events || {},
          }),
        }).then(function (r) { return r.ok; });
      });
    }).catch(function () { return false; });
  }

  function subscribe(watch, events) {
    if (!supported()) return Promise.reject(new Error('нет поддержки'));
    return navigator.serviceWorker.ready.then(function (reg) {
      return reg.pushManager.getSubscription().then(function (old) {
        if (old) return old;
        return reg.pushManager.subscribe({
          userVisibleOnly: true,                     // iOS иначе откажет
          applicationServerKey: b64ToBytes(VAPID_PUBLIC),
        });
      });
    }).then(function () { return sync(watch, events); });
  }

  function unsubscribe() {
    if (!supported()) return Promise.resolve(false);
    return navigator.serviceWorker.ready.then(function (reg) {
      return reg.pushManager.getSubscription();
    }).then(function (sub) {
      if (!sub) return false;
      var ep = sub.endpoint;
      return sub.unsubscribe().then(function () {
        return fetch(API + '/push/unsub', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ endpoint: ep }),
        }).then(function () { return true; });
      });
    }).catch(function () { return false; });
  }

  // Какая версия сервис-воркера реально работает на этом телефоне.
  // Старая копия без обработчика пуша - самая частая причина того, что
  // сервер отправил, служба доставки приняла, а уведомления нет.
  function swVersion() {
    if (!('serviceWorker' in navigator) || !navigator.serviceWorker.controller) {
      return Promise.resolve('не управляет страницей');
    }
    return new Promise(function (resolve) {
      var ch = new MessageChannel();
      var done = false;
      ch.port1.onmessage = function (e) {
        done = true;
        resolve((e.data && e.data.version) || 'без версии');
      };
      try {
        navigator.serviceWorker.controller.postMessage({ ask: 'version' }, [ch.port2]);
      } catch (err) { resolve('не отвечает'); return; }
      setTimeout(function () {
        if (!done) resolve('старая копия — не отвечает');
      }, 1500);
    });
  }

  function state() {
    if (!supported()) return Promise.resolve('нет поддержки');
    if (!standalone()) return Promise.resolve('только в установленном приложении');
    return navigator.serviceWorker.ready.then(function (reg) {
      return reg.pushManager.getSubscription();
    }).then(function (sub) {
      return sub ? 'подписан' : 'не подписан';
    }).catch(function () { return 'ошибка'; });
  }

  // Просит сервер прислать настоящий пуш прямо сейчас. Проверяет всю цепочку
  // разом: подпись ключом, доставку через Apple, пробуждение телефона.
  // Возвращает понятную человеку строку.
  function test() {
    if (!supported()) return Promise.resolve('этот браузер не умеет пуши');
    if (!standalone()) return Promise.resolve('работает только в приложении с домашнего экрана');
    return navigator.serviceWorker.ready.then(function (reg) {
      return reg.pushManager.getSubscription();
    }).then(function (sub) {
      if (!sub) return 'подписки нет — переключите уведомления';
      return fetch(API + '/push/selftest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ endpoint: sub.endpoint }),
      }).then(function (r) { return r.json(); }).then(function (d) {
        if (d && d.ok) return 'отправлен — уведомление должно прийти';
        return 'сервер не смог отправить: ' + ((d && (d.why || d.result)) || 'неизвестно');
      });
    }).catch(function (e) { return 'ошибка: ' + (e && e.message ? e.message : e); });
  }

  global.TanayPush = {
    supported: supported, standalone: standalone,
    subscribe: subscribe, unsubscribe: unsubscribe, sync: sync,
    state: state, test: test, swVersion: swVersion,
  };
})(window);
