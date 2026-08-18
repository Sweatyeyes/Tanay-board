// Сервис-воркер онлайн-табло.
// Статика кэшируется, данные (board.json, картинка, last_update) всегда
// идут в сеть - воркер их не перехватывает, чтобы табло не показывало
// устаревшее. Без сети страница открывается из кэша и сама покажет
// "данные не обновлялись".
var CACHE = 'tanay-v9';
var ASSETS = [
  './',
  './index.html',
  './text.html',
  './wl.html',
  './log.html',
  './journal.js',
  './push.js',
  './support.js',
  './logo.png',
  './manifest.webmanifest',
  './icon-192-v3.png',
  './icon-512-v3.png',
  './apple-touch-icon-v3.png'
];

self.addEventListener('install', function (e) {
  e.waitUntil(
    caches.open(CACHE).then(function (c) { return c.addAll(ASSETS); })
      .then(function () { return self.skipWaiting(); })
  );
});

self.addEventListener('activate', function (e) {
  e.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(keys.map(function (k) {
        if (k !== CACHE) return caches.delete(k);
      }));
    }).then(function () { return self.clients.claim(); })
  );
});

// Пуш приходит пустым: текст лежит на сервере и забирается по адресу
// /push/msg. Так не нужно шифровать содержимое пуша по RFC 8291.
// iOS требует показать уведомление на каждый принятый пуш, поэтому
// при любой осечке показываем хотя бы общую фразу - иначе Safari
// может отобрать подписку.
var API = 'https://tanay-board.myshkevich.workers.dev';

self.addEventListener('push', function (e) {
  e.waitUntil((async function () {
    var title = 'Табло Танай', body = 'Есть изменения по вашему взлёту';
    try {
      var sub = await self.registration.pushManager.getSubscription();
      if (sub) {
        var buf = await crypto.subtle.digest(
          'SHA-256', new TextEncoder().encode(sub.endpoint));
        var id = '';
        new Uint8Array(buf).forEach(function (b) {
          id += b.toString(16).padStart(2, '0');
        });
        var r = await fetch(API + '/push/msg?id=' + id.slice(0, 24),
                            { cache: 'no-store' });
        if (r.ok) {
          var m = await r.json();
          if (m && m.title) { title = m.title; body = m.body || ''; }
        }
      }
    } catch (err) { /* показываем общую фразу */ }
    return self.registration.showNotification(title, {
      body: body,
      icon: './icon-192-v3.png',
      badge: './icon-192-v3.png',
      tag: 'tanay-board',
    });
  })());
});

// тап по уведомлению открывает табло, а не новую вкладку
self.addEventListener('notificationclick', function (e) {
  e.notification.close();
  e.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true })
      .then(function (list) {
        for (var i = 0; i < list.length; i++) {
          if (list[i].url.indexOf('text.html') !== -1 && 'focus' in list[i]) {
            return list[i].focus();
          }
        }
        if (self.clients.openWindow) return self.clients.openWindow('./text.html');
      })
  );
});

self.addEventListener('fetch', function (e) {
  var req = e.request;
  if (req.method !== 'GET') return;
  var url = new URL(req.url);
  if (url.origin !== self.location.origin) return; // данные и шрифты - мимо кэша

  e.respondWith(
    fetch(req).then(function (resp) {
      if (resp && resp.ok) {
        var copy = resp.clone();
        caches.open(CACHE).then(function (c) { c.put(req, copy); });
      }
      return resp;
    }).catch(function () {
      return caches.match(req).then(function (hit) {
        return hit || caches.match('./text.html');
      });
    })
  );
});
