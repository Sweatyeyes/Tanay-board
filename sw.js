// Сервис-воркер онлайн-табло.
// Статика кэшируется, данные (board.json, картинка, last_update) всегда
// идут в сеть - воркер их не перехватывает, чтобы табло не показывало
// устаревшее. Без сети страница открывается из кэша и сама покажет
// "данные не обновлялись".
var CACHE = 'tanay-v2';
var ASSETS = [
  './',
  './index.html',
  './text.html',
  './logo.png',
  './manifest.webmanifest',
  './icon-192.png',
  './icon-512.png',
  './apple-touch-icon.png'
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
