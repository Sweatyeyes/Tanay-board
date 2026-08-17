/* Кнопка «поддержать проект».
 *
 * Ссылка задаётся здесь одной строкой и подхватывается всеми страницами.
 * Пока строка пустая - на экране не появляется ничего, поэтому файл можно
 * загрузить в репозиторий заранее и включить кнопку позже.
 *
 * Куда вставлять: адрес страницы сбора (CloudTips, ЮMoney, Boosty - что
 * выберете). Ссылка открывается в новой вкладке, деньги через приложение
 * не проходят и никаких данных о плательщике оно не видит.
 */
var TANAY_SUPPORT_URL = "https://netmonet.co/tip/784656?o=6";

/* Подпись под кнопкой. Коротко и без давления: табло бесплатное,
   и просьба не должна выглядеть как плата за вход. */
var TANAY_SUPPORT_TEXT = "поддержать проект";
var TANAY_SUPPORT_NOTE = "табло и приложение бесплатные — на связь и развитие";

(function () {
  'use strict';
  if (!TANAY_SUPPORT_URL) return;

  function render() {
    var slot = document.getElementById('support');
    if (!slot) return;

    var css = document.createElement('style');
    css.textContent =
      '#support{text-align:center;padding:6px 16px 18px;line-height:1.5}' +
      '#support a{display:inline-block;color:rgba(255,255,255,.62);' +
      'font-size:12.5px;font-weight:800;letter-spacing:.05em;' +
      'text-transform:uppercase;text-decoration:none;padding:8px 16px;' +
      'border:1px solid rgba(255,255,255,.22);background:rgba(255,255,255,.05)}' +
      '#support a:active{background:rgba(255,255,255,.12)}' +
      '#support small{display:block;color:rgba(255,255,255,.32);' +
      'font-size:11px;margin-top:7px;letter-spacing:normal;text-transform:none}';
    document.head.appendChild(css);

    var a = document.createElement('a');
    a.href = TANAY_SUPPORT_URL;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.textContent = TANAY_SUPPORT_TEXT;
    slot.appendChild(a);

    if (TANAY_SUPPORT_NOTE) {
      var s = document.createElement('small');
      s.textContent = TANAY_SUPPORT_NOTE;
      slot.appendChild(s);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', render);
  } else {
    render();
  }
})();
