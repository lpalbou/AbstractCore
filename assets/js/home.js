// abstractcore.ai home page: one sample plays at a time, so two voices can be compared in turn.
document.addEventListener('play', function (e) {
    if (!(e.target instanceof HTMLMediaElement)) return;
    document.querySelectorAll('audio, video').forEach(function (m) { if (m !== e.target) m.pause(); });
}, true);
