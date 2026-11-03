"""Button state machine and album catalogue; independent of Linux/GTK."""
import json
import re
from pathlib import Path

SPEEDS = (.75, 1., 1.25, 1.5)
ORIENTATION_WARNING = '这一页可能放反了：请把乐谱上方朝向拍摄界面的箭头（右侧）重拍'


def playback_clock(score, started, speed, now):
    """Keep the shared SVG/timemap in base time; only the audio rate changes."""
    speed = float(speed) if speed in SPEEDS else 1.
    duration = max(0., float(score.get('duration', 0)))
    elapsed = max(0., now - started) if started else 0.
    base = min(duration, elapsed * speed)
    return dict(elapsed=base / speed, duration=duration / speed, base_ms=base * 1000,
                playing=bool(started and base < duration), speed=speed)


def current_measure(score, base_ms, page=None):
    """Timemap entries may revisit a measure when the score repeats."""
    selected = None
    for item in score.get('measures', []):
        if item.get('ms', item.get('start_ms', 0)) > base_ms:
            break
        selected = item
    if selected and (page is None or selected.get('page') == page):
        return selected
    return None


class Button:
    def __init__(self, long_ms=1500, debounce_ms=30):
        self.long_ms, self.debounce_ms = long_ms, debounce_ms
        self.raw = self.stable = False
        self.changed = self.pressed = 0
        self.long_sent = False
        self.initialized = False

    def update(self, down, now):
        if not self.initialized:
            self.initialized = True
            self.raw = self.stable = down
            self.changed = self.pressed = now
            # A button held across boot must first be released.
            self.long_sent = down
            return None
        if down != self.raw:
            self.raw, self.changed = down, now
        event = None
        if self.stable != self.raw and now - self.changed >= self.debounce_ms:
            self.stable = self.raw
            if self.stable:
                self.pressed, self.long_sent = now, False
            elif not self.long_sent:
                event = 'long' if now-self.pressed>=self.long_ms else 'short'
                self.long_sent = event=='long'
        if self.stable and not self.long_sent and now - self.pressed >= self.long_ms:
            self.long_sent = True
            event = 'long'
        return event


def catalogue(album, results):
    songs = {}
    for path in sorted((Path(album) / 'Meta').glob('??-??.json')):
        if not re.fullmatch(r'\d{2}-\d{2}\.json', path.name):
            continue
        try:
            meta = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        grey = Path(album) / 'Grey' / (path.stem + '.bmp')
        if meta.get('complete') is not True or not grey.is_file():
            continue
        song = songs.setdefault(path.stem[:2], dict(id=path.stem[:2], pages=[], xml=False))
        song['pages'].append(dict(id=path.stem, status='pending', progress=0, stage='等待识别'))
    # Finished scores are owned by the FAT card, never resurrected from ext4.
    for folder in (Path(album) / 'Outputs',):
        for path in folder.glob('??.musicxml'):
            if re.fullmatch(r'\d{2}\.musicxml', path.name):
                songs.setdefault(path.stem, dict(id=path.stem, pages=[]))['xml'] = True
    return [songs[key] for key in sorted(songs)]


class Panel:
    def __init__(self, songs):
        self.songs = songs
        self.view = 'recognize'
        self.selected = 0
        self.busy = False
        self.playing = False
        self.speed = 1.
        self.return_pending = False
        self.message = '请选择曲目，长按 S2 开始识别'

    def items(self):
        return [x for x in self.songs if x.get('xml')] if self.view == 'library' else [x for x in self.songs if x['pages']]

    def current(self):
        items = self.items()
        return items[self.selected % len(items)] if items else None

    def action(self, key, kind):
        if key == 'S3' and kind == 'short':
            self.return_pending = True
            self.message = '当前曲目处理完成后安全返回拍摄' if self.busy else '正在安全返回拍摄'
            return 'return' if not self.busy else 'defer_return'
        if self.busy:
            return None
        if key == 'S1' and kind == 'long':
            self.view = 'library' if self.view == 'recognize' else 'recognize'
            self.selected = 0
            self.playing = False
            self.message = '长按 S2 打开乐谱并播放' if self.view == 'library' else '长按 S2 识别选定曲目'
            return 'stop'
        if key == 'S1' and kind == 'short' and self.view == 'library':
            self.speed = SPEEDS[(SPEEDS.index(self.speed) + 1) % len(SPEEDS)]
            self.message = '播放速度 ' + format(self.speed, 'g') + '×；已打开的曲目从头播放'
            return 'speed'
        if key == 'S2' and kind == 'short':
            self.selected += 1
            self.playing = False
            return 'stop'
        if key == 'S2' and kind == 'long' and self.current():
            return 'play' if self.view == 'library' else 'recognize'
        return None
