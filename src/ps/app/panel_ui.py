"""Unprivileged fullscreen GTK/Cairo DP interface; controller owns all card writes."""
import json
import os
import signal
import shutil
import socket
import subprocess
import sys
import time
import cairo
from pathlib import Path
from panel_model import ORIENTATION_WARNING, playback_clock, current_measure
from panel_client import StateWatch,request
from display_policy import keep_composited
import gi
gi.require_foreign('cairo')
gi.require_version('Gtk', '3.0')
gi.require_version('Rsvg', '2.0')
gi.require_version('PangoCairo', '1.0')
from gi.repository import Gtk, GLib, Gdk, Pango, PangoCairo, Rsvg

BG, CARD, ACTIVE, INK, MUTED, ACCENT = '#f5f5f7', '#ffffff', '#e8f2ff', '#1d1d1f', '#6e6e73', '#007aff'
LINE, TRACK, KEYCAP, BLUE_INK, WARNING = '#e5e5ea', '#e9e9ee', '#ededf0', '#005fcb', '#a64600'


def color(cr, value):
    cr.set_source_rgb(*(int(value[n:n+2], 16) / 255 for n in (1, 3, 5)))


def rounded_path(cr, x, y, w, h, radius):
    cr.new_sub_path()
    import math
    radius = min(radius, w / 2, h / 2)
    for cx, cy, first in ((x+w-radius, y+radius, -90), (x+w-radius, y+h-radius, 0),
                           (x+radius, y+h-radius, 90), (x+radius, y+radius, 180)):
        cr.arc(cx, cy, radius, math.radians(first), math.radians(first+90))
    cr.close_path()


def box(cr, x, y, w, h, fill, radius=16):
    if w <= 0 or h <= 0:
        return
    rounded_path(cr, x, y, w, h, radius)
    color(cr, fill); cr.fill()


def card_shape(cr, x, y, w, h):
    for spread, dy, alpha in ((5, 4, .012), (2, 2, .018), (0, 1, .025)):
        rounded_path(cr, x-spread, y-spread+dy, w+2*spread, h+2*spread, 20+spread)
        cr.set_source_rgba(0, 0, 0, alpha); cr.fill()
    box(cr, x, y, w, h, CARD, 20)
    rounded_path(cr, x+.5, y+.5, w-1, h-1, 20)
    color(cr, LINE); cr.set_line_width(1); cr.stroke()


CARD_CACHE = {}

def card(cr, x, y, w, h):
    # Card shadows cover most of the screen. Cache their alpha compositing at
    # the actual output scale, retaining native-resolution edges and text.
    import math
    scale = abs(cr.get_matrix().xx)
    key = (w, h, scale)
    if key not in CARD_CACHE:
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,
            math.ceil((w+12)*scale), math.ceil((h+16)*scale))
        canvas = cairo.Context(surface); canvas.scale(scale, scale)
        card_shape(canvas, 6, 6, w, h)
        CARD_CACHE[key] = surface
    cr.save()
    cr.rectangle(x-6, y-6, w+12, h+16); cr.clip()
    cr.translate(x-6, y-6); cr.scale(1/scale, 1/scale)
    cr.set_source_surface(CARD_CACHE[key], 0, 0); cr.paint()
    cr.restore()


def rule(cr, x, y, width, fill=LINE):
    color(cr, fill); cr.set_line_width(1)
    cr.move_to(x, y+.5); cr.line_to(x+width, y+.5); cr.stroke()


def document_icon(cr, x, y, selected=False):
    box(cr, x, y, 28, 34, '#d6e8ff' if selected else '#f0f0f3', 6)
    for offset, width in ((10, 14), (16, 14), (22, 9)):
        rule(cr, x+7, y+offset, width, ACCENT if selected else '#9a9aa1')


def page_indicator(cr, x, y, status):
    import math
    cr.new_sub_path()
    color(cr, ACCENT if status in ('running', 'ok') else WARNING if status == 'failed' else '#b6b6bf')
    cr.arc(x, y, 8, 0, math.tau)
    if status == 'ok':
        cr.fill(); cr.set_source_rgb(1, 1, 1); cr.set_line_width(1.7)
        cr.move_to(x-3.5, y); cr.line_to(x-1, y+2.5); cr.line_to(x+4, y-3); cr.stroke()
    else:
        cr.set_line_width(1.5); cr.stroke()
        if status == 'running':
            color(cr, ACCENT); cr.arc(x, y, 3, 0, math.tau); cr.fill()


def text(cr, value, x, y, size=18, fill=INK, bold=False, width=None, centered=False, wrap=False):
    layout = PangoCairo.create_layout(cr)
    layout.set_font_description(Pango.FontDescription(('WenQuanYi Micro Hei ' + ('Bold ' if bold else '') + str(size))))
    layout.set_text(str(value), -1)
    if width:
        layout.set_width(int(width * Pango.SCALE))
        if wrap: layout.set_wrap(Pango.WrapMode.WORD_CHAR)
        else: layout.set_ellipsize(Pango.EllipsizeMode.END)
        if centered: layout.set_alignment(Pango.Alignment.CENTER)
    color(cr, fill); cr.move_to(x, y); PangoCairo.show_layout(cr, layout)


def score_sheet(cr, handle, score, page, base_ms, x, y, w, h):
    """Use the same viewBox fit as Rsvg, so the bar follows its engraved measure."""
    cr.save()
    cr.rectangle(x, y, w, h); cr.clip()
    cr.set_source_rgb(1, 1, 1); cr.paint()
    measure = current_measure(score, base_ms, page)
    geometry = score.get('page_geometry', {}).get(str(page))
    if measure and geometry and measure.get('bbox'):
        vx, vy, vw, vh = geometry
        bx, by, bw, bh = measure['bbox']
        if vw > 0 and vh > 0 and bw > 0 and bh > 0:
            scale = min(w/vw, h/vh)
            left = x + (w-vw*scale)/2 + (bx-vx)*scale
            top = y + (h-vh*scale)/2 + (by-vy)*scale
            rounded_path(cr, left-4, top-5, bw*scale+8, bh*scale+10, 5)
            cr.set_source_rgba(0, .478, 1, .115); cr.fill_preserve()
            cr.set_source_rgba(0, .478, 1, .35); cr.set_line_width(1); cr.stroke()
    rectangle = Rsvg.Rectangle(); rectangle.x=x; rectangle.y=y; rectangle.width=w; rectangle.height=h
    handle.render_document(cr, rectangle)
    cr.restore()


def playback_screen(cr, state, score, svg, page, clock):
    """Playback owns the display; controls stay in a slim, quiet toolbar."""
    text(cr, 'Score Studio', 25, 15, 12, MUTED, True)
    text(cr, '曲目 ' + str(state.get('play', {}).get('song') or state.get('selected') or ''), 25, 37, 18, INK, True)
    measure = current_measure(score, clock['base_ms'], page)
    label = '第 ' + str(measure.get('number', '')) + ' 小节' if measure and measure.get('number') else '跟随乐谱播放'
    text(cr, label, 830, 28, 12, MUTED, width=190)
    box(cr, 1042, 22, 86, 30, ACTIVE, 15)
    text(cr, format(clock['speed'], 'g')+'×', 1063, 27, 12, BLUE_INK, True)
    text(cr, f'{page} / {len(score.get("pages",[]))} 页', 1150, 28, 12, MUTED)
    card(cr, 15, 75, 1250, 570)
    handle = svg.get(page)
    if handle:
        score_sheet(cr, handle, score, page, clock['base_ms'], 27, 84, 1226, 552)
    ratio = clock['elapsed']/max(.001, clock['duration'])
    box(cr, 27, 662, 1226, 4, TRACK, 2)
    if ratio: box(cr, 27, 662, max(4, 1226*min(1, ratio)), 4, ACCENT, 2)
    text(cr, 'S1 短按 调速 · 长按 返回识别     S2 短按 选曲 · 长按 重播     S3 返回拍摄', 27, 681, 11, MUTED)
    elapsed, duration = int(clock['elapsed']), int(clock['duration'])
    text(cr, f'{elapsed//60:02d}:{elapsed%60:02d} / {duration//60:02d}:{duration%60:02d}', 1084, 681, 11, MUTED)


class Window(Gtk.Window):
    def __init__(self, demo=None):
        super().__init__(title='AC920 Score Studio')
        self.state = dict(view='recognize', songs=[], selected=None, message='正在连接采集卡…', busy=False, play={})
        self.demo = demo
        self.token = None; self.player = None; self.audio_log = None; self.started = 0; self.score = None; self.page = 1
        self.svg = {}; self.audio_error = ''; self.key_down = {}; self.speed = 1.
        self.rendered_state = None
        self.drawn_input_sequence = None
        self.was_animating = False
        self.closed=False;self.watcher=None;self.scene_surface=None
        self.loaded_directory=None;self.pending_frame=None
        self.timing_log=Path('/run/user')/str(os.getuid())/'ac920-ui-latency.jsonl'
        self.set_default_size(1280, 720); self.fullscreen()
        self.connect('destroy', self.close)
        self.connect('map',lambda *_:keep_composited(self))
        self.connect('key-press-event', self.key_press)
        self.connect('key-release-event', self.key_release)
        self.canvas = Gtk.DrawingArea(); self.canvas.connect('draw', self.draw); self.add(self.canvas)
        GLib.timeout_add(40, self.tick)
        self.show_all()
        self.get_frame_clock().connect('after-paint',self.after_paint)
        if not self.demo:
            self.watcher=StateWatch(lambda value:GLib.idle_add(self.receive_state,value))
            self.watcher.start()

    def close(self, *_):
        self.closed=True
        if self.watcher:self.watcher.close()
        self.stop(); Gtk.main_quit()

    def receive_state(self,value):
        if self.closed:return False
        self.state={k:v for k,v in value.items() if k!='revision'};self.tick()
        return False

    def stop(self):
        if self.player and self.player.poll() is None:
            os.killpg(self.player.pid, signal.SIGTERM)
            try: self.player.wait(timeout=.5)
            except subprocess.TimeoutExpired: os.killpg(self.player.pid, signal.SIGKILL); self.player.wait()
        if self.audio_log: self.audio_log.close(); self.audio_log = None
        self.player = None; self.started = 0

    def key_press(self, _, event):
        name = Gdk.keyval_name(event.keyval)
        if name not in self.key_down: self.key_down[name] = time.monotonic()
        return True

    def key_release(self, _, event):
        name = Gdk.keyval_name(event.keyval)
        mapping = {'1': 'S1', '2': 'S2', '3': 'S3'}
        if name in mapping and name in self.key_down:
            duration = time.monotonic() - self.key_down.pop(name)
            # Physical F13-F15 are handled once by the backend, never duplicated here.
            try: self.state = request(dict(action='key', key=mapping[name], kind='long' if duration >= 1.5 else 'short'))
            except OSError: pass
        return True

    def tick(self):
        try:
            if self.demo:self.state = json.loads(Path(self.demo).read_text())
        except (OSError, ValueError):
            self.canvas.queue_draw(); return True
        playback = self.state.get('play', {})
        if playback.get('token') != self.token:
            self.stop(); self.token = playback.get('token'); self.audio_error = ''
            if playback.get('state') in ('preparing','ready'):
                directory = Path(playback['directory'])
                try:
                    self.score = json.loads((directory / 'score.json').read_text())
                    if self.loaded_directory!=directory:
                        self.svg = {1:Rsvg.Handle.new_from_file(str(directory/self.score['pages'][0]))}
                        self.loaded_directory=directory
                    self.page = 1
                    self.speed = playback.get('speed', self.state.get('speed', 1.))
                    rate = self.score.get('rates', {}).get(format(self.speed, 'g'), {})
                    if playback.get('state')=='preparing':
                        self.canvas.queue_draw()
                        return True
                    if not rate:
                        raise RuntimeError('所选速度的音频未准备完成，请重新打开乐谱')
                    self.audio_log = (directory / 'playback.log').open('w')
                    # ALSA card chosen explicitly; never fall back to an unrelated USB card.
                    cards = Path('/proc/asound/cards').read_text()
                    import re
                    found = re.search(r'^\s*(\d+)\s+\[.*(?:xilinx|DP|dp)', cards, re.M | re.I)
                    if not found: raise RuntimeError('DP 声卡未就绪')
                    command=['aplay', '-q', '-D', 'plughw:' + found[1] + ',0', str(directory / rate.get('wav', 'score.wav'))]
                    if shutil.which('pasuspender'): command=['pasuspender', '--', *command]
                    self.player = subprocess.Popen(command, stderr=self.audio_log, stdout=subprocess.DEVNULL,
                                                   start_new_session=True)
                    self.started = time.monotonic()
                except Exception as error:
                    self.audio_error = str(error)
                    if self.audio_log:
                        self.audio_log.close(); self.audio_log = None
        if self.player and self.player.poll() is not None and self.player.returncode:
            self.audio_error = 'DP 音频播放失败，详见 playback.log'; self.started = 0
        if self.score and self.started:
            ms = playback_clock(self.score, self.started, self.speed, time.monotonic())['base_ms']
            for turn in self.score['turns']:
                if ms >= turn['ms']: self.page = turn['page']
            if self.page not in self.svg:
                self.svg[self.page]=Rsvg.Handle.new_from_file(str(self.loaded_directory/self.score['pages'][self.page-1]))
        animating = bool(self.score and playback_clock(self.score, self.started, self.speed,
                                                       time.monotonic())['playing'])
        if self.state != self.rendered_state or animating or self.was_animating != animating:
            self.canvas.queue_draw()
        self.was_animating = animating
        return True

    def draw(self, widget, cr):
        self.draw_started = time.monotonic()
        self.rendered_state = self.state
        # ARMSOC XRender is slow for many small Cairo/Pango operations. Render
        # locally and upload one completed image instead of many X primitives.
        size=(widget.get_allocated_width(),widget.get_allocated_height())
        surface=cairo.ImageSurface(cairo.FORMAT_ARGB32,*size)
        self.draw_content(widget, cairo.Context(surface))
        cr.set_source_surface(surface, 0, 0)
        cr.paint()
        self.draw_latency()

    def draw_content(self, widget, cr):
        width, height = widget.get_allocated_width(), widget.get_allocated_height()
        cr.scale(width / 1280, height / 720)
        color(cr, BG); cr.paint()
        state = self.state; library = state.get('view') == 'library'
        speed = state.get('play', {}).get('speed', state.get('speed', 1.))
        clock = playback_clock(self.score or {}, self.started, speed, time.monotonic())
        if library and clock['playing'] and self.svg and state.get('play', {}).get('state') == 'ready':
            playback_screen(cr, state, self.score, self.svg, self.page, clock)
            return
        text(cr, 'Score Studio', 32, 18, 12, MUTED, True)
        text(cr, '乐谱库' if library else '曲目识别', 30, 43, 29, INK, True)
        text(cr, 'AC920  /  IMX477  /  DP', 951, 22, 12, MUTED)
        playing=bool(library and clock['playing'])
        box(cr, 1101, 59, 145, 29, ACTIVE if state.get('busy') or playing else '#eaeaee', 14)
        text(cr, '正在处理' if state.get('busy') else '正在播放' if playing else '等待操作', 1136, 63, 12,
             BLUE_INK if state.get('busy') or playing else MUTED)
        card(cr, 30, 112, 290, 496)
        text(cr, '已识别曲目' if library else '采集批次', 50, 132, 13, MUTED)
        songs = [s for s in state.get('songs', []) if s.get('xml')] if library else [s for s in state.get('songs', []) if s['pages']]
        box(cr, 268, 132, 31, 24, '#f1f1f4', 12)
        text(cr, len(songs), 278 if len(songs)<10 else 272, 135, 10, MUTED, True)
        selected = next((s for s in songs if s['id'] == state.get('selected')), None)
        index = songs.index(selected) if selected else 0
        first = max(0, min(index - 2, len(songs) - 5))
        for offset, song in enumerate(songs[first:first+5]):
            y = 174 + offset * 79
            if song == selected: box(cr, 44, y-1, 262, 70, ACTIVE, 11)
            document_icon(cr, 59, y+16, song == selected)
            text(cr, '曲目 ' + song['id'], 103, y+7, 19, BLUE_INK if song == selected else INK, True, width=190)
            done = sum(p['status'] == 'ok' for p in song['pages'])
            detail = (f"{len(song['pages'])} 页 · MusicXML" if song['pages'] else 'MusicXML 乐谱') if library else f"{len(song['pages'])} 页 · {done} 页已完成"
            text(cr, detail, 103, y+38, 11, MUTED, width=189)
        if not songs: text(cr, '暂无乐谱' if library else '暂无采集批次', 60, 220, 17, MUTED)
        rule(cr, 50, 563, 250)
        text(cr, f'{index+1} / {len(songs)} 首曲目' if songs else '拍摄后，曲目会显示在这里', 51, 577, 11, MUTED)
        card(cr, 340, 112, 910, 496)
        if library and self.score and self.svg and state.get('play', {}).get('state') in ('preparing','ready'):
            handle = self.svg.get(self.page)
            if handle:
                box(cr, 356, 127, 878, 431, '#ffffff', 12)
                rounded_path(cr, 356.5, 127.5, 877, 430, 12)
                color(cr, '#eeeeef'); cr.set_line_width(1); cr.stroke()
                score_sheet(cr, handle, self.score, self.page, clock['base_ms'], 366, 132, 858, 420)
            elapsed, duration = clock['elapsed'], clock['duration']
            ratio = max(0, min(1, elapsed / max(.001, duration)))
            text(cr, '音频准备中…' if state.get('play',{}).get('state')=='preparing' else '播放完成' if self.started and elapsed >= duration else '正在播放' if self.started else '乐谱就绪',
                 367, 570, 11, MUTED)
            box(cr, 465, 579, 590, 4, TRACK, 2)
            if ratio > 0: box(cr, 465, 579, max(4,590*ratio), 4, ACCENT, 2)
            text(cr, f"{int(elapsed)//60:02d}:{int(elapsed)%60:02d}  /  {int(duration)//60:02d}:{int(duration)%60:02d}", 1080, 568, 11, MUTED)
            text(cr, f"第 {self.page} / {len(self.score.get('pages',[]))} 页", 967, 65, 11, MUTED)
        elif library and selected:
            text(cr, '曲目 ' + selected['id'], 370, 139, 25, INK, True)
            text(cr, '已完成识别，随时可以聆听。', 371, 185, 13, MUTED)
            box(cr, 1089, 141, 126, 31, ACTIVE, 15)
            text(cr, '速度 '+format(state.get('speed', 1.), 'g')+'×', 1107, 146, 11, BLUE_INK, True)
            # A vector document mark stays crisp at any DP display resolution.
            box(cr, 727, 254, 136, 145, '#f6f8fc', 18)
            for offset in (39, 54, 69, 84): rule(cr, 753, 254+offset, 83, '#ced9e9')
            color(cr, ACCENT); cr.set_line_width(4)
            cr.move_to(804, 297); cr.line_to(804, 355); cr.stroke()
            cr.arc(794, 355, 10, 0, 6.283185307179586); cr.fill()
            text(cr, '正在准备乐谱…' if state.get('busy') else '把乐谱，听成音乐。', 430, 430, 23, INK, True, width=730, centered=True)
            text(cr, '长按 S2 全屏播放 · 短按 S1 调速', 430, 479, 14, MUTED, width=730, centered=True)
        elif selected:
            text(cr, '曲目 ' + selected['id'], 369, 137, 25, INK, True)
            pages = selected['pages']; done = sum(p['status'] == 'ok' for p in pages)
            total = max(0, min(100, sum(p['progress'] for p in pages) / max(1, len(pages))))
            text(cr, '逐页识别进度', 371, 184, 13, MUTED)
            box(cr, 1041, 143, 176, 32, '#f1f6ff', 16)
            text(cr, f'{done} / {len(pages)} 页完成', 1060, 148, 12, BLUE_INK, True, width=150)
            box(cr, 370, 222, 848, 7, TRACK, 3.5)
            if total: box(cr, 370, 222, max(7, 848 * total/100), 7, ACCENT, 3.5)
            running = next((i for i,p in enumerate(pages) if p['status'] == 'running'), max(0,done-1))
            suspect = next((i for i,p in enumerate(pages) if p.get('orientation_suspect')), None)
            if not state.get('busy') and suspect is not None: running = suspect
            count = 3 if any(p.get('orientation_suspect') for p in pages) else 4
            start = max(0, min(running-1, len(pages)-count))
            y = 253
            for page in pages[start:start+count]:
                page_indicator(cr, 379, y+16, page['status'])
                text(cr, '第 ' + str(int(page['id'][3:])+1) + ' 页', 399, y, 16, INK, True)
                text(cr, page['stage'], 530, y+2, 13, WARNING if page['status']=='failed' else MUTED, width=570)
                text(cr, str(page['progress'])+'%', 1155, y+2, 13, BLUE_INK)
                warned = bool(page.get('orientation_suspect'))
                if warned:
                    box(cr, 393, y+29, 831, 44, '#fff1e5', 6)
                    text(cr, ORIENTATION_WARNING, 401, y+32, 10, WARNING, width=814, wrap=True)
                bar = y+81 if warned else y+40
                box(cr, 399, bar, 819, 4, TRACK, 2)
                progress=max(0,min(100,page['progress']))
                if progress: box(cr, 399, bar, max(4,819*progress/100), 4, ACCENT, 2)
                y += 104 if warned else 77
        else:
            text(cr, '从纸面，到可演奏的乐谱。', 430, 281, 24, INK, True, width=730, centered=True)
            text(cr, 'S3 返回拍摄，采集一首新的作品。', 430, 334, 15, MUTED, width=730, centered=True)
        message = self.audio_error or state.get('button_error') or (
            '已请求返回拍摄，当前任务完成后执行。' if state.get('return_pending') else state.get('message', ''))
        warning=bool(self.audio_error or state.get('button_error'))
        if not songs and not warning and not state.get('return_pending'):
            message='长按 S1 返回识别界面，先完成一首曲目的识别。' if library else '短按 S3 返回拍摄，采集新的曲目。'
        if warning:
            box(cr, 30, 622, 1220, 29, '#fff1e5', 8)
            text(cr, message, 43, 625, 12, WARNING, width=1193)
        else:
            text(cr, message, 32, 625, 12, MUTED, width=1210)
        rule(cr, 30, 660, 1220)
        for x, key, label in ((30,'S1','短按调速 · 长按返回识别' if library else '长按浏览 XML'),
                               (440,'S2','短按选曲 · 长按播放' if library else '短按选曲 · 长按识别'),
                               (920,'S3','返回拍摄')):
            key_width=96 if key=='S1 长按' else 54
            box(cr, x, 678, key_width, 27, KEYCAP, 7)
            rounded_path(cr, x+.5, 678.5, key_width-1, 26, 7)
            color(cr, '#dadade'); cr.set_line_width(1); cr.stroke()
            text(cr, key, x+13, 681, 11, '#45454b', True)
            text(cr, label, x+key_width+17, 681, 12, INK)

    def draw_latency(self):
        event = self.state.get('input_event', {})
        sequence = event.get('sequence', 0)
        if self.drawn_input_sequence is None:
            self.drawn_input_sequence = sequence
            return  # A previous action can predate this GUI process.
        if sequence != self.drawn_input_sequence:
            self.drawn_input_sequence = sequence
            now = time.monotonic()
            print('[UI LATENCY] seq={} key={} {} action_to_draw_ms={:.1f} draw_ms={:.1f}'.format(
                sequence, event.get('key'), event.get('kind'),
                (now - event.get('monotonic', now)) * 1000,
                (now - self.draw_started) * 1000), flush=True)
            self.pending_frame=dict(event)
            self.log_timing(dict(phase='draw',monotonic=now,draw_ms=(now-self.draw_started)*1000,event=event))

    def log_timing(self,value):
        try:
            if self.timing_log.exists() and self.timing_log.stat().st_size>262144:self.timing_log.unlink()
            with self.timing_log.open('a') as log:log.write(json.dumps(value)+'\n')
        except OSError:pass

    def after_paint(self,clock):
        if not self.pending_frame:return
        event=self.pending_frame;self.pending_frame=None
        frame=clock.get_frame_counter()
        self.log_timing(dict(phase='after_paint',monotonic=time.monotonic(),frame=frame,event=event))
        GLib.timeout_add(40,self.presentation_timing,clock,frame,event,0)

    def presentation_timing(self,clock,frame,event,attempt):
        timings=clock.get_timings(frame)
        if timings and timings.get_complete():
            self.log_timing(dict(phase='presentation',monotonic=time.monotonic(),frame=frame,event=event,
                presentation_us=timings.get_presentation_time(),predicted_us=timings.get_predicted_presentation_time()))
        elif attempt<5:
            GLib.timeout_add(40,self.presentation_timing,clock,frame,event,attempt+1)
        return False


if __name__ == '__main__':
    Window(sys.argv[1] if len(sys.argv)>1 else None); Gtk.main()
