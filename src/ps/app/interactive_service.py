"""Resident album controller. Recognition and reboot require explicit button actions."""
import argparse
import copy
import datetime
import json
import os
import pwd
import re
import signal
import socket
import socketserver
import struct
import subprocess
import threading
import time
from pathlib import Path
from buttons_linux import Keys
from onecard_common import atomic, mounted, read_multiboot, run, runtime_card, select_boot, sha, write_json
from panel_model import Button, Panel, catalogue
from fat_library import FatLibrary
from score_assets import RATES,load_score,rate_ready

STAGES = {'segment': (25, '谱线与符号分割'), 'relation': (55, '符号关联'),
          'headers': (65, '谱号与调号'), 'redecode': (80, '语义恢复'),
          'decode': (95, '节奏解码'), 'write': (98, '写入 MusicXML')}
HERE = Path(__file__).resolve().parent


def recognition_command(config, album, results, work, song):
    return ['runuser', '-u', config['user'], '--', 'env',
        'HF_HUB_OFFLINE=1', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=1',
        'PYTHONPYCACHEPREFIX=/home/fpga/claude-pycache', config['python'],
        str(HERE/'engine/omr_album.py'), '--album', str(album), '--out', str(results),
        '--work', str(work), '--rotate', config['rotate'], '--songs', str(int(song)), '--',
        config['python'], str(HERE/'engine/run_page_ui.py'), '--serve', '--pipeline',
        '--bundle', config['app'], '--threads', '4', '--csolve', '--fast-homr',
        '--fast-relation', '--fast-redecode', '--shared', config['recognizer']+'/shared',
        '--relation-models', config['recognizer']+'/shared/relation_models', '--link-margin', '0.5']


class Controller:
    def __init__(self, config, mount=Path('/mnt/ac920-onecard'), base=Path('/var/lib/ac920-onecard')):
        self.config, self.mount, self.base = config, Path(mount), Path(base)
        self.results = self.base / 'results'
        self.library = FatLibrary(self.mount,self.base)
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.revision = 0
        self.panel = Panel([])
        self.device = None
        self.attached = False
        self.play = dict(token=0, song=None, directory=None, state='stopped', error=None)
        self.button_error = ''
        self.log = None
        self.stopping = threading.Event()
        self.workers = set()
        self.processes = set()
        self.input_event = dict(sequence=0)
        self.prefetch_generation = 0
        self.prefetch_process = None

    def notify(self):
        with self.changed:
            self.revision += 1
            self.changed.notify_all()

    def wait_snapshot(self,after):
        with self.changed:
            self.changed.wait_for(lambda:self.revision!=after or self.stopping.is_set(),timeout=1)
            return self.snapshot()

    def cancel_prefetch(self):
        # Called with the state lock held. Never wait for a worker under it.
        self.prefetch_generation += 1
        process=self.prefetch_process
        if process and process.poll() is None:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass

    def worker(self, target, *args):
        def work():
            try:
                target(*args)
            finally:
                with self.lock:
                    self.workers.discard(threading.current_thread())
        thread = threading.Thread(target=work, daemon=True)
        self.workers.add(thread)
        thread.start()

    def check_running(self):
        if self.stopping.is_set():
            raise InterruptedError('控制器正在停止')

    def process(self, command, **kwargs):
        """Own the process group, so SIGTERM cannot leave a recognizer behind."""
        with self.lock:
            self.check_running()
            process = subprocess.Popen(command, start_new_session=True, **kwargs)
            self.processes.add(process)
        return process

    def forget(self, process):
        with self.lock:
            self.processes.discard(process)

    def start(self):
        self.device = runtime_card(self.config)
        if mounted(self.device):
            raise RuntimeError('boot FAT already mounted: ' + repr(mounted(self.device)))
        self.mount.mkdir(parents=True, exist_ok=True)
        run('mount', '-o', 'ro', self.device, self.mount)
        self.attached = True
        manifest = json.loads((self.mount / 'OneCard/manifest.json').read_text())
        for key in ('root_partuuid', 'boot_partuuid'):
            if manifest[key] != self.config[key]:
                raise RuntimeError('card manifest mismatch: ' + key)
        if sha(self.mount / 'BOOT.BIN') != self.config['capture_sha256']:
            raise RuntimeError('capture BOOT SHA mismatch')
        for folder in ('results', 'work', 'score-cache'):
            path = self.base / folder
            path.mkdir(parents=True, exist_ok=True)
            user = pwd.getpwnam(self.config['user'])
            os.chown(path, user.pw_uid, user.pw_gid)
        deleted=self.library.reconcile()
        if deleted:print('[LIBRARY] FAT deletion; removed local mirrors/cache: '+','.join(deleted),flush=True)
        self.panel = Panel(catalogue(self.mount, self.results))
        self.refresh_completed()
        print('[PANEL] READY; no automatic recognition or return; songs=' +
              ','.join(x['id'] for x in self.panel.songs), flush=True)

    def refresh_completed(self):
        try:
            done = json.loads((self.results / 'album.json').read_text()).get('pages', {})
        except (OSError, ValueError):
            done = {}
        for song in self.panel.songs:
            for page in song['pages']:
                record = done.get(page['id'], {})
                path = self.mount / 'Grey' / (page['id'] + '.bmp')
                if (record.get('status') == 'ok' and record.get('sha256', '').upper() == sha(path)
                        and record.get('rotate_used') == self.config.get('rotate', 'ccw')):
                    page.update(status='ok', progress=100, stage='已完成')
                    self.orientation(page, record)
            if song['pages']:
                song['xml'] = (self.mount/'Outputs'/(song['id']+'.musicxml')).is_file()

    @staticmethod
    def orientation(page, record):
        page.update(rotate_used=record.get('rotate_used'), clef_ratio=record.get('clef_ratio'),
                    orientation_suspect=bool(record.get('orientation_suspect', False)))

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(dict(view=self.panel.view, songs=self.panel.songs,
                        selected=self.panel.current()['id'] if self.panel.current() else None,
                        busy=self.panel.busy, message=self.panel.message, speed=self.panel.speed,
                        return_pending=self.panel.return_pending, play=self.play,
                        button_error=self.button_error, input_event=self.input_event,revision=self.revision))

    def command(self, key, kind, source_time=None):
        if key not in ('S1', 'S2', 'S3') or kind not in ('short', 'long'):
            raise ValueError('invalid panel action')
        with self.lock:
            if self.stopping.is_set():
                return
            action = self.panel.action(key, kind)
            self.input_event = dict(sequence=self.input_event['sequence'] + 1,
                                   key=key, kind=kind, monotonic=time.monotonic(),source_time=source_time)
            if action in ('stop','recognize','play','speed','return','defer_return'):self.cancel_prefetch()
            song = self.panel.current()
            print(f'[PANEL] {key} {kind}: {action}; selected={song["id"] if song else "-"}', flush=True)
            if action in ('stop', 'return', 'defer_return'):
                self.play.update(token=self.play['token'] + 1, state='stopped')
            if action == 'recognize':
                self.panel.busy = True
                self.panel.message = '加载识别引擎…'
                self.worker(self.recognize, song)
            elif action == 'play':
                self.panel.busy = True
                self.panel.message = '正在排版乐谱并准备音频…'
                self.worker(self.prepare_score, song)
            elif action == 'speed' and self.play['state'] == 'ready':
                self.panel.busy=True
                self.panel.message='正在准备 '+format(self.panel.speed,'g')+'× 钢琴音频…'
                self.worker(self.prepare_score,song)
            elif action == 'return':
                self.panel.busy = True
                self.worker(self.return_capture)
            self.notify()

    def recognize(self, song):
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        log = Path('/var/log/ac920-onecard') / (stamp + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            command = recognition_command(self.config,self.mount,self.results,self.base/'work',song['id'])
            with log.open('x') as output:
                process = self.process(command, cwd=self.config['recognizer'], stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
                for line in process.stdout:
                    output.write(line); output.flush()
                    self.progress(song, line)
                rc = process.wait()
                self.forget(process)
                output.flush(); os.fsync(output.fileno())
            state = json.loads((self.results / 'album.json').read_text())
            self.check_running()
            for page in song['pages']:
                record = state['pages'].get(page['id'], {})
                if record.get('status') != 'ok' or record.get('sha256', '').upper() != sha(self.mount / 'Grey' / (page['id'] + '.bmp')):
                    raise RuntimeError('页面未完成：' + page['id'])
                if record.get('rotate_used') != self.config['rotate']:
                    raise RuntimeError('页面方向与固定配置不一致：' + page['id'])
                with self.lock:
                    self.orientation(page, record)
            xml = self.results / (song['id'] + '.musicxml')
            if rc or not xml.is_file():
                raise RuntimeError('识别退出码 ' + str(rc) + '，或 MusicXML 未生成')
            # busy stays true, so a second worker cannot start. Do not hold the
            # UI-state lock across filesystem flushes; S3 can still queue return.
            run('mount', '-o', 'remount,rw', self.mount)
            try:
                atomic(self.mount / 'Outputs' / xml.name, xml.read_bytes())
                atomic(self.mount / 'Outputs/album.json', (self.results / 'album.json').read_bytes())
                atomic(self.mount / 'Outputs/last_run.log', log.read_bytes())
                write_json(self.mount / 'OneCard/LAST.JSON', dict(status='complete', song=song['id'], pages=len(song['pages'])))
                os.sync()
                self.library.record_current()
            finally:
                run('mount', '-o', 'remount,ro', self.mount)
            with self.lock:
                for page in song['pages']:
                    page.update(status='ok', progress=100, stage='已完成')
                song['xml'] = True
                suspect = [page['id'] for page in song['pages'] if page.get('orientation_suspect')]
                self.panel.message = (' / '.join(suspect) + '：这一页可能放反了，请按右侧箭头重拍' if suspect else
                    '曲目 ' + song['id'] + ' 已完成；长按 S1 浏览乐谱')
                self.notify()
            print('[PANEL] SONG ' + song['id'] + ' complete; staying in Linux', flush=True)
        except Exception as error:
            with self.lock:
                self.panel.message = '识别失败：' + str(error)
                for page in song['pages']:
                    if page['status'] != 'ok': page.update(status='failed', stage='未完成，可重试')
            print('[PANEL] Recognition error: ' + repr(error), flush=True)
        finally:
            with self.lock:
                requested = self.panel.return_pending
                self.panel.busy = bool(requested)
                self.notify()
            if requested and not self.stopping.is_set():
                self.return_capture()

    def progress(self, song, line):
        with self.lock:
            if line.startswith('[UI_STAGE]'):
                value = json.loads(line[len('[UI_STAGE]'):])
                key = Path(value['photo']).stem
                progress, label = STAGES.get(value['ui_stage'], (5, value['ui_stage']))
                for page in song['pages']:
                    if page['id'] == key and page['status'] != 'ok':
                        page.update(status='running', progress=max(page['progress'], progress), stage=label)
                self.panel.message = key + ' · ' + label
            match = re.search(r'(\d{2}-\d{2}): (.*)', line)
            if match:
                for page in song['pages']:
                    if page['id'] != match[1]: continue
                    if ' -> ' + song['id'] + '.musicxml' in match[2]:
                        page.update(status='ok', progress=100, stage='已完成')
                    elif match[2].startswith('FAILED') or 'marked failed' in match[2]:
                        page.update(status='failed', stage='识别失败')
                    elif match[2].startswith('sent') and page['status'] != 'ok':
                        page.update(status='running', stage='已送入识别流水线', progress=max(page['progress'], 2))
            self.notify()

    def render_command(self,xml,directory,speed):
        return ['runuser','-u',self.config['user'],'--','env',
            'AC920_PIANO_SF2='+self.config.get('piano_soundfont','/usr/share/sounds/sf2/FluidR3_GM.sf2'),
            self.config['python'],str(HERE/'render_score.py'),str(xml),str(directory),'--speed',format(speed,'g')]

    def publish_score(self,song,directory,state,speed):
        with self.lock:
            self.play=dict(token=self.play['token']+1,song=song['id'],directory=str(directory),
                state=state,error=None,speed=speed)
            self.panel.message=('曲目 '+song['id']+' · 正在准备钢琴音频…' if state=='preparing' else '曲目 '+song['id']+' · 乐谱播放')
            self.notify()

    def prefetch_rates(self,song,xml,directory,speed,generation):
        for rate in RATES:
            if rate==speed:continue
            with self.lock:
                if (generation!=self.prefetch_generation or self.stopping.is_set() or self.panel.busy
                    or self.panel.return_pending or self.play.get('song')!=song['id']):return
                if rate_ready(directory,load_score(directory,sha(xml)),rate):continue
                process=self.process(['nice','-n','15',*self.render_command(xml,directory,rate)],
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                self.prefetch_process=process
            try:rc=process.wait()
            finally:
                self.forget(process)
                with self.lock:
                    if self.prefetch_process is process:self.prefetch_process=None
            if rc:return
            print('[SCORE PREFETCH] '+song['id']+' '+format(rate,'g')+'x ready',flush=True)

    def prepare_score(self, song):
        success=False;started=time.monotonic()
        try:
            xml = self.mount / 'Outputs' / (song['id'] + '.musicxml')
            if not xml.is_file():raise RuntimeError('SD 卡中的曲谱已删除，请刷新曲库或重新识别')
            directory = self.base / 'score-cache' / (song['id'] + '_' + sha(xml)[:12] + '_r3')
            directory.mkdir(exist_ok=True)
            user = pwd.getpwnam(self.config['user'])
            os.chown(directory, user.pw_uid, user.pw_gid)
            speed=self.panel.speed
            cached=load_score(directory,sha(xml),self.config.get('piano_soundfont','/usr/share/sounds/sf2/FluidR3_GM.sf2'))
            if not rate_ready(directory,cached,speed):
                if cached:self.publish_score(song,directory,'preparing',speed)
                process=self.process(self.render_command(xml,directory,speed),stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
                try:
                    for line in process.stdout:
                        print(line.rstrip(),flush=True)
                        if line.startswith('[SCORE_LAYOUT]'):self.publish_score(song,directory,'preparing',speed)
                    if process.wait():raise RuntimeError('乐谱排版或钢琴音频生成失败')
                finally:self.forget(process)
            self.check_running()
            self.publish_score(song,directory,'ready',speed);success=True
            print('[SCORE OPEN] '+song['id']+' ready_ms='+str(round((time.monotonic()-started)*1000)),flush=True)
        except Exception as error:
            with self.lock:
                self.panel.message = '乐谱打开失败：' + str(error)
                self.play.update(token=self.play['token']+1,state='stopped')
            print('[PANEL] Score error: ' + repr(error), flush=True)
        finally:
            with self.lock:
                requested = self.panel.return_pending
                self.panel.busy = bool(requested)
                self.notify()
                if success and not requested and not self.stopping.is_set():
                    self.worker(self.prefetch_rates,song,xml,directory,speed,self.prefetch_generation)
            if requested and not self.stopping.is_set(): self.return_capture()

    def return_capture(self):
        if self.stopping.is_set(): return
        with self.lock:
            self.panel.busy = True
            self.play.update(token=self.play['token'] + 1, state='stopped')
            self.panel.message = '同步 SD 卡，正在返回拍摄…'
            prefetch=self.prefetch_process
            self.cancel_prefetch();self.notify()
        old_slot = None
        try:
            # A low-priority asset worker must release its FAT input before
            # unmount. Foreground recognition/export has already completed.
            if prefetch and prefetch.poll() is None:
                try:prefetch.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(prefetch.pid,signal.SIGKILL);prefetch.wait(timeout=5)
            os.sync()
            if self.attached:
                run('umount', self.mount); self.attached = False
            if mounted(self.device): raise RuntimeError('FAT still mounted')
            run('blockdev', '--flushbufs', self.device)
            self.check_running()
            old_slot = read_multiboot(); select_boot(0)
            print('[PANEL] S3 return: FAT unmounted, MultiBoot=0 verified', flush=True)
            subprocess.run(['systemctl', 'reboot'], check=True)
        except Exception as error:
            if old_slot is not None: select_boot(old_slot)
            with self.lock:
                self.panel.message = '返回失败：' + str(error)
                self.panel.busy = self.panel.return_pending = False
            print('[PANEL] Return error: ' + repr(error), flush=True)

    def close(self):
        self.stopping.set()
        self.notify()
        with self.lock:
            processes, workers = list(self.processes), list(self.workers)
        for process in processes:
            if process.poll() is None:
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
        for process in processes:
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait()
        for thread in workers:
            thread.join(timeout=20)
        if any(thread.is_alive() for thread in workers):
            raise RuntimeError('worker did not stop; FAT retained until systemd cleanup')
        if self.attached:
            os.sync(); run('umount', self.mount); self.attached = False
        print('[PANEL] STOPPED; workers joined; FAT unmounted', flush=True)


def serve(controller):
    directory = Path('/run/ac920-panel'); directory.mkdir(exist_ok=True)
    user = pwd.getpwnam(controller.config['user'])
    os.chown(directory, 0, user.pw_gid); os.chmod(directory, 0o750)
    path = directory / 'control.sock'
    path.unlink(missing_ok=True)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.request.settimeout(2)
            _, uid, _ = struct.unpack('3i', self.request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            if uid not in (0, user.pw_uid): return
            try:
                value = json.loads(self.rfile.readline(4096))
                if value.get('action') == 'key': controller.command(value['key'], value['kind'])
                response = controller.wait_snapshot(value.get('after',-1)) if value.get('action')=='wait' else controller.snapshot()
            except Exception as error:
                response = dict(error=str(error))
            try:
                self.wfile.write((json.dumps(response, ensure_ascii=False) + '\n').encode())
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass # A disconnected/slow UI must not flood the serial console.

    server = socketserver.ThreadingUnixStreamServer(str(path), Handler)
    server.daemon_threads = True
    os.chown(path, 0, user.pw_gid); os.chmod(path, 0o660)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/etc/ac920-onecard.json')
    args = parser.parse_args()
    controller = Controller(json.loads(Path(args.config).read_text()))
    server = None
    keys = None
    def stop(_signum, _frame):
        controller.stopping.set()
    previous = {number: signal.signal(number, stop) for number in (signal.SIGTERM, signal.SIGINT)}
    try:
        controller.start()
        server = serve(controller)
        # gpio-keys-polled has already debounced the electrical input in-kernel.
        buttons = {key: Button(debounce_ms=0) for key in ('S1', 'S2', 'S3')}
        try: keys = Keys()
        except Exception as error:
            controller.button_error = str(error)
            print('[PANEL] Physical keys unavailable: ' + str(error), flush=True)
        for button in buttons.values():button.update(False,time.monotonic()*1000)
        while not controller.stopping.is_set():
            if keys:
                levels=keys.poll()
                for key,down,stamp in keys.transitions:
                    kind=buttons[key].update(down,stamp*1000)
                    if kind:controller.command(key,kind,source_time=stamp if kind=='short' else buttons[key].pressed/1000+1.5)
                for key, down in levels.items():
                    kind = buttons[key].update(down, time.monotonic() * 1000)
                    if kind: controller.command(key, kind,source_time=buttons[key].pressed/1000+1.5)
            time.sleep(.005)
    finally:
        try:
            if server:
                server.shutdown()
                server.server_close()
            if keys:
                keys.close()
            controller.close()
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)


if __name__ == '__main__':
    main()
