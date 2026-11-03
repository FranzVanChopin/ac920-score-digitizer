"""Serial, resumable Image/SS-PP.jpg -> Outputs/SS-PP.musicxml on an SD card.

The parent uses only the standard library on Linux. Each real recognizer runs
in a fresh process group; desktop validation is explicitly labelled throughout.
No reference scores, camera, FPGA programming, or training are used here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

NAME = re.compile(r"^(\d{2})-(\d{2})\.(jpg|jpeg|png)$", re.IGNORECASE | re.ASCII)
VERSION = 1


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def inside(path, root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def atomic_json(path, payload):
    path = Path(path)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('x', encoding='utf-8', newline='\n') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        sync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def sync_directory(path):
    if sys.platform == 'linux':
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def remove_work(path, control):
    path, control = Path(path), Path(control).resolve()
    if path.is_symlink() or path.resolve() == control or not inside(path, control):
        raise ValueError('Refusing to remove a directory outside this batch workspace')
    shutil.rmtree(path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def validate_xml(path):
    # Parse only local generated files; disallow DTD/entity declarations.
    path = Path(path)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError('MusicXML exceeds the 32 MiB output bound')
    data = path.read_bytes()
    if b'<!DOCTYPE' in data.upper() or b'<!ENTITY' in data.upper():
        raise ValueError('Unexpected DTD/entity in generated MusicXML')
    root = ET.fromstring(data)
    parts = root.findall('part')
    measures = sum(len(p.findall('measure')) for p in parts)
    if root.tag != 'score-partwise' or not parts or not measures:
        raise ValueError('Recognizer did not produce a nonempty MusicXML score')
    notes = sum(len(m.findall('note')) for p in parts for m in p.findall('measure'))
    if not notes:
        raise ValueError('Recognizer produced no notes or rests')
    return {'parts': len(parts), 'measures': measures, 'notes_and_rests': notes}


def scan_images(folder):
    valid, ignored, by_key = [], [], {}
    for p in sorted(Path(folder).iterdir(), key=lambda x: x.name):
        if not p.is_file():
            continue
        m = NAME.fullmatch(p.name)
        if not m:
            ignored.append(p.name); continue
        if p.is_symlink() or not inside(p, folder):
            raise ValueError('Image cannot be a symlink: ' + p.name)
        key = (int(m[1]), int(m[2]))
        if key in by_key:
            raise ValueError('Duplicate song/page: ' + by_key[key].name + ' and ' + p.name)
        by_key[key] = p
        valid.append((key, p))
    return sorted(valid), ignored


def certify_sources(app):
    bundle = app.parent
    manifest_file = bundle / 'manifest.json'
    if manifest_file.is_file():
        manifest = read_json(manifest_file)
        if not manifest.get('files'):
            raise ValueError('Empty or unreadable deployment manifest')
        for row in manifest['files']:
            p = bundle / row['path']
            if not inside(p, bundle) or p.is_symlink() or not p.is_file():
                raise ValueError('Missing/unsafe bundle asset: ' + row['path'])
            if p.stat().st_size != row['bytes'] or digest(p) != row['sha256']:
                raise ValueError('Bundle integrity mismatch: ' + row['path'])
        return digest(manifest_file)
    # Development tree only. Real SD releases have an immutable manifest.
    h = hashlib.sha256()
    for p in sorted(app.rglob('*')):
        if p.suffix in {'.py', '.json'} and not {'__pycache__', 'tests', 'experiments'} & set(p.parts):
            h.update(p.relative_to(app).as_posix().encode()); h.update(digest(p).encode())
    return h.hexdigest()


@contextmanager
def single_runner(path):
    path = Path(path)
    with path.open('a+b') as f:
        if path.stat().st_size == 0:
            f.write(b' '); f.flush()
        f.seek(0)
        try:
            if sys.platform == 'win32':
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another SD batch is already running') from exc
        try:
            yield
        finally:
            f.seek(0)
            if sys.platform == 'win32':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


def windows_processes():
    import ctypes as C
    from ctypes import wintypes as W
    class Entry(C.Structure):
        _fields_=[('size',W.DWORD),('usage',W.DWORD),('pid',W.DWORD),('heap',C.c_size_t),
                  ('module',W.DWORD),('threads',W.DWORD),('parent',W.DWORD),
                  ('priority',C.c_long),('flags',W.DWORD),('exe',W.WCHAR*260)]
    kernel=C.WinDLL('kernel32',use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes=[W.DWORD,W.DWORD];kernel.CreateToolhelp32Snapshot.restype=W.HANDLE
    kernel.Process32FirstW.argtypes=[W.HANDLE,C.POINTER(Entry)]
    kernel.Process32NextW.argtypes=[W.HANDLE,C.POINTER(Entry)]
    kernel.CloseHandle.argtypes=[W.HANDLE]
    handle=kernel.CreateToolhelp32Snapshot(2,0)
    if handle == W.HANDLE(-1).value:
        raise OSError(C.get_last_error(),'Cannot enumerate worker processes')
    result={};entry=Entry();entry.size=C.sizeof(Entry)
    try:
        more=kernel.Process32FirstW(handle,C.byref(entry))
        while more:
            result[int(entry.pid)]=int(entry.parent)
            more=kernel.Process32NextW(handle,C.byref(entry))
    finally:kernel.CloseHandle(handle)
    return result


def windows_descendants(pid, known=()):
    parents=windows_processes();seen={pid}|set(known)
    while True:
        expanded=seen|{p for p,parent in parents.items() if parent in seen}
        if expanded==seen:return seen
        seen=expanded


def process_group_rss(pid):
    if sys.platform == 'linux':
        total = 0
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                if int(fields[2]) == pid:  # pgrp, after comm and state/ppid
                    rss_pages = int(fields[21])  # rss: field 24 of proc stat
                    total += rss_pages * os.sysconf('SC_PAGE_SIZE')
            except (OSError, ValueError, IndexError):
                continue
        return total
    if sys.platform != 'win32':
        raise RuntimeError('RSS monitoring supports Linux and explicit Windows desktop validation')
    import ctypes as C
    from ctypes import wintypes as W
    class Memory(C.Structure):
        _fields_=[('cb',W.DWORD),('faults',W.DWORD)]+[(name,C.c_size_t) for name in
            ('peak_working','working','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
    kernel=C.WinDLL('kernel32',use_last_error=True);psapi=C.WinDLL('psapi',use_last_error=True)
    kernel.OpenProcess.argtypes=[W.DWORD,W.BOOL,W.DWORD];kernel.OpenProcess.restype=W.HANDLE
    kernel.CloseHandle.argtypes=[W.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes=[W.HANDLE,C.POINTER(Memory),W.DWORD]
    total=0
    for child_pid in windows_descendants(pid):
        h=kernel.OpenProcess(0x1000|0x10,False,child_pid)
        if h:
            try:
                m=Memory();m.cb=C.sizeof(m)
                if psapi.GetProcessMemoryInfo(h,C.byref(m),m.cb):total+=m.working
            finally:kernel.CloseHandle(h)
    return total


def stop_group(process, known=()):
    if sys.platform == 'win32':
        import ctypes as C
        from ctypes import wintypes as W
        kernel=C.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[W.DWORD,W.BOOL,W.DWORD];kernel.OpenProcess.restype=W.HANDLE
        kernel.TerminateProcess.argtypes=[W.HANDLE,W.UINT]
        kernel.WaitForSingleObject.argtypes=[W.HANDLE,W.DWORD]
        kernel.CloseHandle.argtypes=[W.HANDLE]
        known=windows_descendants(process.pid,known)
        for _ in range(2):
            remaining=windows_descendants(process.pid,known);known|=remaining
            if process.poll() is not None:remaining.discard(process.pid)
            # Terminate the leader first to stop it creating more model children.
            for pid in sorted(remaining,key=lambda value:value!=process.pid):
                h=kernel.OpenProcess(0x100001,False,pid)
                if h:
                    try:
                        kernel.TerminateProcess(h,1)
                        if kernel.WaitForSingleObject(h,5000)==0x102:
                            raise RuntimeError('Worker descendant did not terminate: '+str(pid))
                    finally:kernel.CloseHandle(h)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            deadline=time.monotonic()+2
            while time.monotonic()<deadline:
                time.sleep(.05)
                os.killpg(process.pid,0)
            os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill(); process.wait(timeout=5)


def worker_command(config_file):
    return [sys.executable, str(Path(__file__).resolve()), '--_worker', str(config_file)]


def available_memory_mib():
    if sys.platform=='linux':
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):return int(line.split()[1])//1024
    return None


def launch_worker(config, directory, timeout, memory_limit):
    available=available_memory_mib()
    effective_limit=min(memory_limit,available-384) if available is not None else memory_limit
    if effective_limit<256:
        raise RuntimeError('Insufficient available RAM to reserve 384 MiB for the operating system')
    config_file = directory / 'job.json'
    atomic_json(config_file, config)
    log_path = directory / 'worker.log'
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS=str(config['threads']),
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    # Avoid tmpfs (/tmp) exhausting the board's DDR. All worker scratch is on SD.
    scratch = directory / 'scratch'; scratch.mkdir()
    env.update(TMPDIR=str(scratch), TEMP=str(scratch), TMP=str(scratch))
    kwargs = {'start_new_session': True} if sys.platform != 'win32' else {
        'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
    start, peak, reason = time.monotonic(), 0, None
    with log_path.open('wb') as log:
        child = subprocess.Popen(worker_command(config_file), stdout=log, stderr=subprocess.STDOUT,
                                 env=env, cwd=Path(__file__).resolve().parent, **kwargs)
        known_windows={child.pid}
        try:
            while child.poll() is None:
                if sys.platform == 'win32':
                    # Keep ancestry after a venv launcher/worker exits first.
                    known_windows |= windows_descendants(child.pid,known_windows)
                peak = max(peak, process_group_rss(child.pid))
                if peak > effective_limit * 1024 * 1024:
                    reason = 'memory_limit'; stop_group(child,known_windows); break
                if time.monotonic() - start > timeout:
                    reason = 'timeout'; stop_group(child,known_windows); break
                time.sleep(.1)
        finally:
            # A failed/successful worker must not leave an unmonitored model child.
            stop_group(child,known_windows)
    stats = {'elapsed_seconds': round(time.monotonic()-start, 3),
             'process_group_peak_rss_mib': round(peak/1024/1024, 2),
             'rss_sampling_seconds': .1, 'returncode': child.returncode,
             'configured_memory_limit_mib': memory_limit, 'effective_memory_limit_mib': effective_limit,
             'mem_available_before_worker_mib': available,
             'stop_reason': reason, 'log': str(log_path)}
    if reason or child.returncode:
        raise RuntimeError(json.dumps(stats, ensure_ascii=False))
    report = read_json(directory / 'worker_report.json')
    if not report or report.get('error'):
        raise RuntimeError('Worker report missing or unsuccessful: ' + str(log_path))
    return report, stats


def runtime_probe(desktop):
    import board_homr as board
    if not desktop:
        result=board.runtime_preflight()
        available=result.get('mem_available_kib')
        if available is not None and available < 1_500_000:
            result['errors'].append('Less than 1.5 GB MemAvailable: close desktop applications before inference')
        if not result['errors']:
            certify_installed_models()
        return result
    # The only bypass is explicit PC validation. Real inference/dependencies stay active.
    errors, models = [], {}
    if sys.version_info[:2] != (3, 12):
        errors.append('Desktop comparison also requires Python 3.12')
    try:
        import cv2, numpy, rapidocr, homr.main, onnxruntime as ort
        import homr
        if 'CPUExecutionProvider' not in ort.get_available_providers():
            errors.append('CPUExecutionProvider missing')
        package = Path(homr.__file__).parent
        for folder, prefix in [('segmentation','segnet_'),('transformer','encoder_'),('transformer','decoder_')]:
            paths = list((package / folder).glob(prefix+'*.onnx'))
            models[prefix] = [str(p) for p in paths]
            if len(paths) != 1:
                errors.append('Expected exactly one installed ' + prefix + '*.onnx')
    except Exception as exc:
        errors.append(str(exc))
    if not errors:certify_installed_models()
    return {'execution_target': 'desktop_validation_only', 'python': sys.version,
            'models': models, 'errors': errors, 'warnings': ['Not an ARM or physical SD validation']}


def certify_installed_models():
    expected_dir=Path(__file__).resolve().parent.parent/'models'
    if not expected_dir.is_dir():return  # source-tree development only
    import homr
    package=Path(homr.__file__).parent
    for expected in expected_dir.glob('*.onnx'):
        folder='segmentation' if expected.name.startswith('segnet_') else 'transformer'
        installed=package/folder/expected.name
        if not installed.is_file() or digest(installed)!=digest(expected):
            raise ValueError('Installed Homr model differs from frozen bundle: '+expected.name)


def worker_main(config_file):
    c = read_json(config_file)
    if not c:
        raise ValueError('Invalid worker configuration')
    if sys.platform == 'linux' and hasattr(os, 'sched_setaffinity'):
        cpus = sorted(os.sched_getaffinity(0))[:c['threads']]
        os.sched_setaffinity(0, cpus)
    probe = runtime_probe(c['desktop_validation'])
    if probe['errors']:
        raise RuntimeError('Runtime preflight: ' + '; '.join(probe['errors']))
    report_file = Path(config_file).parent / 'worker_report.json'
    if c.get('probe_only'):
        atomic_json(report_file, {'preflight': probe}); return
    import board_homr as board
    if c['desktop_validation']:
        board.runtime_preflight = lambda: probe
    result = board.transcribe(Path(c['input']), Path(c['staged_output']),
                             None if c['allow_directory'] else Path(c['sd_root']),
                             previous_score=Path(c['previous_score']) if c.get('previous_score') else None,
                             relation_models=Path(c['relation_models']) if c.get('relation_models') else None,
                             relations=c['relations'])
    atomic_json(report_file, {'preflight': probe, 'recognizer': result})


def prepare_root(root, allow_directory, init=False):
    root = Path(root).resolve(strict=True)
    if not root.is_dir() or (not allow_directory and not os.path.ismount(root)):
        raise ValueError('--sd-root must be an existing mounted filesystem; directory emulation requires --allow-directory')
    for name in ('Image', 'Outputs'):
        p = root / name
        if p.is_symlink() or not inside(p, root):
            raise ValueError('SD directory escapes the card: ' + name)
        if init or name == 'Outputs':
            p.mkdir(exist_ok=True)
        if not p.is_dir():
            raise ValueError('Missing SD folder ' + name + '; use --init once')
    return root


def batch(args):
    root = prepare_root(args.sd_root, args.allow_directory, args.init)
    target = 'desktop_validation_only' if args.desktop_validation else 'aarch64_linux'
    if args.init:
        return {'execution_target': target, 'initialized': str(root)}, 0
    pages, ignored = scan_images(root/'Image')
    outputs, control = root/'Outputs', root/'Outputs'/'.omr'
    control.mkdir(exist_ok=True)
    for p in [outputs, control]:
        if p.is_symlink() or not inside(p, root):
            raise ValueError('Unsafe output/control directory')
    if shutil.disk_usage(root).free < 256*1024*1024:
        raise ValueError('Need at least 256 MiB free on SD for bounded working files')
    fingerprint = certify_sources(Path(__file__).resolve().parent)
    relation_models = Path(__file__).resolve().parent.parent/'relation_models'
    if not relation_models.is_dir():
        relation_models = Path(__file__).resolve().parent/'relation_models'
    if args.relation_models:
        relation_models = args.relation_models.resolve(strict=True)
    if args.relations == 'auto' and not (relation_models/'manifest.json').is_file():
        raise ValueError('Validated relation models are required in auto mode; use --relations off explicitly for a base-only run')
    config = {'sd_root': str(root), 'allow_directory': args.allow_directory,
              'desktop_validation': args.desktop_validation, 'threads': args.threads,
              'relations': args.relations, 'relation_models': str(relation_models) if args.relations != 'off' else None}
    signature = hashlib.sha256(json.dumps({'source': fingerprint, 'options': config}, sort_keys=True).encode()).hexdigest()
    summary = {'schema': VERSION, 'execution_target': target, 'started_utc': now(),
               'sd_root': str(root), 'pipeline_sha256': fingerprint, 'options': config,
               'ignored_files': ignored, 'pages': [], 'physical_board_validated': False}
    with single_runner(control/'runner.lock'):
        probe_dir = control/('preflight-'+uuid.uuid4().hex); probe_dir.mkdir()
        report, stats = launch_worker({**config, 'probe_only': True}, probe_dir,
                                     min(args.timeout_seconds, 120), args.memory_limit_mib)
        summary['preflight'] = report['preflight']
        if args.check:
            summary['discovered_pages'] = [p.name for _,p in pages]
            summary['checked_utc'] = now()
            atomic_json(control/'preflight.json', summary)
            remove_work(probe_dir,control)
            return summary, 0
        remove_work(probe_dir,control)
        previous = None
        previous_key = None
        for key, image in pages:
            stem = f'{key[0]:02d}-{key[1]:02d}'
            dest, state_path = outputs/(stem+'.musicxml'), control/(stem+'.json')
            if dest.is_symlink() or state_path.is_symlink():
                raise ValueError('Output/state symlink is not allowed')
            if previous_key != (key[0], key[1]-1):
                previous = None
            old = read_json(state_path)
            image_hash = digest(image)
            prev_hash = digest(previous) if previous else None
            job_signature = hashlib.sha256((signature+image_hash+str(prev_hash)).encode()).hexdigest()
            row = {'page': stem, 'input': image.name, 'input_sha256': image_hash,
                   'job_sha256': job_signature, 'previous_score': str(previous) if previous else None,
                   'execution_target': target, 'started_utc': now()}
            owned_hashes = {old.get('output_sha256'), old.get('owned_output_sha256')}
            current_hash = digest(dest) if dest.is_file() else None
            if (old.get('state') in {'succeeded','ready_to_commit'} and current_hash
                    and current_hash == old.get('output_sha256') and old.get('job_sha256') == job_signature):
                row.update(state='skipped_verified', output=str(dest), output_sha256=current_hash, **validate_xml(dest))
                if old.get('state') == 'ready_to_commit':
                    atomic_json(state_path, {**old, 'state': 'succeeded', 'recovered_commit': True})
                previous, previous_key = dest, key
            elif current_hash and current_hash not in owned_hashes:
                row.update(state='failed', error='Existing XML is not owned by this run or was externally modified; preserved')
                previous, previous_key = None, None
            else:
                work = control/('work-'+stem+'-'+uuid.uuid4().hex); work.mkdir()
                row.update(state='running', owned_output_sha256=current_hash, work_directory=str(work))
                atomic_json(state_path, row)
                try:
                    report, stats = launch_worker({**config, 'input': str(image),
                        'staged_output': str(work/'result.musicxml'), 'previous_score': str(previous) if previous else None},
                        work, args.timeout_seconds, args.memory_limit_mib)
                    staged = work/'result.musicxml'
                    metrics = validate_xml(staged)
                    if digest(image) != image_hash:
                        raise ValueError('Input photo changed during recognition; staged output not committed')
                    output_hash = digest(staged)
                    if dest.exists():
                        history = control/'history'; history.mkdir(exist_ok=True)
                        shutil.copyfile(dest, history/(stem+'.'+current_hash+'.musicxml'))
                    row.update(state='ready_to_commit', output=str(dest), output_sha256=output_hash,
                               worker_stats=stats, **metrics)
                    atomic_json(state_path, row)
                    # The recognizer fsyncs this staged file. Commit on the same SD filesystem.
                    os.replace(staged, dest); sync_directory(outputs)
                    if digest(dest) != output_hash:
                        raise RuntimeError('SD output readback checksum failed')
                    row.update(state='succeeded', finished_utc=now())
                    atomic_json(state_path, row)
                    previous, previous_key = dest, key
                    # Keep diagnostic report/log/geometry, release bulk scratch after success.
                    remove_work(work/'scratch',control)
                except KeyboardInterrupt:
                    row.update(state='interrupted', finished_utc=now()); atomic_json(state_path,row); raise
                except Exception as exc:
                    row.update(state='failed', error=str(exc), finished_utc=now())
                    atomic_json(state_path, row)
                    previous, previous_key = None, None
            summary['pages'].append(row)
            atomic_json(control/'batch.json', summary)
            print(json.dumps({k:row[k] for k in ('page','state','error') if k in row}, ensure_ascii=False), flush=True)
        summary.update(finished_utc=now(), succeeded=sum(r['state']=='succeeded' for r in summary['pages']),
                       skipped_verified=sum(r['state']=='skipped_verified' for r in summary['pages']),
                       failed=sum(r['state']=='failed' for r in summary['pages']))
        atomic_json(control/'batch.json', summary)
    return summary, int(bool(summary['failed']))


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if len(argv)==2 and argv[0]=='--_worker':
        worker_main(Path(argv[1])); return 0
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sd-root',type=Path,required=True)
    p.add_argument('--threads',type=int,choices=range(1,5),default=4)
    p.add_argument('--memory-limit-mib',type=int,default=2300)
    p.add_argument('--timeout-seconds',type=float,default=600)
    p.add_argument('--relations',choices=['auto','off'],default='auto')
    p.add_argument('--relation-models',type=Path)
    p.add_argument('--allow-directory',action='store_true',help='Emulate an SD mount with a directory; does not bypass ARM preflight')
    p.add_argument('--desktop-validation',action='store_true',help='Explicit PC-only real-model validation; never an ARM/SD claim')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--check',action='store_true');mode.add_argument('--init',action='store_true')
    a=p.parse_args(argv)
    if not 256<=a.memory_limit_mib<=2800 or a.timeout_seconds<=0:
        p.error('memory limit must be 256..2800 MiB; timeout must be positive')
    try:
        summary, code=batch(a)
        print(json.dumps(summary,ensure_ascii=False)); return code
    except KeyboardInterrupt:
        print('Interrupted; committed MusicXML files are retained. Run again to resume.',file=sys.stderr);return 130
    except Exception as exc:
        print(json.dumps({'state':'error','error':str(exc)},ensure_ascii=False),file=sys.stderr);return 2


if __name__=='__main__':
    raise SystemExit(main())
