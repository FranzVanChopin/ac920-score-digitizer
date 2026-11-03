"""Install the fixed-ccw review revision on the verified, backed-up single card."""
import json,os,pwd,signal,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,'/opt/ac920-panel')
from onecard_common import atomic,boot_header,mounted,run,runtime_card,sha,write_json
from panel_ui import request
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
OLD='C3F986B444C53D56B73E5BD2C5491C5E0DE574E02F395AF30BC5914298C6C397'

def main():
    if os.geteuid()!=0:raise RuntimeError('root required')
    config=json.loads(Path('/etc/ac920-onecard.json').read_text());device=runtime_card(config)
    state=request({'action':'state'},timeout=2)
    if state['busy'] or state['return_pending'] or state['play']['state']=='ready':raise RuntimeError('controller busy')
    if config['rotate']!='ccw':raise RuntimeError('fixed direction must remain ccw')
    mount=Path('/mnt/ac920-onecard')
    if config['capture_sha256']!=OLD or sha(mount/'BOOT.BIN')!=OLD:raise RuntimeError('unexpected current BOOT')
    backup=Path('/home/fpga/codex-light1-review-backup-20261030')
    backup_record=json.loads((backup/'backup.json').read_text())
    if sha(backup/'light1_before_review.tar.gz')!=backup_record['archive_sha256']:raise RuntimeError('backup SHA mismatch')
    for item in json.loads((ROOT/'payload_manifest.json').read_text()):
        if sha(ROOT/item['path'])!=item['sha256']:raise RuntimeError('payload mismatch: '+item['path'])
    seal=json.loads((ROOT/'validation/release_identity.json').read_text())
    if seal['version']!='a4_interactive_ui1_light2' or sha(ROOT/'BOOT.BIN')!=seal['boot_sha256'] or not boot_header((ROOT/'BOOT.BIN').read_bytes()):raise RuntimeError('BOOT seal mismatch')
    prefix=Path('/opt/ac920-panel').resolve()
    if prefix!=Path('/opt/ac920-panel-ui1'):raise RuntimeError('unexpected installed prefix')
    gui=[];environment=None
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():continue
        try:
            args=(item/'cmdline').read_bytes().split(b'\0')
            if args[:2]!=[b'/usr/bin/python3',b'/opt/ac920-panel/panel_ui.py']:continue
            gui.append(int(item.name))
            old=dict(value.split(b'=',1) for value in (item/'environ').read_bytes().split(b'\0') if b'=' in value)
            environment={key:old[key.encode()].decode() for key in ('DISPLAY','XAUTHORITY','DBUS_SESSION_BUS_ADDRESS','XDG_RUNTIME_DIR','HOME') if key.encode() in old}
        except (FileNotFoundError,ProcessLookupError):continue
    run('systemctl','stop','ac920-panel.service')
    # Old Light1 does not clean up SIGTERM. Only unmount its exact verified target.
    where=mounted(device)
    if where:
        if where!=[str(mount)]:raise RuntimeError('unexpected boot mount after old stop: '+repr(where))
        run('umount',mount)
    run('mount','-o','rw',device,mount)
    try:
        manifest=json.loads((mount/'OneCard/manifest.json').read_text())
        if sha(mount/'BOOT.BIN')!=OLD:raise RuntimeError('BOOT changed during update')
        for name in ('panel_ui.py','panel_model.py','interactive_service.py','patch_progress.py','render_score.py','headless_check.py','update_review.py','validate_review.py'):
            atomic(prefix/name,(HERE/name).read_bytes());os.chmod(prefix/name,0o644)
        for item in json.loads((ROOT/'album/provenance.json').read_text()):
            name=item['file']
            if name not in ('omr_album.py','merge_pages.py','engrave.py','README.md'):raise RuntimeError('unexpected album helper')
            target=Path(config['recognizer'])/'album'/name
            atomic(target,(ROOT/'album'/name).read_bytes());os.chmod(target,0o644)
        # Execute the new patcher against the newly installed upstream, unchanged inference.
        subprocess.run(['/usr/bin/python3','-c',
            'import sys;sys.path.insert(0,sys.argv[1]);from patch_progress import make;make(sys.argv[2],sys.argv[1]+"/engine")',
            str(prefix),config['recognizer']],check=True)
        atomic('/etc/systemd/system/ac920-panel.service',(HERE/'ac920-panel.service').read_bytes())
        os.chmod('/etc/systemd/system/ac920-panel.service',0o644)
        config.update(capture_sha256=seal['boot_sha256'],interface_version=seal['version'],theme='light2',
            piano_soundfont='/usr/share/sounds/sf2/FluidR3_GM.sf2')
        manifest.update(capture_sha256=seal['boot_sha256'],interface_version=seal['version'],theme='light2')
        atomic(mount/'BOOT.BIN',(ROOT/'BOOT.BIN').read_bytes())
        if sha(mount/'BOOT.BIN')!=seal['boot_sha256']:raise RuntimeError('BOOT readback mismatch')
        write_json('/etc/ac920-onecard.json',config);write_json(mount/'OneCard/manifest.json',manifest);os.sync()
    finally:run('umount',mount)
    # Use FluidSynth only as an offline renderer, never as an audio-device daemon.
    run('systemctl','--global','disable','fluidsynth.service')
    user=pwd.getpwnam(config['user'])
    subprocess.run(['runuser','-u',config['user'],'--','env','XDG_RUNTIME_DIR=/run/user/'+str(user.pw_uid),
        'systemctl','--user','stop','fluidsynth.service'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    run('systemctl','daemon-reload');run('systemctl','start','ac920-panel.service')
    if environment:
        for pid in gui:
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:pass
        time.sleep(1)
        with (ROOT/'gui.log').open('ab') as log:
            subprocess.Popen(['runuser','-u',config['user'],'--','env',*(key+'='+value for key,value in environment.items()),
                '/usr/bin/python3','/opt/ac920-panel/panel_ui.py'],stdout=log,stderr=log,start_new_session=True)
    write_json(ROOT/'installed.json',dict(version=seal['version'],boot_sha256=seal['boot_sha256'],
        backup_sha256=backup_record['archive_sha256'],rotate=config['rotate'],gui_relaunched=bool(environment)))
    print('PASS Light2 installed, fixed ccw, old version backed up; idle manual controller started.')

if __name__=='__main__':main()
