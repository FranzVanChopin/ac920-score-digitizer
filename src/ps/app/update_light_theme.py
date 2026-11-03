"""Apply UI-only light-theme upgrade to the verified card; preserve dark backup."""
import json
import os
import pwd
import signal
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,'/opt/ac920-panel')
from onecard_common import atomic,boot_header,run,runtime_card,sha,write_json
from panel_ui import request

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OLD='88923BC825538F7CD0F11287E776FF9ACF289659C58333E8009463205AA350E1'


def main():
    if os.geteuid()!=0:raise RuntimeError('requires root')
    config=json.loads(Path('/etc/ac920-onecard.json').read_text())
    device=runtime_card(config)
    state=request(dict(action='state'),timeout=2)
    if state['busy'] or state['return_pending'] or state['play']['state']=='ready':
        raise RuntimeError('controller busy or playback active; refusing to interrupt it')
    mount=Path('/mnt/ac920-onecard')
    if sha(mount/'BOOT.BIN')!=OLD or config['capture_sha256']!=OLD:raise RuntimeError('unexpected current firmware')
    backup=Path('/home/fpga/codex-ui1-dark-backup-20261030')
    record=json.loads((backup/'backup.json').read_text())
    if sha(backup/'dark_ui1.tar.gz')!=record['archive_sha256']:raise RuntimeError('dark backup SHA mismatch')
    seal=json.loads((ROOT/'validation/release_identity.json').read_text())
    if seal['version']!='a4_interactive_ui1_light1' or sha(ROOT/'BOOT.BIN')!=seal['boot_sha256']:
        raise RuntimeError('new light firmware seal mismatch')
    if not boot_header((ROOT/'BOOT.BIN').read_bytes()):raise RuntimeError('new BOOT header invalid')
    # Capture the existing GUI display environment privately, to relaunch the
    # same application after replacing its drawing code. No session data logged.
    gui=[];environment=None
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():continue
        try:
            args=(item/'cmdline').read_bytes().split(b'\0')
            if args[:2]!=[b'/usr/bin/python3',b'/opt/ac920-panel/panel_ui.py']:continue
            gui.append(int(item.name))
            old=dict(value.split(b'=',1) for value in (item/'environ').read_bytes().split(b'\0') if b'=' in value)
            environment={key:old[key.encode()].decode() for key in
                ('DISPLAY','XAUTHORITY','DBUS_SESSION_BUS_ADDRESS','XDG_RUNTIME_DIR','HOME') if key.encode() in old}
        except (FileNotFoundError,ProcessLookupError):continue
    run('systemctl','stop','ac920-panel.service')
    run('umount',mount)
    run('mount','-o','rw',device,mount)
    try:
        manifest=json.loads((mount/'OneCard/manifest.json').read_text())
        if sha(mount/'BOOT.BIN')!=OLD:raise RuntimeError('BOOT changed before update')
        prefix=Path('/opt/ac920-panel').resolve()
        if prefix!=Path('/opt/ac920-panel-ui1'):raise RuntimeError('unexpected UI prefix')
        for name in ('panel_ui.py','collect_final.py','render_light_preview.py','update_light_theme.py'):
            atomic(prefix/name,(HERE/name).read_bytes());os.chmod(prefix/name,0o644)
        config.update(capture_sha256=seal['boot_sha256'],interface_version=seal['version'],theme='light1')
        manifest.update(capture_sha256=seal['boot_sha256'],interface_version=seal['version'],theme='light1')
        atomic(mount/'BOOT.BIN',(ROOT/'BOOT.BIN').read_bytes())
        if sha(mount/'BOOT.BIN')!=seal['boot_sha256']:raise RuntimeError('BOOT readback mismatch')
        write_json('/etc/ac920-onecard.json',config)
        write_json(mount/'OneCard/manifest.json',manifest)
        os.sync()
    finally:run('umount',mount)
    run('systemctl','start','ac920-panel.service')
    if environment is not None:
        for pid in gui:
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:pass
        time.sleep(2) # GNOME may relaunch its autostart entry itself.
        restarted=False
        for item in Path('/proc').iterdir():
            if not item.name.isdigit():continue
            try:
                args=(item/'cmdline').read_bytes().split(b'\0')
                if args[:2]==[b'/usr/bin/python3',b'/opt/ac920-panel/panel_ui.py']:restarted=True;break
            except (FileNotFoundError,ProcessLookupError):continue
        if not restarted:
            output=ROOT/'light_ui.log'
            with output.open('ab') as log:
                subprocess.Popen(['runuser','-u',config['user'],'--','env',
                    *(key+'='+value for key,value in environment.items()),
                    '/usr/bin/python3','/opt/ac920-panel/panel_ui.py'],stdout=log,stderr=log,start_new_session=True)
            os.chmod(output,0o644)
    result=dict(theme='light1',boot_sha256=seal['boot_sha256'],backup_sha256=record['archive_sha256'],
                controller_restarted=True,gui_relaunched=environment is not None)
    write_json(ROOT/'theme_installed.json',result);os.chmod(ROOT/'theme_installed.json',0o644)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
