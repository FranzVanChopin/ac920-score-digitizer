"""Install the UI on the verified one-card system. Never repartitions or formats."""
import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import zlib
from pathlib import Path
from onecard_common import atomic, boot_header, mounted, run, runtime_card, sha, write_json
from patch_progress import make as progress_patch
from prepare_dtb import make as dtb_patch
from boot_script import unpack

HERE = Path(__file__).resolve().parent
OLD_SHA = '47CB0381C0493B5FF89DD32E5ED013012999C0DFA57DCFE6540C08B73EB8A37C'
BACKUP_SHA = '9F37713B25F7D24993609FF27A5AC629DE2CD4C19D3DF2C531DAD7459E53B1C1'


def boot_script(image):
    header, text = unpack(image)
    before = 'fpga loadb 0 ${bitstream_load_address} ${bitstream_size}'
    if text.count(before) != 1: raise RuntimeError('Unknown vendor bitstream load command')
    text = text.replace(before, 'fpga loadb 0 ${bitstream_load_address} ${filesize}')
    # Request a readable UI on DP; only when the monitor advertises this mode.
    # Keep standard EDID selection rather than forcing a disconnected sink.
    payload = text.encode() + b'\n'
    payload = struct.pack('>II', len(payload), 0) + payload
    if image[64:66] == b'\x1f\x8b':
        import gzip
        payload = gzip.compress(payload, mtime=0)
    header[1] = 0; header[3] = len(payload); header[6] = zlib.crc32(payload)
    raw = struct.pack('>7I4B32s', *header); header[1] = zlib.crc32(raw)
    result = struct.pack('>7I4B32s', *header) + payload
    assert '${filesize}' in unpack(result)[1]
    return result


def prepare(config, where):
    if sha(Path('/home/fpga/codex-interactive-original-20261030/original.tar.gz')) != BACKUP_SHA:
        raise RuntimeError('Original rollback archive missing or changed')
    if sha(where/'BOOT.BIN') != OLD_SHA: raise RuntimeError('Installed BOOT is not the backed-up DP2')
    build = HERE.parent / 'build'
    if not boot_header((build/'BOOT.BIN').read_bytes()): raise RuntimeError('New BOOT header invalid')
    seal = json.loads((HERE.parent/'validation/release_identity.json').read_text())
    if sha(build/'BOOT.BIN') != seal['boot_sha256']: raise RuntimeError('New BOOT seal mismatch')
    stage = HERE.parent/'prepared'; stage.mkdir(exist_ok=True)
    dtb_patch(where/'linux/system.dtb', stage/'system.dtb')
    progress_patch(config['recognizer'], stage/'engine')
    atomic(stage/'boot.scr', boot_script((where/'boot.scr').read_bytes()))
    print('PREPARED', dict(boot=seal['boot_sha256'], dtb=sha(stage/'system.dtb')), flush=True)
    return stage, seal


def apply(config, where, stage, seal):
    if sha(where/'BOOT.BIN') != OLD_SHA: raise RuntimeError('Old BOOT changed since prepare')
    prefix = Path('/opt/ac920-panel-ui1')
    if prefix.exists(): raise RuntimeError('Version prefix exists; inspect partial install instead of overwriting')
    prefix.mkdir()
    for path in HERE.glob('*.py'): shutil.copy2(path, prefix/path.name)
    shutil.copytree(stage/'engine', prefix/'engine')
    for path in prefix.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    bridge=HERE.parent/'build/debs/python3-gi-cairo_3.42.1-0ubuntu1_arm64.deb'
    if sha(bridge) != '6EDD8BEA1453263BC408EDB41359517574B2E22CAA07A385CFF3D99998EBD7F7':
        raise RuntimeError('GTK Cairo bridge package hash mismatch')
    subprocess.run(['dpkg','-i',str(bridge)],check=True)
    font=HERE.parent/'build/debs/fonts-wqy-microhei_0.2.0-beta-3.1_all.deb'
    if sha(font)!='58D0A29AF79CE1FC5C63B4276065F0FCE27DECF6839835F3F9C2E68D9F84CA20':
        raise RuntimeError('CJK font package hash mismatch')
    subprocess.run(['dpkg','-i',str(font)],check=True)
    subprocess.run(['/usr/bin/python3','-c','import gi;gi.require_foreign("cairo")'],check=True)
    # Wheel installs are self-contained UI dependencies; the existing models and
    # recognizer implementation stay byte-identical.
    subprocess.run([config['python'], '-m', 'pip', 'install', '--no-index', '--find-links',
        str(HERE.parent/'build/wheels'), 'verovio==6.3.0', 'mido==1.3.3'], check=True)
    subprocess.run([config['python'], '-c', 'import verovio,mido; print(verovio.toolkit().getVersion())'], check=True)
    Path('/opt/ac920-panel').symlink_to(prefix)
    shutil.copy2(HERE/'ac920-panel.service', '/etc/systemd/system/ac920-panel.service')
    shutil.copy2(HERE/'ac920-panel.desktop', '/etc/xdg/autostart/ac920-panel.desktop')
    lightdm=Path('/etc/lightdm/lightdm.conf.d/90-ac920-panel.conf')
    lightdm.parent.mkdir(parents=True, exist_ok=True)
    atomic(lightdm, ('[Seat:*]\nautologin-user='+config['user']+
        '\nautologin-user-timeout=0\nuser-session=ubuntu-xorg\nautologin-session=ubuntu-xorg\n').encode())
    for unit, command in (
        ('serial-getty@ttyPS0.service', '-/sbin/agetty --autologin '+config['user']+' --keep-baud 115200,38400,9600 %I $TERM'),
        ('getty@tty1.service', '-/sbin/agetty --autologin '+config['user']+' --noclear %I $TERM')):
        target=Path('/etc/systemd/system')/(unit+'.d/autologin.conf')
        atomic(target, ('[Service]\nExecStart=\nExecStart='+command+'\n').encode())
    subprocess.run(['usermod', '-a', '-G', 'audio,video', config['user']], check=True)
    # DP codec clock supports the exact48k family on the current PS PLL setup.
    import pwd
    account=pwd.getpwnam(config['user'])
    pulse=Path(account.pw_dir)/'.config/pulse/daemon.conf'
    backup=Path('/home/fpga/codex-interactive-original-20261030')/'pulse-daemon-before-ui1'
    if pulse.exists(): shutil.copy2(pulse,backup)
    else: backup.write_text('ABSENT\n')
    atomic(pulse,b'default-sample-rate = 48000\nalternate-sample-rate = 48000\navoid-resampling = yes\n')
    os.chown(pulse,account.pw_uid,account.pw_gid);os.chmod(pulse,0o644)
    subprocess.run(['systemctl','disable','ac920-onecard.service'], check=True)
    # This maintenance boot masks the old unit in /run. systemctl disable can
    # ignore that unit, so remove only its known persistent enable symlink too.
    old_enable=Path('/etc/systemd/system/multi-user.target.wants/ac920-onecard.service')
    if old_enable.is_symlink(): old_enable.unlink()
    elif old_enable.exists(): raise RuntimeError('Unexpected non-symlink old enable entry')
    subprocess.run(['systemctl','daemon-reload'], check=True)
    subprocess.run(['systemctl','enable','ac920-panel.service'], check=True)
    # Files installed first. FAT changes are verified; capture BOOT commits last.
    run('mount','-o','remount,rw',where)
    try:
        atomic(where/'linux/system.dtb', (stage/'system.dtb').read_bytes())
        atomic(where/'linux/system.bit', (HERE.parent/'build/camera_capture_wrapper.bit').read_bytes())
        atomic(where/'boot.scr', (stage/'boot.scr').read_bytes())
        next_config = dict(config, capture_sha256=seal['boot_sha256'], interface='interactive-ui1')
        manifest=json.loads((where/'OneCard/manifest.json').read_text())
        manifest.update(capture_sha256=seal['boot_sha256'], interface='interactive-ui1',
                        linux_dtb_sha256=sha(where/'linux/system.dtb'),
                        linux_bit_sha256=sha(where/'linux/system.bit'))
        write_json(where/'OneCard/manifest.json', manifest)
        write_json('/etc/ac920-onecard.json', next_config)
        atomic(where/'BOOT.BIN', (HERE.parent/'build/BOOT.BIN').read_bytes())
        if sha(where/'BOOT.BIN') != seal['boot_sha256']: raise RuntimeError('BOOT readback SHA mismatch')
        write_json(HERE.parent/'installed.json', dict(boot_sha256=seal['boot_sha256'],
                   dtb_sha256=sha(where/'linux/system.dtb'), bit_sha256=sha(where/'linux/system.bit'),
                   backup_sha256=BACKUP_SHA, status='INSTALLED_REBOOT_PENDING'))
        os.sync()
    finally:
        run('mount','-o','remount,ro',where)
    print('INSTALLED; safe reboot required to activate new PL, keys/audio DT, auto-login and UI.', flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--apply', action='store_true')
    args=parser.parse_args()
    if os.geteuid()!=0: raise RuntimeError('requires root')
    config=json.loads(Path('/etc/ac920-onecard.json').read_text())
    dev=runtime_card(config); where=Path('/mnt/ac920-ui-install')
    if mounted(dev): raise RuntimeError('FAT mounted; stop controller before installation')
    where.mkdir(exist_ok=True); run('mount','-o','ro',dev,where)
    try:
        stage,seal=prepare(config,where)
        if args.apply: apply(config,where,stage,seal)
    finally:
        os.sync(); run('umount',where)


if __name__=='__main__': main()
