"""Apply same-version UI runtime fixes to our owned version prefix only."""
import os
import pwd
import shutil
import subprocess
from pathlib import Path
from onecard_common import sha,atomic
HERE=Path(__file__).resolve().parent


def main():
    prefix=Path('/opt/ac920-panel').resolve(strict=True)
    if prefix!=Path('/opt/ac920-panel-ui1'):raise RuntimeError('unowned prefix')
    bridge=HERE.parent/'build/debs/python3-gi-cairo_3.42.1-0ubuntu1_arm64.deb'
    if sha(bridge)!='6EDD8BEA1453263BC408EDB41359517574B2E22CAA07A385CFF3D99998EBD7F7':raise RuntimeError('package SHA')
    subprocess.run(['dpkg','-i',str(bridge)],check=True)
    font=HERE.parent/'build/debs/fonts-wqy-microhei_0.2.0-beta-3.1_all.deb'
    if sha(font)!='58D0A29AF79CE1FC5C63B4276065F0FCE27DECF6839835F3F9C2E68D9F84CA20':raise RuntimeError('font SHA')
    subprocess.run(['dpkg','-i',str(font)],check=True)
    for path in HERE.glob('*.py'):
        atomic(prefix/path.name,path.read_bytes());os.chmod(prefix/path.name,0o644)
    account=pwd.getpwnam('fpga')
    pulse=Path(account.pw_dir)/'.config/pulse/daemon.conf'
    backup=Path('/home/fpga/codex-interactive-original-20261030/pulse-daemon-before-ui1')
    if not backup.exists():
        if pulse.exists():backup.write_bytes(pulse.read_bytes())
        else:backup.write_bytes(b'ABSENT\n')
        os.chmod(backup,0o644)
    atomic(pulse,b'default-sample-rate = 48000\nalternate-sample-rate = 48000\navoid-resampling = yes\n')
    os.chown(pulse,account.pw_uid,account.pw_gid);os.chmod(pulse,0o644)
    subprocess.run(['/usr/bin/python3','-c','import gi;gi.require_foreign("cairo");print("GI Cairo bridge OK")'],check=True)
    print('Runtime updated, original pulse config backed up; reboot to apply audio defaults.')


if __name__=='__main__':main()
