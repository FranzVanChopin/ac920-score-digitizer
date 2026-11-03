"""Recognize a completed capture request, export scores, safely return to slot0.
 * Recognition writes ext4. The capture FAT stays read-only until final export.
 * A failed page or any sync/unmount/register failure prevents automatic reboot.
"""
import argparse,datetime,json,os,re,subprocess,sys,time
from pathlib import Path
from onecard_common import run,sha,atomic,write_json,runtime_card,mounted,read_multiboot,select_boot
def completed_pages(album,state):
    pages=state.get('pages',{});count=0
    for meta in sorted((album/'Meta').glob('??-??.json')):
        if not re.fullmatch(r'\d{2}-\d{2}\.json',meta.name):continue
        item=json.loads(meta.read_text())
        if item.get('complete') is not True:continue
        key=meta.stem;grey=album/'Grey'/(key+'.bmp');done=pages.get(key,{})
        if not grey.is_file() or done.get('status')!='ok' or done.get('sha256','').lower()!=sha(grey).lower():
            raise RuntimeError('page incomplete/failed/changed: '+key)
        count+=1
    return count
def export_results(results,destination):
    destination.mkdir(parents=True,exist_ok=True)
    for p in sorted(results.glob('??.musicxml')):
        if not re.fullmatch(r'\d{2}\.musicxml',p.name):continue
        atomic(destination/p.name,p.read_bytes())
    if (results/'album.json').is_file():atomic(destination/'album.json',(results/'album.json').read_bytes())
def execute(config,manual_return=False,*,mount_dir=None,state_dir=None,log_dir=None):
    if os.geteuid()!=0:raise RuntimeError('run with sudo/systemd')
    dev=runtime_card(config);mount=Path(mount_dir or '/mnt/ac920-onecard')
    if mounted(dev):raise RuntimeError('boot FAT already mounted elsewhere; refusing to interfere: '+repr(mounted(dev)))
    mount.mkdir(parents=True,exist_ok=True);attached=False;old_slot=None
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    try:
        run('mount','-o','ro',dev,mount);attached=True
        manifest=json.loads((mount/'OneCard/manifest.json').read_text())
        if manifest['root_partuuid']!=config['root_partuuid'] or manifest['boot_partuuid']!=config['boot_partuuid']:
            raise RuntimeError('card manifest mismatch')
        if sha(mount/'BOOT.BIN')!=config['capture_sha256']:raise RuntimeError('capture BOOT changed')
        request=mount/'OneCard/REQUEST.JSON'
        if not manual_return and not request.is_file():print('[ONECARD] No request; Ubuntu stays running',flush=True);return
        if not manual_return:
            value=json.loads(request.read_text())
            if value.get('schema')!=1 or value.get('action')!='recognize':raise RuntimeError('invalid capture request')
            base=Path(state_dir or '/var/lib/ac920-onecard');state=base/'results';work=base/'work'
            log=Path(log_dir or '/var/log/ac920-onecard')/(stamp+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            print('[ONECARD] Recognition started; log='+str(log),flush=True)
            command=['runuser','-u',config['user'],'--','env','APP='+config['app'],
                     'PY='+config['python'],'WORK='+str(work),'sh',str(Path(config['recognizer'])/'album/run_album.sh'),
                     str(mount),str(state),'--rotate',config['rotate']]
            with log.open('xb') as output:
                proc=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT)
                output.flush();os.fsync(output.fileno())
            if proc.returncode:raise RuntimeError('recognizer exit '+str(proc.returncode)+'; see '+str(log))
            done=json.loads((state/'album.json').read_text())
            count=completed_pages(mount,done) # album can exit0 even when individual pages failed
            songs={k[:2] for k,v in done.get('pages',{}).items() if v.get('status')=='ok'}
            if any(not (state/(song+'.musicxml')).is_file() for song in songs):raise RuntimeError('merged MusicXML missing')
            run('mount','-o','remount,rw',mount)
            export_results(state,mount/'Outputs')
            atomic(mount/'Outputs/last_run.log',log.read_bytes())
            result=dict(schema=1,status='complete',pages=count,finished=stamp,rotate=config['rotate'])
            write_json(mount/'Outputs/onecard_status.json',result)
            write_json(mount/'OneCard/LAST.JSON',result)
            # Acknowledgment is last, after score export; interruption is retryable.
            os.replace(request,mount/'OneCard/DONE.JSON')
            print(f'[ONECARD] {count} pages complete; MusicXML exported to Outputs',flush=True)
        os.sync();run('umount',mount);attached=False
        if mounted(dev):raise RuntimeError('boot FAT remains mounted')
        run('blockdev','--flushbufs',dev)
        old_slot=read_multiboot();select_boot(0)
        print('[ONECARD] FAT unmounted; MultiBoot=0 verified; reboot to capture',flush=True)
        subprocess.run(['systemctl','reboot'],check=True)
    except BaseException:
        if old_slot is not None:
            try:select_boot(old_slot)
            except Exception as e:print('[ONECARD] MultiBoot rollback failed: '+str(e),file=sys.stderr)
        raise
    finally:
        if attached:
            os.sync();run('umount',mount)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='/etc/ac920-onecard.json')
    parser.add_argument('--return-to-capture',action='store_true')
    args=parser.parse_args()
    if args.return_to_capture:
        # Manual recovery is explicit; avoid concurrent recognizer/FAT access.
        run('systemctl','stop','ac920-onecard.service')
    try:execute(json.loads(Path(args.config).read_text()),args.return_to_capture)
    except Exception as e:print('[ONECARD] FAILED; no automatic return: '+str(e),file=sys.stderr,flush=True);return 1
    return 0
if __name__=='__main__':raise SystemExit(main())
