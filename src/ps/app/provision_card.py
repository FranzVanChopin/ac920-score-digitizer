"""Run on the AC920's ORIGINAL Ubuntu card, with the 512GB card in USB.
Default is a read-only plan. --apply requires the exact freshly printed token.
Never formats p1 or a pre-existing unknown p2; never targets eMMC/system card.
"""
import argparse,contextlib,datetime,hashlib,json,os,platform,re,shutil,subprocess,sys
from pathlib import Path
from boot_script import rewrite
from onecard_common import (run,sha,atomic,write_json,blkid,parent,mounted,ready_bytes,
                            EXPECTED_BYTES,P1_BYTES,P1_OFFSET,P2_BYTES,boot_header,check_partitions,read_multiboot)
HERE=Path(__file__).resolve().parent
def inspect(device):
    device=str(Path(device).resolve(strict=True))
    if not re.fullmatch(r'/dev/sd[a-z]+',device):raise RuntimeError('only a whole external USB /dev/sdX device is accepted')
    root=str(Path(run('findmnt','-n','-o','SOURCE','/')).resolve())
    if not re.fullmatch(r'/dev/mmcblk\d+p2',root):raise RuntimeError('run on original AC920 Ubuntu SD p2')
    source=parent(root)
    if source==device:raise RuntimeError('target is the running system disk')
    size=int(run('blockdev','--getsize64',device))
    if size!=EXPECTED_BYTES:raise RuntimeError(f'target size {size} differs from expected {EXPECTED_BYTES}')
    properties=dict(x.split('=',1) for x in run('udevadm','info','--query=property','--name',device).splitlines() if '=' in x)
    if properties.get('ID_BUS')!='usb':raise RuntimeError('target is not USB')
    table=json.loads(run('sfdisk','--json',device))['partitiontable']
    parts=check_partitions(table,device)
    p1=parts[0]
    if blkid(p1['node'],'TYPE')!='vfat' or blkid(p1['node'],'LABEL')!='AC920_BOOT':raise RuntimeError('p1 must be FAT32 AC920_BOOT')
    if len(parts)==2:
        p2=parts[1]
        if blkid(p2['node'],'TYPE')!='ext4':raise RuntimeError('existing p2 is not ext4')
    for p in parts:
        if mounted(p['node']):raise RuntimeError(p['node']+' is mounted: '+repr(mounted(p['node']))+'; unmount it first')
    source_p1=source+'p1'
    if mounted(source_p1):raise RuntimeError('unmount original boot FAT before provisioning; it will be mounted read-only')
    return dict(device=device,bytes=size,serial=properties.get('ID_SERIAL_SHORT',properties.get('ID_SERIAL','unknown')),
                model=properties.get('ID_MODEL','unknown'),table=table,source_root=root,source_disk=source,
                source_p1=source_p1,source_uuid=blkid(root,'UUID'),source_partuuid=blkid(root,'PARTUUID'))
def token(plan):return hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()[:16].upper()
@contextlib.contextmanager
def mount_device(device,where,options):
    where=Path(where);where.mkdir(parents=True,exist_ok=True)
    if mounted(device):raise RuntimeError('device unexpectedly mounted: '+device)
    run('mount','-o',options,device,where)
    try:yield where
    finally:os.sync();run('umount',where)
def copy_verified(src,dst):
    atomic(dst,Path(src).read_bytes())
    if sha(src)!=sha(dst):raise RuntimeError('copy hash mismatch: '+str(dst))
def audit_references(root,plan,new_uuid,new_partuuid):
    report=[]
    paths=[root/'etc/fstab',root/'etc/crypttab',*sorted((root/'etc/systemd/system').glob('*.mount'))]
    for p in paths:
        if not p.is_file() or p.is_symlink():continue
        text=p.read_text()
        changed=text.replace('UUID='+plan['source_uuid'],'UUID='+new_uuid).replace('PARTUUID='+plan['source_partuuid'],'PARTUUID='+new_partuuid)
        if changed!=text:p.write_text(changed)
        refs=[x for x in changed.splitlines() if not x.lstrip().startswith('#') and re.search(r'(?:PART)?UUID=|LABEL=',x)]
        report.append(dict(file=str(p.relative_to(root)),references=refs,rewritten=changed!=text))
    return report
def install_runtime(root,config):
    import pwd
    prefix=root/'opt/ac920-onecard';prefix.mkdir(parents=True,exist_ok=True)
    for name in ('service.py','onecard_common.py'):shutil.copy2(HERE/name,prefix/name)
    write_json(root/'etc/ac920-onecard.json',config)
    shutil.copy2(HERE/'ac920-onecard.service',root/'etc/systemd/system/ac920-onecard.service')
    wants=root/'etc/systemd/system/multi-user.target.wants';wants.mkdir(parents=True,exist_ok=True)
    link=wants/'ac920-onecard.service'
    if not link.exists():link.symlink_to('../ac920-onecard.service')
    rules=root/'etc/udev/rules.d/99-ac920-onecard.rules';rules.parent.mkdir(parents=True,exist_ok=True)
    rules.write_text(f'SUBSYSTEM=="block", ENV{{ID_FS_UUID}}=="{config["boot_uuid"]}", ENV{{UDISKS_IGNORE}}="1"\n')
    account=pwd.getpwnam(config['user'])
    for folder in ('results','work'):
        p=root/'var/lib/ac920-onecard'/folder;p.mkdir(parents=True,exist_ok=True);os.chown(p,account.pw_uid,account.pw_gid)
    (root/'var/log/ac920-onecard').mkdir(parents=True,exist_ok=True)
def apply(plan,args):
    if platform.machine() not in ('aarch64','arm64'):raise RuntimeError('provision only on AC920 ARM64 Ubuntu')
    for command in ('rsync','parted','partprobe','udevadm','mkfs.ext4','sfdisk','blockdev','mount','umount'):
        if not shutil.which(command):raise RuntimeError('missing prerequisite: '+command)
    capture=Path(args.capture_boot).resolve(strict=True)
    if not boot_header(capture.read_bytes()):raise RuntimeError('capture BOOT header invalid')
    seal=json.loads((HERE/'../validation/release_identity.json').read_text())
    if sha(capture)!=seal['boot_sha256']:raise RuntimeError('capture BOOT differs from sealed pure-DP build')
    print('Original Ubuntu PMU MultiBoot readback:',read_multiboot())
    for p in (Path(args.recognizer)/'album/run_album.sh',Path(args.recognizer)/'run_page.py',Path(args.python),Path(args.app)):
        if not p.exists():raise RuntimeError('missing installed recognizer/runtime: '+str(p))
    if inspect(args.device)!=plan:raise RuntimeError('device identity changed since the plan')
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup=Path('/var/log/ac920-onecard-provision')/stamp;backup.mkdir(parents=True)
    (backup/'target_partition_table.sfdisk').write_text(run('sfdisk','--dump',plan['device']))
    source_table=run('sfdisk','--dump',plan['source_disk'])
    write_json(backup/'plan.json',plan)
    base=Path('/mnt/ac920-onecard-provision');base.mkdir(parents=True,exist_ok=True)
    p1=plan['device']+'1';p2=plan['device']+'2'
    owner={'schema':1,'source_uuid':plan['source_uuid'],'disk_bytes':plan['bytes'],'serial':plan['serial']}
    with mount_device(plan['source_p1'],base/'original-boot','ro') as original:
        for name in ('BOOT.BIN','Image','system.dtb','system.bit','boot.scr','uboot.env','uboot-redund.env'):
            if not (original/name).is_file():raise RuntimeError('original Ubuntu boot file missing: '+name)
        if not boot_header((original/'BOOT.BIN').read_bytes()):raise RuntimeError('original Ubuntu BOOT header invalid')
        # Validate the original script BEFORE any partition changes.
        rewrite((original/'boot.scr').read_bytes(),'deadbeef-02')
        for name in ('BOOT.BIN','boot.scr','uboot.env','uboot-redund.env'):shutil.copy2(original/name,backup/('original_'+name))
        if len(plan['table']['partitions'])==1:
            start=P1_OFFSET+P1_BYTES;end=start+P2_BYTES-1
            run('parted','--script',plan['device'],'unit','B','mkpart','primary','ext4',str(start)+'B',str(end)+'B')
            run('partprobe',plan['device']);run('udevadm','settle')
            # Check unchanged p1 and exact new p2 immediately before formatting.
            fresh=json.loads(run('sfdisk','--json',plan['device']))['partitiontable'];table=fresh['partitions']
            if fresh.get('id')!=plan['table'].get('id') or int(run('blockdev','--getsize64',plan['device']))!=plan['bytes'] or table[0]!=plan['table']['partitions'][0] or len(table)!=2 or table[1]['node']!=p2 or table[1]['start']*512!=start or table[1]['size']*512!=P2_BYTES:
                raise RuntimeError('partition postcondition failed; NOT formatting')
            if mounted(p2):raise RuntimeError('new p2 unexpectedly mounted; NOT formatting')
            run('mkfs.ext4','-L','AC920_ROOT','-E','nodiscard',p2)
            with mount_device(p2,base/'root','rw') as root:write_json(root/'.ac920-onecard-owned.json',owner)
        else:
            with mount_device(p2,base/'root','ro') as root:
                if json.loads((root/'.ac920-onecard-owned.json').read_text())!=owner:
                    raise RuntimeError('existing p2 is not owned by this exact migration; NOT formatting or overwriting')
        root_uuid=blkid(p2,'UUID');root_partuuid=blkid(p2,'PARTUUID')
        config=dict(schema=1,disk_bytes=plan['bytes'],root_partuuid=root_partuuid,
                    boot_partuuid=blkid(p1,'PARTUUID'),boot_uuid=blkid(p1,'UUID'),
                    capture_sha256=sha(capture),recognizer=args.recognizer,app=args.app,python=args.python,
                    user=args.user,rotate=args.rotate)
        with mount_device(p2,base/'root','rw') as root:
            command=['rsync','-aHAXx','--numeric-ids','--info=progress2']
            for name in ('/dev/*','/proc/*','/sys/*','/run/*','/tmp/*','/mnt/*','/media/*','/lost+found','/.ac920-onecard-owned.json'):
                command+=['--exclude='+name]
            command+=['/',str(root)+'/']
            result=subprocess.run(command)
            if result.returncode not in (0,24):raise RuntimeError('rootfs copy failed: '+str(result.returncode))
            audit=audit_references(root,plan,root_uuid,root_partuuid)
            install_runtime(root,config)
            write_json(root/'var/log/ac920-onecard-reference-audit.json',audit)
            for absolute in (args.recognizer+'/album/run_album.sh',args.app):
                if not (root/absolute.lstrip('/')).exists():raise RuntimeError('rootfs clone missing '+absolute)
        # Root filesystem is now synchronized and unmounted, before READY exists.
        with mount_device(p1,base/'boot','rw') as boot:
            one=boot/'OneCard';one.mkdir(exist_ok=True)
            if (boot/'Image').exists() and not (boot/'Image').is_dir():raise RuntimeError('target Image must remain photo folder; conflicting file')
            if (boot/'BOOT.BIN').is_file():copy_verified(boot/'BOOT.BIN',one/('PREVIOUS_'+stamp+'.BIN'))
            if (boot/'OneCard/READY.TXT').exists():(boot/'OneCard/READY.TXT').unlink()
            for name in ('Image','system.dtb','system.bit'):copy_verified(original/name,boot/'linux'/name)
            for name in ('uboot.env','uboot-redund.env'):copy_verified(original/name,boot/name)
            script,text=rewrite((original/'boot.scr').read_bytes(),root_partuuid)
            atomic(boot/'boot.scr',script);atomic(one/'boot_script.txt',text.encode())
            copy_verified(original/'BOOT.BIN',boot/'BOOT0001.BIN')
            copy_verified(capture,boot/'BOOT.BIN')
            for folder in ('Image','Grey','Meta','Outputs'):(boot/folder).mkdir(exist_ok=True)
            manifest=dict(config,source_root_uuid=plan['source_uuid'],source_card_retained=True,
                          linux_boot_sha256=sha(boot/'BOOT0001.BIN'),provisioned=stamp)
            write_json(one/'manifest.json',manifest)
            atomic(one/'READY.TXT',ready_bytes(boot/'BOOT0001.BIN',root_partuuid)) # Last commit
            write_json(backup/'result.json',manifest)
        if run('sfdisk','--dump',plan['source_disk'])!=source_table:raise RuntimeError('original card partition table unexpectedly changed')
    print('PREPARED: target safely unmounted; original Ubuntu card retained. Shut down, put big card in SD slot, boot capture, then S3.')
    print('Backup/audit: '+str(backup))
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device',required=True);ap.add_argument('--apply',action='store_true');ap.add_argument('--confirm')
    ap.add_argument('--capture-boot',default=str(HERE/'../BOOT.BIN'))
    ap.add_argument('--recognizer',default='/home/fpga/claude-album-20261029b')
    ap.add_argument('--app',default='/home/fpga/omr-transfer-20261024/sdcard_omr_ac920_20261024/app')
    ap.add_argument('--python',default='/home/fpga/ac920-omr-runtime/venv/bin/python')
    ap.add_argument('--user',default='fpga');ap.add_argument('--rotate',choices=('ccw','cw','180','none'),default='ccw')
    args=ap.parse_args()
    if os.geteuid()!=0:ap.error('use sudo even for the read-only plan')
    try:
        plan=inspect(args.device);print(json.dumps(plan,indent=2));confirmation=token(plan)
        print('PLAN: preserve p1 and photos; create/resume owned64GiB ext4 p2; clone root; relocate Linux files; install automatic album service.')
        print('CONFIRMATION='+confirmation)
        if not args.apply:return 0
        if args.confirm!=confirmation:raise RuntimeError('exact current confirmation token required for --apply')
        apply(plan,args);return 0
    except Exception as e:print('ERROR: '+str(e),file=sys.stderr);return 1
if __name__=='__main__':raise SystemExit(main())
