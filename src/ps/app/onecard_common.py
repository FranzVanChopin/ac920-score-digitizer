import hashlib,json,os,re,subprocess,tempfile,zlib
from pathlib import Path
EXPECTED_BYTES=512711720960
P1_BYTES=17179869184
P1_OFFSET=1048576
P2_BYTES=64*1024**3
MULTIBOOT=0xffca0010
def check_partitions(table,device):
    if table.get('sectorsize',512)!=512:raise RuntimeError('only 512-byte sectors accepted')
    parts=table['partitions']
    if len(parts) not in (1,2):raise RuntimeError('expected p1 and at most one owned p2')
    p=parts[0]
    if p['node']!=device+'1' or p['start']*512!=P1_OFFSET or p['size']*512!=P1_BYTES:
        raise RuntimeError('p1 geometry mismatch; never resize/format p1')
    if len(parts)==2:
        p=parts[1]
        if p['node']!=device+'2' or p['start']*512!=P1_OFFSET+P1_BYTES or p['size']*512!=P2_BYTES:
            raise RuntimeError('p2 geometry mismatch; never format unknown partitions')
    return parts
def run(*args):
    return subprocess.run([str(x) for x in args],check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout.strip()
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest().upper()
def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.onecard-',dir=path.parent)
    tmp=Path(name)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()
    if os.name!='nt':
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
def write_json(path,data):atomic(path,(json.dumps(data,indent=2,ensure_ascii=False)+'\n').encode())
def blkid(dev,key):return run('blkid','-s',key,'-o','value',dev)
def parent(dev):return '/dev/'+run('lsblk','-n','-o','PKNAME',dev)
def mounted(dev):
    result=subprocess.run(['findmnt','-rn','-S',str(dev),'-o','TARGET'],text=True,capture_output=True)
    if result.returncode not in (0,1):raise RuntimeError('findmnt failed: '+result.stderr)
    return result.stdout.strip().splitlines()
def boot_header(data):
    import struct
    if len(data)<76:return False
    words=struct.unpack_from('<11I',data,0x20)
    return words[:2]==(0xaa995566,0x584c4e58) and sum(words)&0xffffffff==0xffffffff
def ready_bytes(boot,root_partuuid):
    raw=Path(boot).read_bytes()
    if not boot_header(raw):raise ValueError('invalid Linux BOOT header')
    return (f'AC920_ONECARD_V1\ncapture=0\nlinux=1\nlinux_bytes={len(raw)}\n'
            f'linux_crc32={zlib.crc32(raw):08X}\nroot_partuuid={root_partuuid}\n').encode()
def runtime_card(config):
    root=Path(run('findmnt','-n','-o','SOURCE','/')).resolve()
    if blkid(root,'PARTUUID').lower()!=config['root_partuuid'].lower():raise RuntimeError('running root PARTUUID mismatch')
    p1=Path('/dev/disk/by-partuuid')/config['boot_partuuid']
    p1=p1.resolve(strict=True)
    disk=parent(root)
    if parent(p1)!=disk or int(run('blockdev','--getsize64',disk))!=config['disk_bytes']:raise RuntimeError('root/boot card identity mismatch')
    if blkid(p1,'TYPE')!='vfat' or blkid(p1,'LABEL')!='AC920_BOOT':raise RuntimeError('boot FAT identity mismatch')
    return str(p1)
def read_multiboot(path=Path('/sys/firmware/zynqmp/config_reg')):
    path.write_text('0xFFCA0010\n')
    values=re.findall(r'0x[0-9a-fA-F]+',path.read_text())
    if not values or int(values[-1],16)>0xfff:raise RuntimeError('unrecognized config_reg readback')
    return int(values[-1],16)
def select_boot(value,path=Path('/sys/firmware/zynqmp/config_reg')):
    if value not in (0,1):raise ValueError('boot slot')
    path.write_text(f'0xFFCA0010 0xFFF 0x{value:X}\n')
    if read_multiboot(path)!=value:raise RuntimeError('MultiBoot readback mismatch')
