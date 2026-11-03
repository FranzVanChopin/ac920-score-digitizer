import copy,gzip,json,struct,sys,tempfile,unittest,zlib
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(HERE/'linux'))
import onecard_common as common
import boot_script,provision_card,service
class TestBoot(unittest.TestCase):
    def image(self,compression=0,compressed=False):
        text=b'setenv bootargs root=/dev/mmcblk1p2 rw rootwait; saveenv; fatload mmc 1 ${k} Image; fatload mmc 1 ${d} system.dtb; fatload mmc 1 ${b} system.bit; booti ${k} - ${d}\n'
        payload=struct.pack('>II',len(text),0)+text
        if compressed:payload=gzip.compress(payload,mtime=0)
        h=[0x27051956,0,123,len(payload),0,0,zlib.crc32(payload),5,22,6,compression,b'boot']
        h[1]=zlib.crc32(struct.pack('>7I4B32s',*h))
        return struct.pack('>7I4B32s',*h)+payload
    def test_relocation(self):
        result,text=boot_script.rewrite(self.image(),'abcd1234-02')
        self.assertEqual(boot_script.unpack(result)[1],text+'\n')
        self.assertIn('root=PARTUUID=abcd1234-02',text)
        for name in ('Image','system.dtb','system.bit'):self.assertIn('linux/'+name,text)
        self.assertIn('booti ${k} - ${d}',text);self.assertIn('saveenv',text)
        for offset in (1,8,70,90):
            bad=bytearray(self.image());bad[offset]^=1
            with self.assertRaises(ValueError):boot_script.rewrite(bad,'abcd1234-02')
    def test_geometry_and_token(self):
        table={'sectorsize':512,'partitions':[dict(node='/dev/sda1',start=2048,size=33554432)]}
        self.assertEqual(len(common.check_partitions(table,'/dev/sda')),1)
        for field,value in (('start',0),('size',600000),('node','/dev/mmcblk0p1')):
            bad=copy.deepcopy(table);bad['partitions'][0][field]=value
            with self.assertRaises(RuntimeError):common.check_partitions(bad,'/dev/sda')
        table['partitions'].append(dict(node='/dev/sda2',start=(common.P1_OFFSET+common.P1_BYTES)//512,size=common.P2_BYTES//512))
        self.assertEqual(len(common.check_partitions(table,'/dev/sda')),2)
        for field,value in (('start',2048),('size',1024),('node','/dev/mmcblk0p2')):
            bad=copy.deepcopy(table);bad['partitions'][1][field]=value
            with self.assertRaises(RuntimeError):common.check_partitions(bad,'/dev/sda')
        table['partitions'].append(dict(node='/dev/sda3',start=999,size=999))
        with self.assertRaises(RuntimeError):common.check_partitions(table,'/dev/sda')
        self.assertNotEqual(provision_card.token({'bytes':512}),provision_card.token({'bytes':128}))
    def test_vendor_compression_flag(self):
        for compressed in (False,True):
            original=self.image(compression=1,compressed=compressed)
            result,text=boot_script.rewrite(original,'d3c4cd8d-02')
            self.assertEqual(boot_script.unpack(result)[1],text+'\n')
            self.assertEqual(result[31],1)
            self.assertEqual(result[64:66]==b'\x1f\x8b',compressed)
            self.assertIn('root=PARTUUID=d3c4cd8d-02',text)
    def test_register_readback(self):
        class Sysfs:
            def __init__(self):self.value=1;self.commands=[];self.reject=False
            def write_text(self,text):
                self.commands.append(text)
                if len(text.split())==3 and not self.reject:self.value=int(text.split()[2],16)
            def read_text(self):return f'0xFFCA0010 = 0x{self.value:08X}'
        p=Sysfs();self.assertEqual(common.read_multiboot(p),1)
        common.select_boot(0,p);self.assertEqual(p.value,0)
        p.reject=True
        with self.assertRaises(RuntimeError):common.select_boot(1,p)
    def test_atomic_and_header(self):
        with tempfile.TemporaryDirectory(dir=HERE/'validation') as d:
            p=Path(d)/'BOOT.BIN';b=bytearray(4096)
            struct.pack_into('<II',b,0x20,0xaa995566,0x584c4e58)
            struct.pack_into('<I',b,0x48,~(0xaa995566+0x584c4e58)&0xffffffff)
            common.atomic(p,b);self.assertTrue(common.boot_header(p.read_bytes()))
            ready=common.ready_bytes(p,'abcd1234-02');self.assertIn(b'capture=0\nlinux=1',ready)
            common.atomic(p,b+bytes(4));self.assertEqual(p.stat().st_size,4100)
class TestService(unittest.TestCase):
    def scenario(self,failure=None):
        with tempfile.TemporaryDirectory(dir=HERE/'validation') as d,ExitStack() as stack:
            root=Path(d);card=root/'card';state=root/'state';results=state/'results';results.mkdir(parents=True)
            for name in ('OneCard','Meta','Grey'):(card/name).mkdir(parents=True,exist_ok=True)
            (card/'BOOT.BIN').write_bytes(b'capture')
            (card/'Grey/00-00.bmp').write_bytes(b'pixels')
            (card/'Meta/00-00.json').write_text(json.dumps({'complete':True}))
            (card/'OneCard/REQUEST.JSON').write_text(json.dumps(dict(schema=1,action='recognize')))
            config=dict(root_partuuid='r',boot_partuuid='b',capture_sha256=common.sha(card/'BOOT.BIN'),
                        user='fpga',app='/app',python='/python',recognizer='/recognizer',rotate='ccw')
            (card/'OneCard/manifest.json').write_text(json.dumps(config))
            calls=[];is_mounted=False
            def run(*args):
                nonlocal is_mounted
                calls.append(args)
                if args[0]=='mount' and 'remount,rw' not in args:is_mounted=True
                if args[0]=='umount':
                    if failure=='umount':raise RuntimeError('busy')
                    is_mounted=False
                return ''
            def sub(args,**kw):
                calls.append(tuple(args))
                if args[0]=='runuser':
                    if failure=='recognizer':return type('P',(),{'returncode':1})()
                    record=dict(status='failed' if failure=='page' else 'ok',sha256=common.sha(card/'Grey/00-00.bmp'))
                    (results/'album.json').write_text(json.dumps({'pages':{'00-00':record}}))
                    if failure!='xml':(results/'00.musicxml').write_text('<score-partwise/>')
                    return type('P',(),{'returncode':0})()
                if failure=='reboot':raise RuntimeError('reboot denied')
                return type('P',(),{'returncode':0})()
            def select(value):
                calls.append(('select',value))
                if failure=='register' and value==0:raise RuntimeError('readback mismatch')
            for name,value in (('runtime_card',lambda c:'/dev/sda1'),('mounted',lambda dev:['mount'] if is_mounted else []),
                               ('run',run),('read_multiboot',lambda:1),('select_boot',select)):
                stack.enter_context(patch.object(service,name,value))
            if failure=='export':
                stack.enter_context(patch.object(service,'export_results',side_effect=RuntimeError('disk full')))
            stack.enter_context(patch.object(service.subprocess,'run',sub))
            stack.enter_context(patch.object(service.os,'geteuid',lambda:0,create=True))
            stack.enter_context(patch.object(service.os,'sync',lambda:None,create=True))
            if failure:
                with self.assertRaises(RuntimeError):service.execute(config,mount_dir=card,state_dir=state,log_dir=root/'logs')
            else:
                service.execute(config,mount_dir=card,state_dir=state,log_dir=root/'logs')
                self.assertTrue((card/'Outputs/00.musicxml').exists())
                self.assertFalse((card/'OneCard/REQUEST.JSON').exists())
                self.assertTrue((card/'OneCard/DONE.JSON').exists())
            commands=[x[0] for x in calls]
            if failure in ('recognizer','page','xml','umount','export'):self.assertNotIn('select',commands)
            if failure!='reboot':
                if failure:self.assertNotIn(('systemctl','reboot'),calls)
            if not failure:
                self.assertLess(commands.index('umount'),commands.index('select'))
                self.assertLess(calls.index(('select',0)),calls.index(('systemctl','reboot')))
            if failure=='reboot':self.assertIn(('select',1),calls)
    def test_success(self):self.scenario()
    def test_failures_never_reboot(self):
        for stage in ('recognizer','page','xml','export','umount','register','reboot'):
            with self.subTest(stage=stage):self.scenario(stage)
if __name__=='__main__':unittest.main(verbosity=2)
