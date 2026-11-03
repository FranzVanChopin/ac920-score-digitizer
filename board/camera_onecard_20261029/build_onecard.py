"""Build isolated DP app against the exact validated camera BSP; no XSCT changes."""
from pathlib import Path
import hashlib,json,os,re,shutil,subprocess,zipfile

HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('CAMERA_WORKSPACE',str(HERE.parents[1])))
SRC=HERE/'software/camera_capture'
BUILD=HERE/'build'
VAL=HERE/'validation'
BASE=HERE
PLATFORM=HERE/'build/camera_capture_vitis/camera_capture_platform'
BSP=Path(os.environ.get('CAMERA_BSP_DIR',str(PLATFORM/'psu_cortexa53_0/camera_standalone/bsp/psu_cortexa53_0')))
VITIS=Path(os.environ.get('VITIS_ROOT','D:/Xlinix/Vitis/2024.2'))
GCC=VITIS/'gnu/aarch64/nt/aarch64-none/bin/aarch64-none-elf-gcc.exe'
BOOTGEN=VITIS/'bin/bootgen.bat'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def run(args):
    invocation=[str(x) for x in args]
    if Path(invocation[0]).suffix.lower()=='.bat':
        invocation=[os.environ.get('COMSPEC','cmd.exe'),'/d','/c',*invocation]
    p=subprocess.run(invocation,capture_output=True,text=True,errors='replace')
    log.write('$ '+' '.join(str(x) for x in args)+'\n'+p.stdout+p.stderr+'\n');log.flush()
    if p.returncode: raise RuntimeError(f'Exit {p.returncode}: '+p.stdout+p.stderr)
    return p.stdout

BUILD.mkdir(exist_ok=True);VAL.mkdir(exist_ok=True)
with (VAL/'build.log').open('w',encoding='utf-8') as log:
    xsa=HERE/'build/vivado_camera/camera_capture.xsa'
    bit=HERE/'build/vivado_camera/camera_capture.runs/impl_1/camera_capture_wrapper.bit'
    marker=(HERE/'build/vivado_camera/hardware.complete').read_text()
    assert sha(xsa) in marker and sha(bit) in marker
    assert sha(BSP.parents[3]/'hw/camera_capture.xsa')==sha(xsa)
    objects=[]
    for p in sorted(SRC.glob('*.c')):
        obj=BUILD/(p.stem+'.o')
        print('COMPILE '+p.name,flush=True)
        run([GCC,'-mcpu=cortex-a53','-std=c11','-O2','-g3','-Wall','-Wextra','-Werror',
             '-I'+str(SRC),'-I'+str(BSP/'include'),'-c',p,'-o',obj])
        objects.append(obj)
    elf=BUILD/'camera_capture.elf'
    run([GCC,'-mcpu=cortex-a53','-o',elf,*objects,'-Wl,-T,'+str(SRC/'lscript.ld'),
         '-Wl,-Map,'+str(BUILD/'camera_capture.map'),'-L'+str(BSP/'lib'),
         '-Wl,--start-group','-lxilffs','-lxil','-lgcc','-lc','-lm','-Wl,--end-group'])
    for name in ('fsbl.elf','pmufw.elf'): shutil.copy2(HERE/'build/camera_capture_vitis/artifacts'/name,BUILD/name)
    bif='''the_ROM_image:
{
  [bootloader, destination_cpu=a53-0] "%s"
  [pmufw_image] "%s"
  [destination_device=pl] "%s"
  [destination_cpu=a53-0, exception_level=el-3, trustzone=secure] "%s"
}
'''%tuple(p.as_posix() for p in (BUILD/'fsbl.elf',BUILD/'pmufw.elf',bit,elf))
    (BUILD/'camera_capture.bif').write_text(bif,encoding='utf-8')
    boot=BUILD/'BOOT.BIN'
    run([BOOTGEN,'-arch','zynqmp','-image',BUILD/'camera_capture.bif','-w','on','-o',boot])
    readback=run([BOOTGEN,'-arch','zynqmp','-read',boot])
    (VAL/'bootgen_readback.txt').write_text(readback,encoding='utf-8')
    assert 'camera_capture_wrapper.bit' in readback and 'camera_capture.elf' in readback
    assert 'fsbl.elf' in readback and 'dest_device [PL]' in readback
    bins=GCC.parent
    headers=run([bins/'aarch64-none-elf-readelf.exe','-lW',elf])
    (VAL/'elf_headers.txt').write_text(headers,encoding='utf-8')
    assert re.search(r'Entry point 0x100000',headers)
    size=run([bins/'aarch64-none-elf-size.exe',elf])
    symbols=run([bins/'aarch64-none-elf-nm.exe','-n',elf])
    assert '_startup' in symbols and 'capture_dp_present' in symbols
    shutil.copy2(boot,HERE/'BOOT.BIN')
    (VAL/'release_identity.json').write_text(json.dumps(dict(version='a4_onecard_dp2',boot_sha256=sha(boot),boot_bytes=boot.stat().st_size),indent=2))
    sources=[dict(file=p.name,sha256=sha(p)) for p in sorted(SRC.iterdir()) if p.suffix in ('.c','.h','.ld')]
    validation=dict(version='a4_onecard_dp2',boot_bytes=boot.stat().st_size,boot_sha256=sha(boot),
                    xsa_sha256=sha(xsa),bitstream_sha256=sha(bit),sources=sources,application_size=size,
                    build='PASS: A53 -Wall -Wextra -Werror',physical_board_test='pending',
                    display='PS DP 640x480 RGB8, polled VSYNC, double RGBA buffers',
                    capture='native Image/Grey/Meta retained; pure DP; S1 unchanged, S2 focus, S3 safe Ubuntu handoff')
    (VAL/'build_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
print(json.dumps({k:validation[k] for k in ('version','boot_bytes','boot_sha256','build')}),flush=True)
