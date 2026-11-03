"""Offline .bit/.hwh preflight. No pynq import, MMIO, download or board connection."""
import argparse,hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET

def inspect(bit,hwh,expected_device=None):
    bit,hwh=Path(bit),Path(hwh)
    if not bit.is_file() or not hwh.is_file():raise ValueError('Both .bit and .hwh must exist')
    if bit.suffix.lower()!='.bit' or hwh.suffix.lower()!='.hwh':raise ValueError('Expected .bit and .hwh files')
    if bit.stem!=hwh.stem or bit.resolve().parent!=hwh.resolve().parent:raise ValueError('PYNQ pair must share basename and directory')
    if bit.stat().st_size==0:raise ValueError('Empty bitstream')
    root=ET.fromstring(hwh.read_bytes())
    for e in root.iter():e.tag=e.tag.split('}')[-1].upper()
    info=next(root.iter('SYSTEMINFO'),None)
    device=info.get('DEVICE') if info is not None else None
    if expected_device and device!=expected_device:raise ValueError('HWH device differs from expected board device')
    ranges=[]
    for e in root.iter('MEMRANGE'):
        lo=int(e.get('BASEVALUE'),0);hi=int(e.get('HIGHVALUE'),0)
        if lo>hi:raise ValueError('Reversed address range')
        ranges.append(dict(instance=e.get('INSTANCE'),interface=e.get('SLAVEBUSINTERFACE'),base=hex(lo),high=hex(hi),kind=e.get('MEMTYPE')))
    if not ranges:raise ValueError('No memory ranges; inspect the metadata format before using this skill')
    return dict(offline_preflight_pass=True,device_from_hwh=device,artifact_sha256={'bit':hashlib.sha256(bit.read_bytes()).hexdigest(),'hwh':hashlib.sha256(hwh.read_bytes()).hexdigest()},address_ranges=ranges,hardware_tested=False,bitstream_device_parsed=False,pair_generation_proven=False,scope='File/name/metadata preflight only. Match hashes to a trusted build manifest before board use.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bit',required=True);p.add_argument('--hwh',required=True);p.add_argument('--expected-device');a=p.parse_args()
    try:r=inspect(a.bit,a.hwh,a.expected_device)
    except (OSError,ValueError,ET.ParseError) as e:print(json.dumps({'offline_preflight_pass':False,'error':str(e)}));raise SystemExit(1)
    print(json.dumps(r,ensure_ascii=False,indent=2))
