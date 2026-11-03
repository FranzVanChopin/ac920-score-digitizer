"""Validate and precisely relocate the supplied U-Boot legacy script image."""
import gzip,re,struct,zlib
MAGIC=0x27051956
def unpack(image):
    if len(image)<72:raise ValueError('short boot.scr')
    h=list(struct.unpack('>7I4B32s',image[:64]))
    if h[0]!=MAGIC or h[9]!=6 or h[10] not in (0,1):raise ValueError('expected legacy SCRIPT image (none/gzip)')
    header=bytearray(image[:64]);header[4:8]=b'\0'*4
    if zlib.crc32(header)!=h[1]:raise ValueError('boot.scr header CRC')
    payload=image[64:]
    if len(payload)!=h[3] or zlib.crc32(payload)!=h[6]:raise ValueError('boot.scr payload CRC/length')
    # The AC920 vendor image sets IH_COMP_GZIP but stores a plain script table.
    # U-Boot accepts it. Preserve that working header flag and actual encoding.
    if h[10]==1 and payload.startswith(b'\x1f\x8b'):payload=gzip.decompress(payload)
    size,terminator=struct.unpack('>II',payload[:8])
    if terminator or len(payload)!=size+8:raise ValueError('expected one script component')
    return h,payload[8:].decode('utf-8').rstrip('\0')
def rewrite(image,partuuid):
    if not re.fullmatch(r'[0-9a-fA-F-]{8,40}',partuuid):raise ValueError('invalid PARTUUID')
    header,text=unpack(image)
    if not re.search(r'fatload\s+mmc\s+1(?:[:\s])',text):raise ValueError('script does not use expected SD1')
    text,count=re.subn(r'root=/dev/mmcblk1p2\b','root=PARTUUID='+partuuid,text)
    if count!=1:raise ValueError('expected exactly one known root argument')
    for name in ('Image','system.dtb','system.bit'):
        pattern=r'(?<![\w/])'+re.escape(name)+r'(?![\w/.])'
        text,count=re.subn(pattern,'linux/'+name,text)
        if count<1:raise ValueError('expected load of '+name)
    # No other boot commands, addresses, clocks or saveenv behavior are changed.
    script=text.encode()+b'\n'
    payload=struct.pack('>II',len(script),0)+script
    if header[10]==1 and image[64:66]==b'\x1f\x8b':payload=gzip.compress(payload,mtime=0)
    header[1]=0;header[3]=len(payload);header[6]=zlib.crc32(payload)
    raw=struct.pack('>7I4B32s',*header)
    header[1]=zlib.crc32(raw)
    result=struct.pack('>7I4B32s',*header)+payload
    assert unpack(result)[1]==text+'\n'
    return result,text
