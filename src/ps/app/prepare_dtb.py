"""Build Linux keys/audio DT for the exact DP2 PL bitstream, preserving PS setup."""
import re
import subprocess
from pathlib import Path


def span(text, name):
    match = re.search(r'(?m)^\s*' + re.escape(name) + r'\s*\{', text)
    if not match: raise ValueError('Missing DT node ' + name)
    start = text.index('{', match.start())
    depth = 1; at = start + 1; quoted = False; escape = False
    while depth:
        c = text[at]
        if escape: escape = False
        elif c == '\\': escape = True
        elif c == '"': quoted = not quoted
        elif not quoted:
            if c == '{': depth += 1
            elif c == '}': depth -= 1
        at += 1
    return match.start(), at+1


def status(text, name, value):
    start, end = span(text, name)
    part = text[start:end]
    child = re.search(r'(?m)^\s*[\w,@-]+\s*\{', part[part.index('{')+1:])
    boundary = part.index('{') + 1 + child.start() if child else len(part)
    head, tail = part[:boundary], part[boundary:]
    if re.search(r'\bstatus = "[^"]+";', head):
        head = re.sub(r'\bstatus = "[^"]+";', 'status = "' + value + '";', head, count=1)
    else:
        at = head.index('{') + 1
        head = head[:at] + '\n\t\tstatus = "' + value + '";' + head[at:]
    part = head + tail
    return text[:start] + part + text[end:]


def transform(text):
    # Leave old PL phandles resolvable but prevent any stale driver probing old MMIO.
    text = status(text, 'pl-bus', 'disabled')
    text = status(text, 'vcap_mipi_csi_v_proc_ss_0', 'disabled')
    text = status(text, 'keys', 'disabled')
    text = status(text, 'zynqmp-dp-snd-codec0', 'okay')
    start, end = span(text, 'gpio@ff0a0000')
    phandle = re.search(r'phandle = <(0x[0-9a-f]+)>;', text[start:end])[1]
    start, end = span(text, 'gpio@a0080000')
    # Keep the REAL PS PL0 clock referenced. A fixed-clock placeholder allows
    # clk_disable_unused() to gate the bus while gpio-keys is polling it.
    clock_provider = re.search(r'clocks = <([^>]+)>;', text[start:end])[1].split()[0]
    # The old factory GPIO was on PL3; the DP2 capture XSA uses PS pl_clk0.
    # xlnx-zynqmp-clk.h: PL0_REF = 71 (0x47).
    pl_clock = clock_provider + ' 0x47'
    if 'phandle = <0x700>' in text: raise ValueError('panel phandle collision')
    # Compact gpio-chip offset = channel1 width(5) + channel2 bit(8 or 9).
    panel = '''
    ac920-panel-gpio@80050000 {
        compatible = "xlnx,xps-gpio-1.00.a";
        reg = <0 0x80050000 0 0x1000>;
        gpio-controller; #gpio-cells = <2>;
        xlnx,is-dual = <1>; xlnx,gpio-width = <5>; xlnx,gpio2-width = <10>;
        xlnx,tri-default = <0>; xlnx,tri-default-2 = <0x3ff>;
        xlnx,dout-default = <0>; xlnx,dout-default-2 = <0>;
        clocks = <%s>; clock-names = "s_axi_aclk";
        phandle = <0x701>;
    };
    ac920-panel {
        compatible = "gpio-keys-polled"; label = "ac920-panel"; poll-interval = <10>;
        s1 { label = "S1"; linux,code = <183>; gpios = <%s 44 1>; debounce-interval = <20>; };
        s2 { label = "S2"; linux,code = <184>; gpios = <0x701 13 1>; debounce-interval = <20>; };
        s3 { label = "S3"; linux,code = <185>; gpios = <0x701 14 1>; debounce-interval = <20>; };
    };
''' % (pl_clock, phandle)
    at = text.rfind('};')
    text = text[:at] + panel + text[at:]
    return text


def make(source, destination):
    source, destination = Path(source), Path(destination)
    text = subprocess.run(['dtc', '-I', 'dtb', '-O', 'dts', str(source)], check=True,
                          text=True, capture_output=True).stdout
    result = transform(text)
    destination.with_suffix('.dts').write_text(result)
    process = subprocess.run(['dtc', '-I', 'dts', '-O', 'dtb', '-o', str(destination),
                              str(destination.with_suffix('.dts'))], capture_output=True, text=True)
    destination.with_suffix('.warnings.txt').write_text(process.stderr)
    process.check_returncode()
    return destination
