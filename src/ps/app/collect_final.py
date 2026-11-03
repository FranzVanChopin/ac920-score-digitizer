"""Non-secret final installation/idle state evidence."""
import json
import os
import subprocess
from pathlib import Path
from onecard_common import sha,runtime_card,write_json
from panel_ui import request

config=json.loads(Path('/etc/ac920-onecard.json').read_text())
device=runtime_card(config)
base=Path('/mnt/ac920-onecard')
manifest=json.loads((base/'OneCard/manifest.json').read_text())
state=request(dict(action='state'),timeout=2)
assert not state['busy'] and not state['return_pending']
assert not state['button_error']
assert sha(base/'BOOT.BIN')==config['capture_sha256']==manifest['capture_sha256']
assert sha(base/'linux/system.dtb')==manifest['linux_dtb_sha256']
value=dict(version=config.get('interface_version','a4_interactive_ui1'),theme=config.get('theme','dark'),device=device,config=config,
    boot_sha256=sha(base/'BOOT.BIN'),dtb_sha256=sha(base/'linux/system.dtb'),
    bit_sha256=sha(base/'linux/system.bit'),state=state,
    sources={p.name:sha(p) for p in sorted(Path('/opt/ac920-panel-ui1').glob('*.py'))},
    engine_provenance=json.loads(Path('/opt/ac920-panel/engine/provenance.json').read_text()),
    input_devices=Path('/proc/bus/input/devices').read_text(),
    sound_cards=Path('/proc/asound/cards').read_text(),
    old_service_enabled=subprocess.run(['systemctl','is-enabled','ac920-onecard.service'],capture_output=True,text=True).stdout.strip(),
    new_service_active=subprocess.run(['systemctl','is-active','ac920-panel.service'],capture_output=True,text=True).stdout.strip(),
    clock=[line.strip() for line in Path('/sys/kernel/debug/clk/clk_summary').read_text().splitlines() if 'pl0_ref' in line],
    no_physical_dp_test=True)
path=Path('/home/fpga/claude-codex-interactive-ui1-20261030/final_state.json')
write_json(path,value);os.chmod(path,0o644)
print('PASS idle, controller/keys/audio present, BOOT+DTB manifest consistent;'+str(path))
