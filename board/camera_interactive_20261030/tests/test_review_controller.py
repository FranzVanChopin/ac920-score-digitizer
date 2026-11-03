import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'linux'))
if sys.platform=='win32':sys.modules.setdefault('pwd',types.ModuleType('pwd'))
import interactive_service as service
from panel_model import Panel
from notation_view import prepare
import xml.etree.ElementTree as ET

class ReviewController(unittest.TestCase):
    def test_close_waits_for_worker_then_unmounts_and_rejects_keys(self):
        controller=service.Controller({})
        controller.attached=True
        done=threading.Event();begun=threading.Event();calls=[]
        def worker():
            begun.set();controller.stopping.wait();done.set()
        controller.worker(worker);self.assertTrue(begun.wait(1))
        def run(*args):
            self.assertTrue(done.is_set());calls.append(args)
        with patch.object(service,'run',run),patch.object(service.os,'sync',lambda:None,create=True):
            controller.close()
        self.assertEqual(calls,[('umount',controller.mount)])
        self.assertFalse(controller.attached)
        controller.command('S3','short')
        self.assertFalse(controller.panel.return_pending)

    def test_orientation_cache_and_suspect_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            card=Path(folder)/'card';state=Path(folder)/'state';(card/'Grey').mkdir(parents=True);(state/'results').mkdir(parents=True)
            photo=card/'Grey/00-00.bmp';photo.write_bytes(b'picture')
            (state/'results/00.musicxml').write_text('<score-partwise/>')
            for rotation,expected in [('cw',False),('ccw',True)]:
                record=dict(status='ok',sha256=service.sha(photo),rotate_used=rotation,clef_ratio=0.,orientation_suspect=True)
                (state/'results/album.json').write_text(json.dumps(dict(pages={'00-00':record})))
                controller=service.Controller({'rotate':'ccw'},card,state)
                page=dict(id='00-00',status='pending',progress=0,stage='等待识别')
                song=dict(id='00',pages=[page],xml=True);controller.panel=Panel([song])
                controller.refresh_completed()
                self.assertEqual(song['xml'],expected)
                if expected:self.assertTrue(page['orientation_suspect']);self.assertEqual(page['clef_ratio'],0.)
                else:self.assertEqual(page['status'],'pending')

    def test_existing_clef_key_beams_preserved(self):
        data='<score-partwise><part id="p"><measure><attributes><divisions>4</divisions><staves>2</staves><key><fifths>4</fifths></key><clef number="1"><sign>F</sign><line>4</line></clef><clef number="2"><sign>G</sign><line>2</line></clef></attributes><note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration><voice>1</voice><staff>1</staff><beam number="1">begin</beam><beam number="2">begin</beam></note></measure></part></score-partwise>'
        after,changes=prepare(data);a,b=ET.fromstring(data),ET.fromstring(after)
        for tag in ('clef','key','beam'):
            self.assertEqual([ET.tostring(n) for n in a.iter(tag)],[ET.tostring(n) for n in b.iter(tag)])
        self.assertEqual(changes,dict(types=1,clefs=0))

    @unittest.skipIf(sys.platform=='win32','POSIX SIGTERM integration runs on the board')
    def test_actual_sigterm_runs_main_finally(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);config=folder/'config.json';config.write_text('{}')
            marker=folder/'marker'
            script="""
import sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1]);import interactive_service as s
marker=Path(sys.argv[2])
s.Controller.start=lambda self:marker.write_text('ready')
s.Controller.close=lambda self:marker.write_text('closed')
s.serve=lambda c:None
class Keys:
 def poll(self):return {}
 def close(self):pass
s.Keys=Keys
sys.argv=['test','--config',sys.argv[3]]
s.main()
"""
            process=subprocess.Popen([sys.executable,'-c',script,str(HERE/'linux'),str(marker),str(config)])
            try:
                deadline=time.monotonic()+5
                while not marker.exists() and time.monotonic()<deadline:time.sleep(.02)
                self.assertTrue(marker.exists())
                process.send_signal(signal.SIGTERM);self.assertEqual(process.wait(timeout=5),0)
                self.assertEqual(marker.read_text(),'closed')
            finally:
                if process.poll() is None:process.kill();process.wait()

if __name__=='__main__':unittest.main()
