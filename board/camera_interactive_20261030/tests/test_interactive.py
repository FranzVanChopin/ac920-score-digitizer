import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
import types
from pathlib import Path
from unittest.mock import patch

if sys.platform == 'win32':
    sys.modules.setdefault('pwd', types.ModuleType('pwd'))

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'linux'))
from panel_model import Button, Panel, catalogue
from prepare_dtb import transform, span
from interactive_service import Controller
from patch_progress import make
from notation_view import prepare as prepare_notation
import xml.etree.ElementTree as ET


class Interactive(unittest.TestCase):
    def songs(self):
        return [dict(id='00',pages=[dict(id='00-00',status='pending',progress=0,stage='')],xml=True),
                dict(id='01',pages=[dict(id='01-00',status='pending',progress=0,stage='')],xml=False)]

    def test_no_automatic_actions(self):
        p=Panel(self.songs())
        self.assertFalse(p.busy); self.assertEqual(p.current()['id'],'00')
        self.assertIsNone(p.action('S1','short'))
        self.assertEqual(p.action('S2','short'),'stop');self.assertEqual(p.current()['id'],'01')
        self.assertEqual(p.action('S2','long'),'recognize')
        p.busy=True
        self.assertIsNone(p.action('S2','short'));self.assertEqual(p.current()['id'],'01')
        self.assertEqual(p.action('S3','short'),'defer_return')
        self.assertTrue(p.return_pending)

    def test_library_only_xml(self):
        p=Panel(self.songs())
        self.assertEqual(p.action('S1','long'),'stop')
        self.assertEqual(p.view,'library');self.assertEqual(len(p.items()),1)
        p.action('S2','short');self.assertEqual(p.current()['id'],'00')
        self.assertEqual(p.action('S2','long'),'play')
        p.action('S1','long');self.assertEqual(p.view,'recognize')
        self.assertEqual(p.action('S3','short'),'return')

    def test_hold_does_not_send_short_and_boot_hold_ignored(self):
        b=Button();self.assertIsNone(b.update(False,0))
        b.update(True,100);b.update(True,140)
        self.assertEqual(b.update(True,1640),'long')
        self.assertIsNone(b.update(True,1900))
        b.update(False,2000);self.assertIsNone(b.update(False,2040))
        b.update(True,2200);b.update(True,2240);b.update(False,2400)
        self.assertEqual(b.update(False,2440),'short')
        b=Button();b.update(True,0)
        self.assertIsNone(b.update(True,4000));b.update(False,4100)
        self.assertIsNone(b.update(False,4140))

    def test_meta_commit_filters_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            album=Path(tmp);(album/'Meta').mkdir();(album/'Grey').mkdir()
            for key,complete in [('00-00',True),('00-01',False),('01-00',True)]:
                (album/'Meta'/(key+'.json')).write_text(json.dumps(dict(complete=complete)))
                (album/'Grey'/(key+'.bmp')).write_bytes(b'grey')
            songs=catalogue(album,album/'results')
            self.assertEqual([s['id'] for s in songs],['00','01'])
            self.assertEqual([p['id'] for p in songs[0]['pages']],['00-00'])

    def test_progress_is_page_scoped(self):
        c=Controller({});song=self.songs()[0];c.panel=Panel([song])
        c.progress(song,'[UI_STAGE]'+json.dumps(dict(photo='/work/01-00.bmp',ui_stage='write')))
        self.assertEqual(song['pages'][0]['progress'],0)
        c.progress(song,'[UI_STAGE]'+json.dumps(dict(photo='/work/00-00.bmp',ui_stage='relation')))
        self.assertEqual(song['pages'][0]['progress'],55)
        c.progress(song,'12:00:00 00-00: 15 measures, 40 notes, 20 s -> 00.musicxml')
        self.assertEqual(song['pages'][0]['status'],'ok')
        self.assertFalse(c.panel.return_pending)

    def test_dt_disables_parent_and_enables_audio(self):
        original=Path('outputs/sd_backups/20261030_before_interactive/codex-interactive-original-20261030/running.dts').read_text()
        result=transform(original)
        for name in ('pl-bus','keys','vcap_mipi_csi_v_proc_ss_0'):
            first,last=span(result,name)
            self.assertIn('status = "disabled";',result[first:result.index('{',first)+60])
        first,last=span(result,'zynqmp-dp-snd-codec0')
        self.assertIn('status = "okay";',result[first:last])
        self.assertIn('gpios = <0x701 13 1>',result)
        self.assertIn('clocks = <0x04 0x47>',result[result.index('ac920-panel-gpio'):])

    def test_no_sensor_algorithm_change(self):
        old=HERE.parent/'camera_onecard_20261029/software/camera_capture'
        new=HERE/'software/camera_capture'
        for path in old.iterdir():
            if path.suffix not in ('.c','.h','.ld') or path.name in ('main.c','capture_dp.c','capture_dp.h','onecard_platform.h'):
                continue
            self.assertEqual(path.read_bytes(),(new/path.name).read_bytes(),path.name)

    def test_compact_xml_engraving_preserves_pitch_duration(self):
        value='''<score-partwise><part-list/><part id="P1"><measure number="1"><attributes>
        <divisions>480</divisions><staves>2</staves></attributes>
        <note><pitch><step>B</step><octave>4</octave></pitch><duration>120</duration><voice>1</voice><staff>1</staff></note>
        <note><rest/><duration>360</duration><staff>2</staff></note>
        <backup><duration>480</duration></backup>
        <note><pitch><step>C</step><octave>3</octave></pitch><duration>80</duration><voice>2</voice><staff>2</staff></note>
        </measure></part></score-partwise>'''
        text,changes=prepare_notation(value)
        root=ET.fromstring(text); notes=root.findall('.//note')
        self.assertEqual(changes,dict(types=3,clefs=2))
        self.assertEqual([n.findtext('duration') for n in notes],['120','360','80'])
        self.assertEqual([n.findtext('type') for n in notes],['16th','eighth','16th'])
        self.assertIsNotNone(notes[1].find('dot'))
        self.assertEqual(notes[2].findtext('time-modification/actual-notes'),'3')
        self.assertEqual(notes[2].findtext('time-modification/normal-notes'),'2')
        self.assertEqual(root.findtext('.//backup/duration'),'480')
        self.assertEqual([n.findtext('pitch/step') for n in notes],['B',None,'C'])
        self.assertEqual(prepare_notation(text)[1],dict(types=0,clefs=0))


if __name__=='__main__':unittest.main()
