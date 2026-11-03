import copy,importlib.util,io,json,os,sys,tempfile,threading,time,types,unittest,wave
from pathlib import Path
from unittest.mock import Mock,patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'app'))
if sys.platform=='win32':
    sys.modules.setdefault('pwd',types.ModuleType('pwd'))
    sys.modules.setdefault('fcntl',types.ModuleType('fcntl'))
import render_score as renderer
from score_assets import load_score,rate_ready
from interactive_service import Controller
from panel_model import Panel,Button

class Responsive(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.xml=self.root/'00.musicxml';self.xml.write_text('<score-partwise/>')
        self.font=self.root/'piano.sf2';self.font.write_bytes(b'font')
        self.directory=self.root/'cache';self.directory.mkdir()

    def layout(self,*_):
        (self.directory/'page-001.svg').write_text('<svg/>')
        (self.directory/'score.mid').write_bytes(b'midi')
        return dict(renderer_revision=3,xml_sha256=renderer.sha(self.xml),pages=['page-001.svg'],
            soundfont=dict(path=str(self.font),bytes=4),rates={},duration=1)

    def piano(self,midi,font,target):
        with wave.open(str(target),'wb') as out:
            out.setparams((2,2,48000,0,'NONE','none'));out.writeframes(b'\0'*192)
        return .001

    def render(self,**kwargs):
        with patch.object(renderer,'soundfont',return_value=self.font),patch.object(renderer,'build_layout',side_effect=self.layout) as layout,patch.object(renderer,'piano',side_effect=self.piano) as piano,patch.object(renderer,'rate_midi'),patch('sys.stdout',new_callable=io.StringIO):
            result=renderer.render(self.xml,self.directory,**kwargs)
            return result,layout.call_count,piano.call_count

    def test_first_open_generates_only_requested_rate(self):
        result,layout,audio=self.render(speeds=[1.])
        self.assertEqual((layout,audio),(1,1));self.assertEqual(list(result['rates']),['1'])
        self.assertTrue(rate_ready(self.directory,result,1))
        self.assertFalse(rate_ready(self.directory,result,.75))

    def test_warm_rate_skips_engraving_and_synthesis(self):
        self.render(speeds=[1.]);result,layout,audio=self.render(speeds=[1.])
        self.assertEqual((layout,audio),(0,0))
        result,layout,audio=self.render(speeds=[.75])
        self.assertEqual((layout,audio),(0,1));self.assertEqual(set(result['rates']),{'1','0.75'})

    def test_layout_is_committed_before_synthesis_and_survives_failure(self):
        def failure(*_):
            score=load_score(self.directory,renderer.sha(self.xml),self.font)
            self.assertIsNotNone(score);self.assertEqual(score['rates'],{})
            raise RuntimeError('synth failed')
        with patch.object(renderer,'soundfont',return_value=self.font),patch.object(renderer,'build_layout',side_effect=self.layout),patch.object(renderer,'piano',side_effect=failure),patch('sys.stdout',new_callable=io.StringIO):
            with self.assertRaises(RuntimeError):renderer.render(self.xml,self.directory,speeds=[1])
        self.assertEqual(json.loads((self.directory/'score.json').read_text())['rates'],{})

    def test_missing_wav_repairs_only_that_rate(self):
        self.render(speeds=[1.]);(self.directory/'score.wav').unlink()
        _,layout,audio=self.render(speeds=[1.]);self.assertEqual((layout,audio),(0,1))

    def test_layout_only_does_not_synthesize(self):
        score,layout,audio=self.render(layout_only=True)
        self.assertEqual((layout,audio),(1,0));self.assertEqual(score['rates'],{})

    def test_cache_rejects_changed_xml_missing_page_and_truncated_audio(self):
        score,_,_=self.render(speeds=[1.]);(self.directory/'score.wav').write_bytes(b'RIFF')
        self.assertFalse(rate_ready(self.directory,score,1))
        self.assertIsNone(load_score(self.directory,'different hash'))
        (self.directory/'page-001.svg').unlink();self.assertIsNone(load_score(self.directory,renderer.sha(self.xml)))

    def test_controller_warm_open_never_starts_renderer(self):
        self.render(speeds=[1.]);album=self.root/'card';(album/'Outputs').mkdir(parents=True)
        (album/'Outputs/00.musicxml').write_bytes(self.xml.read_bytes());base=self.root/'base'
        destination=base/'score-cache'/('00_'+renderer.sha(self.xml)[:12].upper()+'_r3');destination.parent.mkdir(parents=True)
        self.directory.rename(destination)
        c=Controller(dict(user='test',piano_soundfont=str(self.font)),album,base)
        with patch('interactive_service.pwd.getpwnam',create=True,return_value=types.SimpleNamespace(pw_uid=0,pw_gid=0)),patch('interactive_service.os.chown',create=True),patch.object(c,'worker'),patch.object(c,'process',side_effect=AssertionError('renderer should not start')):
            c.prepare_score(dict(id='00',pages=[]))
        self.assertEqual(c.play['state'],'ready');self.assertFalse(c.panel.busy)

    def test_state_wait_wakes_on_command_and_snapshot_is_detached(self):
        c=Controller({});c.panel=Panel([dict(id='00',pages=[dict(id='00-00')],xml=False),dict(id='01',pages=[],xml=False)])
        result=[];ready=threading.Event()
        def wait():ready.set();result.append(c.wait_snapshot(c.revision))
        t=threading.Thread(target=wait);t.start();ready.wait();time.sleep(.02)
        begin=time.monotonic();c.command('S2','short');t.join(.3)
        self.assertFalse(t.is_alive());self.assertLess(time.monotonic()-begin,.3)
        self.assertEqual(result[0]['revision'],c.revision)
        c.panel.songs[0]['pages'][0]['id']='changed'
        self.assertEqual(result[0]['songs'][0]['pages'][0]['id'],'00-00')

    def test_long_press_and_batched_short_presses_are_classified_once(self):
        b=Button(debounce_ms=0);b.update(False,0)
        self.assertIsNone(b.update(True,10));self.assertEqual(b.update(False,1610),'long')
        self.assertIsNone(b.update(False,1620))
        actions=[]
        for down,now in ((True,1700),(False,1760),(True,1800),(False,1860)):
            event=b.update(down,now)
            if event:actions.append(event)
        self.assertEqual(actions,['short','short'])

    def test_foreground_action_cancels_only_owned_prefetch(self):
        c=Controller({});process=Mock(pid=43210);process.poll.return_value=None;c.prefetch_process=process
        with patch('interactive_service.os.killpg',create=True) as kill:c.command('S2','short')
        kill.assert_called_once_with(43210,__import__('signal').SIGTERM)

    def test_actual_engraving_and_midi_are_unchanged(self):
        original=HERE.parent/'dp_restore_20261031/storage_baseline/ac920-panel-ui1/render_score.py'
        if not original.exists():original=HERE/'old_render_score.py'
        spec=importlib.util.spec_from_file_location('old_renderer',original);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        fixture=HERE/'fixtures/playback_reference.musicxml'
        old_dir=self.root/'old';new_dir=self.root/'new'
        with patch.object(old,'soundfont',return_value=self.font),patch.object(old,'piano',side_effect=self.piano),patch.object(renderer,'soundfont',return_value=self.font),patch.object(renderer,'piano',side_effect=self.piano),patch('sys.stdout',new_callable=io.StringIO):
            before=old.render(fixture,old_dir);after=renderer.render(fixture,new_dir,speeds=[1])
        self.assertEqual(old.midi_notes(old_dir/'score.mid'),renderer.midi_notes(new_dir/'score.mid'))
        self.assertEqual(before['notes'],after['notes']);self.assertEqual(before['duration'],after['duration'])
        self.assertEqual(len(before['pages']),len(after['pages']))
        self.assertEqual([(x['ms'],x['page'],x['number']) for x in before['measures']],[(x['ms'],x['page'],x['number']) for x in after['measures']])

if __name__=='__main__':unittest.main(verbosity=2)
