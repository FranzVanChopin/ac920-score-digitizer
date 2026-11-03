"""Exercise the supplied album's real cache/direction loop with a fake inference service."""
import importlib.util,json,os,sys,tempfile,threading,types,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1]

class AlbumDirection(unittest.TestCase):
    def test_fixed_ccw_cache_invalidation_and_warning(self):
        try:
            import cv2
            import numpy as np
        except ImportError:self.skipTest('OpenCV integration runs on the board')
        source=Path(os.environ.get('AC920_ALBUM_SOURCE',str(HERE/'build/review_album')))
        sys.path.insert(0,str(source))
        spec=importlib.util.spec_from_file_location('album_under_review',source/'omr_album.py')
        album=importlib.util.module_from_spec(spec);spec.loader.exec_module(album)
        class Service:
            sent=0
            def __init__(self,*args):
                self.ready=threading.Event();self.ready.set();self.p=types.SimpleNamespace(poll=lambda:None)
            def send(self,photo,run,meter):
                Service.sent+=1;run.mkdir()
                (run/'page.musicxml').write_text('<score-partwise><part-list><score-part id="p"><part-name>Piano</part-name></score-part></part-list><part id="p"><measure number="1"><attributes><divisions>1</divisions><staves>2</staves><time><beats>4</beats><beat-type>4</beat-type></time></attributes><note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><staff>1</staff></note></measure></part></score-partwise>')
                (run/'headers.json').write_text(json.dumps([{'clefs':{},'key':0}]))
            def take(self,*args,**kw):return {'measures':1,'total_s':.01}
            def close(self):pass
        with tempfile.TemporaryDirectory() as folder,patch.object(album,'Service',Service):
            root=Path(folder);card=root/'card';out=root/'results';(card/'Grey').mkdir(parents=True);(card/'Meta').mkdir()
            cv2.imwrite(str(card/'Grey/00-00.bmp'),np.full((16,12),255,dtype=np.uint8))
            (card/'Meta/00-00.json').write_text('{"complete":true}')
            command=['--album',str(card),'--out',str(out)]
            album.main([*command,'--','fake'])
            first=json.loads((out/'album.json').read_text())['pages']['00-00']
            self.assertEqual(first['rotate_used'],'ccw');self.assertTrue(first['orientation_suspect'])
            self.assertEqual(first['clef_ratio'],0.);self.assertEqual(Service.sent,1)
            self.assertEqual(len(first['attempts']),1) # fixed mode must not try the opposite orientation.
            album.main([*command,'--','fake']);self.assertEqual(Service.sent,1)
            album.main([*command,'--rotate','cw','--','fake']);self.assertEqual(Service.sent,2)
            state=json.loads((out/'album.json').read_text())['pages']['00-00']
            self.assertEqual(state['rotate_used'],'cw');self.assertNotEqual(state['run'],first['run'])

if __name__=='__main__':unittest.main()
