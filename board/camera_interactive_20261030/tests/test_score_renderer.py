import json,sys,tempfile,unittest,wave
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'linux'))
import render_score as renderer
import mido

class Renderer(unittest.TestCase):
    def test_midi_velocities_and_tempo_scaling_preserve_notes(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'score.mid';file=mido.MidiFile(ticks_per_beat=480);track=mido.MidiTrack();file.tracks.append(track)
            track.extend([mido.MetaMessage('set_tempo',tempo=500000),mido.MetaMessage('time_signature',numerator=4,denominator=4),
                mido.Message('note_on',note=48,velocity=90),mido.Message('note_on',note=76,velocity=90),
                mido.Message('note_off',note=48,time=480),mido.Message('note_off',note=76),
                mido.Message('note_on',note=76,velocity=90),mido.Message('note_off',note=76,time=480)])
            file.save(source);before,duration=renderer.midi_notes(source)
            stats=renderer.dynamics(source);after,_=renderer.midi_notes(source)
            self.assertEqual([n[:3] for n in before],[n[:3] for n in after])
            self.assertEqual(stats['values'],[81,98,103])
            for speed in renderer.RATES:
                target=Path(folder)/str(speed);renderer.rate_midi(source,target,speed)
                notes,seconds=renderer.midi_notes(target)
                self.assertAlmostEqual(seconds,duration/speed,places=5)
                self.assertEqual([n[2:] for n in notes],[n[2:] for n in after])

    def test_real_verovio_measure_repeat_timeline_and_geometry(self):
        import verovio
        toolkit=verovio.toolkit();toolkit.setOptions(dict(svgViewBox=True,svgBoundingBoxes=True,svgAdditionalAttribute=['measure@n']))
        data='<score-partwise version="4.0"><part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list><part id="P1"><measure number="1"><attributes><divisions>1</divisions><time><beats>1</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes><barline location="left"><repeat direction="forward"/></barline><note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note></measure><measure number="2"><note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note><barline location="right"><repeat direction="backward"/></barline></measure></part></score-partwise>'
        self.assertTrue(toolkit.loadData(data));toolkit.renderToMIDI()
        svg,view,boxes=renderer.page_geometry(toolkit.renderToSVG(1))
        self.assertNotIn('bounding-box',svg);self.assertEqual(len(boxes),2)
        for box in boxes.values():
            x,y,w,h=box['bbox'];self.assertGreater(w,0);self.assertGreater(h,0);self.assertLess(x+w,view[2]+5)
        mapping=toolkit.renderToTimemap(dict(includeMeasures=True,includeRests=True))
        if isinstance(mapping,str):mapping=json.loads(mapping)
        measures,turns=renderer.timeline(toolkit,mapping,{k:dict(page=1,**v) for k,v in boxes.items()})
        self.assertEqual([m['number'] for m in measures],['1','2','1','2'])
        self.assertEqual([m['ms'] for m in measures],[0,500,1000,1500])
        self.assertEqual(turns,[dict(ms=0,page=1)])

if __name__=='__main__':unittest.main()
