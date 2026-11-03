"""Headless behavior checks: physical actions, shared timing, WAV selection and UI states."""
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

LINUX = Path(__file__).resolve().parents[1] / 'linux'
sys.path.insert(0, str(LINUX))
from panel_model import Panel, ORIENTATION_WARNING, playback_clock, current_measure


def load_ui():
    gi = types.ModuleType('gi')
    gi.require_foreign = gi.require_version = lambda *a: None
    repository = types.ModuleType('gi.repository')
    repository.Gtk = types.SimpleNamespace(Window=object)
    repository.Rsvg = types.SimpleNamespace(Rectangle=type('Rectangle', (), {}))
    for name in ('GLib', 'Gdk', 'Pango', 'PangoCairo'):
        setattr(repository, name, types.SimpleNamespace())
    with patch.dict(sys.modules, {'gi': gi, 'gi.repository': repository}):
        spec = importlib.util.spec_from_file_location('panel_ui_headless_test', LINUX / 'panel_ui.py')
        ui = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ui)
    return ui


class Cairo:
    def __getattr__(self, _): return lambda *a: None


class PlaybackPanel(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.ui = load_ui()

    def test_speed_requires_library_and_never_starts_recognition_or_return(self):
        panel = Panel([dict(id='00', pages=[dict(id='00-00')], xml=True)])
        self.assertIsNone(panel.action('S1', 'short'))
        self.assertEqual(panel.speed, 1.)
        self.assertEqual(panel.action('S1', 'long'), 'stop')
        seen = []
        for _ in range(4):
            self.assertEqual(panel.action('S1', 'short'), 'speed')
            seen.append(panel.speed)
        self.assertEqual(seen, [1.25, 1.5, .75, 1.])
        self.assertFalse(panel.return_pending)
        self.assertFalse(panel.busy)
        panel.busy = True
        self.assertIsNone(panel.action('S1', 'short'))
        self.assertEqual(panel.speed, 1.)

    def test_speed_scales_turns_and_measure_clock_together(self):
        score = dict(duration=12, measures=[dict(id='m1', page=1, start_ms=0),
            dict(id='m2', page=2, start_ms=6000)])
        for speed in (.75, 1., 1.25, 1.5):
            clock = playback_clock(score, 100, speed, 100+6.1/speed)
            self.assertAlmostEqual(clock['base_ms'], 6100)
            self.assertAlmostEqual(clock['duration'], 12/speed)
            self.assertEqual(current_measure(score, clock['base_ms'], 2)['id'], 'm2')
            self.assertIsNone(current_measure(score, clock['base_ms'], 1))
        self.assertFalse(playback_clock(score, 100, 1.5, 109)['playing'])
        self.assertFalse(playback_clock(score, 0, 1., 109)['playing'])

    def test_repeat_timeline_returns_to_earlier_measure(self):
        score = dict(measures=[dict(id='m1', page=1, ms=0), dict(id='m2', page=2, ms=1000),
                               dict(id='m1', page=1, ms=2000)])
        self.assertEqual(current_measure(score, 1500)['id'], 'm2')
        self.assertEqual(current_measure(score, 2500)['id'], 'm1')

    def test_tick_uses_prepared_pitch_preserving_audio_rate(self):
        ui = self.ui
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            score = dict(duration=12, pages=[], turns=[dict(ms=0, page=1)],
                         rates={'1.25': dict(wav='score-125.wav', duration=9.6)})
            (directory / 'score.json').write_text(json.dumps(score))
            state = dict(speed=1.25, play=dict(token=1, state='ready', directory=str(directory), speed=1.25))
            instance = types.SimpleNamespace(demo=None, state={}, token=None, score=None, player=None,
                started=0, audio_log=None, canvas=types.SimpleNamespace(queue_draw=lambda:None), stop=lambda:None)
            original_read = Path.read_text
            def read(path, *a, **kw):
                return ' 0 [xilinxDP ]: DP audio\n' if path.as_posix() == '/proc/asound/cards' else original_read(path, *a, **kw)
            player = types.SimpleNamespace(poll=lambda:None)
            with patch.object(ui, 'request', return_value=state), patch.object(Path, 'read_text', read), \
                 patch.object(ui.subprocess, 'Popen', return_value=player) as spawn, \
                 patch.object(ui.shutil, 'which', return_value=None), patch.object(ui.time, 'monotonic', return_value=100):
                self.assertTrue(ui.Window.tick(instance))
            self.assertEqual(instance.audio_error, '')
            self.assertEqual(Path(spawn.call_args.args[0][-1]).name, 'score-125.wav')
            self.assertEqual(instance.speed, 1.25)
            self.assertEqual(instance.started, 100)
            self.assertEqual(instance.audio_error, '')
            instance.audio_log.close()

    def test_playing_score_takes_fullscreen_branch(self):
        ui = self.ui
        state = dict(view='library', songs=[], selected='00', speed=1.25, play=dict(state='ready', speed=1.25))
        instance = types.SimpleNamespace(state=state, score=dict(duration=30), started=100, svg={1:object()}, page=1)
        widget = types.SimpleNamespace(get_allocated_width=lambda:1280, get_allocated_height=lambda:720)
        with patch.object(ui, 'playback_screen') as full, patch.object(ui.time, 'monotonic', return_value=105):
            ui.Window.draw(instance, widget, Cairo())
        full.assert_called_once()
        self.assertEqual(full.call_args.args[-1]['base_ms'], 6250)

    def test_orientation_warning_visible_beside_suspect_page(self):
        ui = self.ui
        page = dict(id='00-00', status='ok', stage='已完成', progress=100, orientation_suspect=True)
        state = dict(view='recognize', selected='00', songs=[dict(id='00', pages=[page], xml=True)], play={})
        instance = types.SimpleNamespace(state=state, score=None, started=0, svg={}, audio_error='', page=1)
        widget = types.SimpleNamespace(get_allocated_width=lambda:1280, get_allocated_height=lambda:720)
        with patch.object(ui, 'text') as labels, patch.object(ui, 'page_indicator'), \
             patch.object(ui, 'card'), patch.object(ui, 'box'), patch.object(ui, 'rule'), \
             patch.object(ui, 'rounded_path'), patch.object(ui, 'document_icon'):
            ui.Window.draw(instance, widget, Cairo())
        warning = [call for call in labels.call_args_list if call.args[1] == ORIENTATION_WARNING]
        self.assertEqual(len(warning), 1)
        self.assertTrue(warning[0].kwargs['wrap'])

    def test_measure_highlight_uses_svg_viewbox_fit(self):
        ui = self.ui
        score = dict(page_geometry={'1':[0, 0, 1000, 500]},
            measures=[dict(page=1, start_ms=0, bbox=[100, 100, 200, 100])])
        handle = types.SimpleNamespace(render_document=lambda *a:None)
        with patch.object(ui, 'rounded_path') as rounded:
            ui.score_sheet(Cairo(), handle, score, 1, 10, 10, 20, 1000, 500)
        self.assertEqual(rounded.call_args.args[1:], (106, 115, 208, 110, 5))


if __name__ == '__main__': unittest.main()
