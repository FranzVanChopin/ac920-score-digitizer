"""Render actual panel states without a connected monitor; never sends actions."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import cairo
import panel_ui


def draw(state, destination, **extra):
    surface=cairo.ImageSurface(cairo.FORMAT_ARGB32,1280,720)
    context=cairo.Context(surface)
    instance=SimpleNamespace(state=state,score=None,svg={},audio_error='',started=0,page=1)
    for key,value in extra.items():setattr(instance,key,value)
    widget=SimpleNamespace(get_allocated_width=lambda:1280,get_allocated_height=lambda:720)
    panel_ui.Window.draw(instance,widget,context)
    surface.write_to_png(str(destination))


if __name__=='__main__':
    folder=Path(sys.argv[1]);folder.mkdir(parents=True,exist_ok=True)
    state=panel_ui.request(dict(action='state'))
    (folder/'state.json').write_text(json.dumps(state,indent=2,ensure_ascii=False))
    draw(state,folder/'recognize.png')
    state['view']='library'
    draw(state,folder/'library.png')
    print('HEADLESS GTK/Cairo renders saved to',folder)
