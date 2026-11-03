"""Keep the window composited; avoid direct fullscreen front-buffer updates."""
import subprocess

def keep_composited(window):
    try:
        import gi
        gi.require_version('GdkX11','3.0')
        from gi.repository import GdkX11
        xid=GdkX11.X11Window.get_xid(window.get_window())
        result=subprocess.run(['xprop','-id',str(xid),'-f','_NET_WM_BYPASS_COMPOSITOR','32c',
            '-set','_NET_WM_BYPASS_COMPOSITOR','2'],capture_output=True,text=True,timeout=2)
        if result.returncode:raise RuntimeError(result.stderr)
        print('[DISPLAY] compositing requested for fullscreen window',flush=True)
    except Exception as error:print('[DISPLAY] compositor policy: '+str(error),flush=True)
