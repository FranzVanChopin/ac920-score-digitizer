"""Keep Linux and camera at the same DP 720p60 timing."""
import os,subprocess,sys,time
os.environ['GDK_SCALE']='1';os.environ['GDK_DPI_SCALE']='1'
runtime='/run/user/'+str(os.getuid())
if os.path.exists(runtime+'/bus'):
    os.environ['XDG_RUNTIME_DIR']=runtime
    os.environ['DBUS_SESSION_BUS_ADDRESS']='unix:path='+runtime+'/bus'
def command(args):
    try:return subprocess.run(args,capture_output=True,text=True,timeout=8)
    except (OSError,subprocess.TimeoutExpired):return None
for _ in range(20):
    status=command(['/usr/bin/xrandr','--query'])
    if status and 'DP-1 connected' in status.stdout:
        try:
            from dp_mode import configure
            configured=configure(1280,720)
        except Exception as error:
            print('[DISPLAY] desktop configuration: '+str(error),flush=True)
            configured=False
        if not configured:
            print('[DISPLAY] applying 720p60 through XRandR fallback',flush=True)
            command(['/usr/bin/xrandr','--output','DP-1','--mode','1280x720','--rate','60'])
        break
    time.sleep(.5)
for args in (['dpms','force','on'],['-dpms'],['s','off'],['s','noblank']):command(['/usr/bin/xset']+args)
os.execv(sys.executable,[sys.executable,'/opt/ac920-panel/panel_ui.py'])
