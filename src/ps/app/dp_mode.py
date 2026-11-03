"""Keep Mutter's logical display and the native DP timing in agreement.

Interface contract: GNOME/mutter gnome-42, data/dbus-interfaces/
org.gnome.Mutter.DisplayConfig.xml, ApplyMonitorsConfig temporary method=1.
"""
import time
from gi.repository import Gio,GLib

def configure(width,height):
    proxy=Gio.DBusProxy.new_for_bus_sync(Gio.BusType.SESSION,Gio.DBusProxyFlags.NONE,None,
        'org.gnome.Mutter.DisplayConfig','/org/gnome/Mutter/DisplayConfig',
        'org.gnome.Mutter.DisplayConfig',None)
    serial,monitors,_,_=proxy.call_sync('GetCurrentState',None,Gio.DBusCallFlags.NONE,3000,None).unpack()
    monitor=next((m for m in monitors if m[0][0]=='DP-1'),None)
    if monitor is None:return False
    modes=[m for m in monitor[1] if m[1:3]==(width,height) or list(m[1:3])==[width,height]]
    modes=[m for m in modes if abs(m[3]-60)<.2 and 1.0 in m[5]]
    if not modes:return False
    mode=min(modes,key=lambda m:abs(m[3]-60))
    args=GLib.Variant('(uua(iiduba(ssa{sv}))a{sv})',
        (serial,1,[(0,0,1.0,0,True,[('DP-1',mode[0],{})])],{}))
    proxy.call_sync('ApplyMonitorsConfig',args,Gio.DBusCallFlags.NONE,6000,None)
    for _ in range(20):
        _,active,logical,_=proxy.call_sync('GetCurrentState',None,Gio.DBusCallFlags.NONE,3000,None).unpack()
        if logical and any(m[0][0]=='DP-1' and any(
            list(x[1:3])==[width,height] and x[6].get('is-current',False) for x in m[1]) for m in active):
            return True
        time.sleep(.1)
    return False
