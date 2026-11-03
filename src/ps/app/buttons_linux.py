"""Read Linux gpio-keys evdev events. A DT-specific installer supplies key codes."""
import glob
import fcntl
import os
import select
import struct
import time

EVENT = struct.Struct('@llHHi')
# F13/F14/F15, independent of the USB keyboard used for service/debugging.
CODES = {183: 'S1', 184: 'S2', 185: 'S3'}
EVIOCGRAB = 0x40044590


class Keys:
    def __init__(self):
        self.fds = []
        self.down = {key: False for key in CODES.values()}
        self.transitions = []
        for path in glob.glob('/sys/class/input/event*/device/name'):
            name = open(path).read().strip()
            if name != 'ac920-panel':
                continue
            event = path.split('/')[4]
            fd = os.open('/dev/input/' + event, os.O_RDONLY | os.O_NONBLOCK)
            try:
                # These are board controls, not GNOME desktop shortcuts. Only
                # grab the named gpio-keys device; USB keyboards stay available.
                fcntl.ioctl(fd, EVIOCGRAB, 1)
            except BaseException:
                os.close(fd)
                self.close()
                raise
            self.fds.append(fd)
        if not self.fds:
            raise RuntimeError('ac920-panel gpio-keys input missing; check Linux DT and bitstream')

    def poll(self):
        self.transitions = []
        ready, _, _ = select.select(self.fds, [], [], 0)
        for fd in ready:
            while True:
                try:
                    data = os.read(fd, EVENT.size * 32)
                except BlockingIOError:
                    break
                if not data:
                    break
                for at in range(0, len(data), EVENT.size):
                    seconds, micros, kind, code, value = EVENT.unpack_from(data, at)
                    if kind == 1 and code in CODES and value in (0, 1):
                        self.down[CODES[code]] = bool(value)
                        delay = (time.time() - seconds - micros / 1000000) * 1000
                        self.transitions.append((CODES[code],bool(value),time.monotonic()-delay/1000))
                        print(f'[KEY] {CODES[code]} {"down" if value else "up"} queue_ms={delay:.1f}', flush=True)
        return self.down

    def close(self):
        for fd in self.fds:
            try:
                fcntl.ioctl(fd, EVIOCGRAB, 0)
            finally:
                os.close(fd)
        self.fds = []
