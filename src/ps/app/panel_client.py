"""Wait for controller changes off the GTK thread."""
import json,socket,threading

SOCKET='/run/ac920-panel/control.sock'

def request(value,timeout=.4,path=SOCKET):
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(timeout);connection.connect(path)
        connection.sendall((json.dumps(value)+'\n').encode())
        buffer=bytearray()
        while not buffer.endswith(b'\n'):
            part=connection.recv(65536)
            if not part:raise OSError('controller disconnected')
            buffer.extend(part)
        return json.loads(buffer)

class StateWatch(threading.Thread):
    def __init__(self,deliver,path=SOCKET):
        super().__init__(daemon=True)
        self.deliver,self.path=deliver,path;self.stopping=threading.Event()
    def close(self):self.stopping.set()
    def run(self):
        previous=None;revision=-1
        while not self.stopping.is_set():
            try:
                value=request(dict(action='wait',after=revision),timeout=2.5,path=self.path)
                if 'error' in value:raise OSError(value['error'])
                revision=value.get('revision',-1)
                if value!=previous and not self.stopping.is_set():self.deliver(value)
                previous=value
                if revision==-1:self.stopping.wait(.04)
            except (OSError,ValueError):self.stopping.wait(.3)
