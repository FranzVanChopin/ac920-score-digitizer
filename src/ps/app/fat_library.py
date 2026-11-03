"""FAT Outputs is authoritative; ext4 results and renders are disposable mirrors."""
import json,os,re,shutil,stat,xml.etree.ElementTree as ET
from pathlib import Path
from onecard_common import atomic,write_json

SONG=re.compile(r'[0-9]{2}')

class FatLibrary:
    def __init__(self,album,base):
        self.album,self.base=Path(album),Path(base)
        self.outputs=self.album/'Outputs';self.results=self.base/'results'
        self.index=self.base/'fat-library.json'

    def ids(self):
        # A missing album mount must never be mistaken for deleting all songs.
        marker=json.loads((self.album/'OneCard/manifest.json').read_text(encoding='utf-8'))
        if not marker.get('boot_partuuid') or not marker.get('root_partuuid'):
            raise ValueError('FAT identity marker missing')
        return {p.stem for p in self.outputs.glob('??.musicxml')
                if SONG.fullmatch(p.stem) and p.is_file()}

    def previous(self):
        if not self.index.exists():return set()
        value=json.loads(self.index.read_text(encoding='utf-8'))
        ids=value.get('ids')
        if value.get('schema')!=1 or not isinstance(ids,list) or any(not isinstance(x,str) or not SONG.fullmatch(x) for x in ids):
            raise ValueError('Invalid FAT library index; cleanup refused')
        return set(ids)

    def record_current(self):
        value=dict(schema=1,ids=sorted(self.ids()))
        if not self.index.exists() or json.loads(self.index.read_text(encoding='utf-8'))!=value:
            write_json(self.index,value)

    def migrate_legacy(self):
        """Explicit one-time install migration. Never import again after deletion."""
        if self.index.exists():return []
        self.ids()  # Validate mounted album before creating anything.
        self.outputs.mkdir(exist_ok=True)
        imported=[]
        for source in sorted(self.results.glob('??.musicxml')):
            if not SONG.fullmatch(source.stem):continue
            dest=self.outputs/source.name
            if dest.exists():continue  # Existing FAT copy wins, including user edits.
            data=source.read_bytes()
            root=ET.fromstring(data)
            if root.tag.rsplit('}',1)[-1] not in ('score-partwise','score-timewise'):
                raise ValueError('Invalid MusicXML: '+source.name)
            atomic(dest,data);assert dest.read_bytes()==data;imported.append(source.stem)
        self.record_current()
        return imported

    @staticmethod
    def remove_owned(path,owner):
        path,owner=Path(path),Path(owner)
        if path.parent.resolve()!=owner.resolve():raise ValueError('Cleanup path outside owned directory')
        if path.is_symlink():path.unlink();return
        if not path.exists():return
        if path.is_dir():shutil.rmtree(path)
        else:path.unlink()

    def reconcile(self):
        """Run at controller startup, after the card is mounted and validated."""
        current=self.ids();deleted=self.previous()-current
        album=self.results/'album.json'
        if album.is_symlink():raise ValueError('Recognition metadata symlink; cleanup refused')
        value=json.loads(album.read_text(encoding='utf-8')) if deleted and album.exists() else None
        if value is not None and (not isinstance(value,dict) or not isinstance(value.get('pages',{}),dict)
                                  or not isinstance(value.get('songs',{}),dict)):
            raise ValueError('Invalid recognition state; cleanup refused')
        for song in sorted(deleted):
            self.remove_owned(self.results/(song+'.musicxml'),self.results)
            for folder in (self.results,self.results/'pages',self.base/'work'):
                if not folder.is_dir():continue
                for p in folder.iterdir():
                    if re.fullmatch(re.escape(song)+r'-\d{2}(?:[_.-].*)?',p.name):
                        self.remove_owned(p,folder)
            cache=self.base/'score-cache'
            if cache.is_dir():
                for p in cache.iterdir():
                    if re.fullmatch(re.escape(song)+r'_[0-9A-Fa-f]{12}_r\d+',p.name):
                        self.remove_owned(p,cache)
        if value is not None:
            pages=value.get('pages',{})
            kept={k:v for k,v in pages.items() if k[:2] not in deleted}
            songs=value.get('songs',{})
            kept_songs={k:v for k,v in songs.items() if k not in deleted}
            if kept!=pages or kept_songs!=songs:
                value.update(pages=kept,songs=kept_songs);write_json(album,value)
        # atomic() creates a private file owned by its caller (the root
        # controller). The recognition subprocess runs as the results owner.
        # Repair existing metadata too, even when no song was deleted today.
        if album.exists() and os.name!='nt':
            owner=self.results.stat();metadata=album.stat()
            if not stat.S_ISREG(metadata.st_mode):raise ValueError('Invalid recognition metadata file')
            os.chown(album,owner.st_uid,owner.st_gid)
            os.chmod(album,0o600)
        self.record_current()
        return sorted(deleted)
