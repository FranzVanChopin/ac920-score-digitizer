import json,os,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/'app')]
from fat_library import FatLibrary
from onecard_common import write_json

class MetadataPermissions(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.card=self.root/'card';self.base=self.root/'base'
        (self.card/'OneCard').mkdir(parents=True);(self.card/'Outputs').mkdir()
        (self.base/'results').mkdir(parents=True)
        write_json(self.card/'OneCard/manifest.json',dict(boot_partuuid='boot',root_partuuid='root'))
        self.library=FatLibrary(self.card,self.base);self.album=self.base/'results/album.json'
        write_json(self.album,dict(pages={'00-00':{}},songs={'00':{}}))
    def separate_owner(self):
        if os.name=='nt' or os.geteuid()!=0:self.skipTest('requires Linux root to simulate root controller + unprivileged recognizer')
        os.chown(self.base/'results',65534,65534)
        self.assertEqual(self.album.stat().st_uid,0)
    def test_existing_root_private_file_is_repaired_without_deletion(self):
        self.separate_owner();self.library.reconcile()
        self.assertEqual(self.album.stat().st_uid,65534)
        self.assertEqual(self.album.stat().st_mode&0o777,0o600)
    def test_cleanup_rewrite_preserves_recognizer_access(self):
        self.separate_owner();write_json(self.library.index,dict(schema=1,ids=['00']))
        self.assertEqual(self.library.reconcile(),['00'])
        self.assertEqual(self.album.stat().st_uid,65534)
        self.assertEqual(json.loads(self.album.read_text())['pages'],{})
    def test_metadata_symlink_is_rejected(self):
        target=self.root/'outside.json';target.write_text('{"keep":true}')
        self.album.unlink()
        try:self.album.symlink_to(target)
        except OSError:self.skipTest('symlink permission')
        with self.assertRaises(ValueError):self.library.reconcile()
        self.assertEqual(target.read_text(),'{"keep":true}')

if __name__=='__main__':unittest.main(verbosity=2)
