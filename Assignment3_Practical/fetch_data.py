"""Fetch five original PROMISE CSVs from the authors' versioned Zenodo archive.

Only requested ZIP members are transferred, using HTTP byte ranges. No source
code or the complete 1 GB archive is downloaded. ZIP CRCs are checked by zipfile.
"""
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RECORD = 'https://zenodo.org/api/records/3693686'
URL = RECORD + '/files/UnifiedBugDataset-1.2.zip/content'
SIZE = 1014131877
MEMBERS = {
    'ant-1.7': 'ant/ant-1.7/ant-1.7.csv',
    'jedit-4.3': 'jedit/jedit-4.3.csv',
    'poi-3.0': 'poi/poi-3.0/poi-3.0.csv',
    'ivy-2.0': 'ivy/ivy-2.0.csv',
    'camel-1.6': 'camel/camel-1.6/camel-1.6.csv',
}

class RemoteZip(io.RawIOBase):
    def __init__(self): self.pos = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else SIZE + offset
        return self.pos
    def read(self, n=-1):
        n = SIZE - self.pos if n < 0 else min(n, SIZE - self.pos)
        if n <= 0: return b''
        start = self.pos
        for attempt in range(3):
            try:
                req = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{start+n-1}'})
                with urllib.request.urlopen(req, timeout=40) as response:
                    if response.status != 206: raise RuntimeError('HTTP ranges are required')
                    if not response.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
                        raise RuntimeError('Unexpected byte range')
                    data = response.read(n)
                if len(data) != n: raise RuntimeError('Incomplete response')
                self.pos += n
                return data
            except Exception:
                if attempt == 2: raise
                time.sleep(2 * (attempt + 1))

def main():
    data_dir = ROOT / 'data'
    data_dir.mkdir(exist_ok=True)
    with urllib.request.urlopen(RECORD, timeout=30) as response:
        record = json.load(response)
    (data_dir / 'zenodo_metadata.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    manifest = {'record_doi': '10.5281/zenodo.3693686', 'version': '1.2',
                'archive_url': URL, 'archive_size_bytes': SIZE,
                'record_license': record['metadata']['license'],
                'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'files': []}
    with zipfile.ZipFile(RemoteZip()) as archive:
        for key, suffix in MEMBERS.items():
            member = 'UnifiedBugDataset-1.2/PROMISE/orig/defect/ck/' + suffix
            content = archive.read(member)
            (data_dir / f'{key}.csv').write_bytes(content)
            manifest['files'].append({'id': key, 'member': member, 'bytes': len(content),
                                      'sha256': hashlib.sha256(content).hexdigest()})
            print('Retrieved', key, len(content), flush=True)
        (data_dir / 'UPSTREAM_README.txt').write_bytes(archive.read('UnifiedBugDataset-1.2/README.txt'))
    (data_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Completed five real datasets; ZIP CRC checks passed.', flush=True)

if __name__ == '__main__': main()
