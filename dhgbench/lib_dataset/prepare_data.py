"""Install the bundled, hash-verified historical processed benchmark caches."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def prepare_data():
    expected = json.loads((ROOT / 'data/archives/members.sha256.json').read_text())
    with tarfile.open(ROOT / 'data/archives/processed-benchmarks.tar.gz') as archive:
        members = {m.name: m for m in archive.getmembers() if m.isfile()}
        if set(members) != set(expected):
            raise ValueError('Archive file inventory differs from manifest')
        for name, digest in expected.items():
            dest = (ROOT / name).resolve()
            if ROOT not in dest.parents or members[name].issym():
                raise ValueError(f'Unsafe member: {name}')
            content = archive.extractfile(members[name]).read()
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f'Archive checksum mismatch: {name}')
            if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
                raise FileExistsError(f'Refusing to replace different local data: {dest}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.write_bytes(content)
    # The legacy loader checks these directories even when all caches exist.
    for name in ('data/trad_data/cocitation', 'data/hete_data'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    print(f'Installed/verified {len(expected)} files for six datasets.')


