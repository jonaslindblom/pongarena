"""Build the downloadable standalone contract without changing the game."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED, ZipInfo

root = Path(__file__).resolve().parents[2]
source = root / 'training/v2'
assert (source / 'game.ts').read_bytes() == (root / 'src/game.ts').read_bytes()
files = ['README.md', 'game.ts', 'headless.ts', 'package.json', 'fixtures.ts',
         'fixtures.json', 'reference.py', 'check.py']
path = root / 'public/training/pong-v2.zip'
path.parent.mkdir(parents=True, exist_ok=True)
with ZipFile(path, 'w', ZIP_DEFLATED) as archive:
    for name in files:
        info = ZipInfo('pong-v2/' + name, date_time=(2026, 11, 5, 0, 0, 0))
        info.compress_type = ZIP_DEFLATED
        archive.writestr(info, (source / name).read_bytes())
print(path)
