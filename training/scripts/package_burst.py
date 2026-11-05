"""Package the experimental burst contract without modifying Classic v4."""
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

root=Path(__file__).resolve().parents[2]
source=root/'training/burst'
target=root/'public/training/pong-burst-v1.zip'
files=['README.md','game.ts','reference.py','headless.ts','check.py','check.ts','fixtures.json','package.json']
with ZipFile(target,'w',ZIP_DEFLATED) as archive:
    for name in files:
        info=ZipInfo('pong-burst-v1/'+name,date_time=(2026,11,5,0,0,0))
        info.compress_type=ZIP_DEFLATED
        archive.writestr(info,(source/name).read_bytes())
print(target)
