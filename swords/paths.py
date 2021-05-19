import os
import pathlib

LIB_DIR = os.path.dirname(os.path.join(pathlib.Path(__file__).absolute()))
REPO_DIR = os.path.dirname(LIB_DIR)

_REPO_ASSETS_DIR = os.path.join(REPO_DIR, 'assets')
if 'SWORDS_ASSETS_DIR' in os.environ:
  ASSETS_DIR = os.environ['SWORDS_ASSETS_DIR']
  os.makedirs(ASSETS_DIR, exist_ok=True)
  for vis in ['public', 'private']:
    for root, dirs, files in os.walk(os.path.join(_REPO_ASSETS_DIR, vis)):
      for fn in files:
        fp = os.path.join(root, fn)
        rel_fp = pathlib.Path(fp).relative_to(pathlib.Path(_REPO_ASSETS_DIR))
        out_fp = os.path.join(ASSETS_DIR, rel_fp)
        out_dir = os.path.split(out_fp)[0]
        os.makedirs(out_dir, exist_ok=True)
        if not os.path.exists(out_fp):
          os.symlink(fp, out_fp)
else:
  ASSETS_DIR = _REPO_ASSETS_DIR

DATASETS_CACHE_DIR = os.path.join(ASSETS_DIR, 'parsed')
