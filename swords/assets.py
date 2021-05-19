import hashlib
import json
import glob
import gzip
import os
import tarfile
import requests
import zipfile

from .paths import ASSETS_DIR, LIB_DIR


ASSETS = {}
for json_fp in glob.glob(os.path.join(LIB_DIR, 'assets', '*.json')):
  with open(json_fp, 'r') as f:
    ASSETS.update(json.load(f))
for fattrs in ASSETS.values():
  visibility = 'private' if bool(fattrs.get('private')) else 'public'
  fattrs['fp'] = os.path.join(ASSETS_DIR, visibility, fattrs['fp_rel'])


def file_from_bundle(bundle_fp, fp):
  if bundle_fp.endswith('.zip'):
    fopen = zipfile.ZipFile
    fextract = lambda z, fn: z.read(fn)
  elif bundle_fp.endswith('.tar.gz'):
    fopen = tarfile.open
    fextract = lambda z, fn: z.extractfile(z.getmember(fn)).read()
  else:
    raise ValueError()
  with fopen(bundle_fp) as z:
    return fextract(z, fp)


def main(argv):
  asset_tags = ASSETS.keys()
  args = [a for a in argv if not a.startswith('--')]
  if len(args) > 0:
    asset_tags = [t for t in asset_tags if t.startswith(args[0].strip())]

  for tag in asset_tags:
    fattrs = ASSETS[tag]
    if bool(fattrs.get('private')) and '--check_private' not in argv:
      continue

    print('-' * 80)
    print(tag)
    fp = fattrs['fp']
    url = fattrs.get('url', '').strip()

    if not os.path.exists(fp):
      # Create directory
      fp_dir = os.path.split(fp)[0]
      os.makedirs(fp_dir, exist_ok=True)

      # Download file
      if len(url) > 0:
        print(f'Downloading from: {url}')
        try:
          with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(fp, 'wb') as f:
              for chunk in r.iter_content(chunk_size=4096):
                f.write(chunk)
        except:
          raise Exception('Could not download file')
      else:
        raise Exception('File not found!')

    if 'checksum_gunzipped' in fattrs:
      open_fn = gzip.open
      expected_checksum = fattrs['checksum_gunzipped']
    else:
      open_fn = open
      expected_checksum = fattrs.get('checksum')

    print(f'Verifying file: {fp}')
    computed_checksum = hashlib.sha256()
    with open_fn(fp, 'rb') as f:
      while True:
        data = f.read(4096)
        if not data:
          break
        computed_checksum.update(data)

    computed_checksum = computed_checksum.hexdigest()
    if expected_checksum is None or len(expected_checksum.strip()) == 0:
      print(computed_checksum)
    else:
      if computed_checksum != expected_checksum:
        raise Exception('File has wrong checksum... delete it and try again!')
      print('Verified!')
