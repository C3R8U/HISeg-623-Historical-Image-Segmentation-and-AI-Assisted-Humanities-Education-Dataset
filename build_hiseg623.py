#!/usr/bin/env python3
"""Build HISeg-623.zip using Python 3.9+ (standard library only).

Local use: place this script beside HISeg-623/ or HISeg-623_parts/, then run:
    python build_hiseg623.py

Download published parts and build the archive:
    python build_hiseg623.py --base-url https://YOUR-HOST/PATH/
The URL directory must contain HISeg-623_part01_of09.zip through part09_of09.zip.
For GitHub Releases, use the release asset directory:
    https://github.com/OWNER/REPO/releases/download/TAG/
Replace OWNER/REPO/TAG with a real published release. No data is hosted by this script.

Custom paths:
    python build_hiseg623.py --images /path/HISeg-623 --output /path/HISeg-623.zip
    python build_hiseg623.py --parts /path/HISeg-623_parts

These are collected historical image samples including crops, not AI-generated
images. This script packages existing data; it does not reconstruct pixels from code.
"""
import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
EXPECTED = {'maps':215, 'rubbings':198, 'murals':210}
# SHA-256 values of the nine supplied archives are filled during delivery.
PART_HASHES = {'HISeg-623_part01_of09.zip': '441422b07763204d74b1d951739f7d1b174a62ac2badb73abbae1efcc49caa0d', 'HISeg-623_part02_of09.zip': '6fffc63f432a9600073927953e5537d26f0a21452edbea290c583bbe9e2d446b', 'HISeg-623_part03_of09.zip': '1da45d87b618d0a6fe95efe5ece6e3319bef11d5191031d0ad9a99069e27bee4', 'HISeg-623_part04_of09.zip': '5da6eab154e0c89f3d6bcb1a25e82de25ad43781f798804f5f0350303d5a6889', 'HISeg-623_part05_of09.zip': 'd8a62f62ae033d06d8108148c4fa21d6a43ec68b8be8305c1956fd687d41292e', 'HISeg-623_part06_of09.zip': '90cc16e76b52ae484df56d75c6bcdf85ce5f12b3e705cee41cf95589a179e2a0', 'HISeg-623_part07_of09.zip': '3ac0287f94e928276c60c4fb03887e77079aecb693565d346ded4546778e3e83', 'HISeg-623_part08_of09.zip': 'a48d2a76ae352e57960848bf6e3cb48be18099cbae790fb575dae2db37610d45', 'HISeg-623_part09_of09.zip': '3319e74046ae754300e624492e35864a2a9996219099b2badff1a85cfcb63529'}

def check_name(name):
    p=PurePosixPath(name)
    if '\\' in name or len(p.parts)!=3 or p.parts[0]!='HISeg-623' or p.parts[1] not in EXPECTED:
        raise ValueError('Unexpected image path: '+name)
    if p.suffix.lower() not in ('.png','.jpg','.jpeg','.tif','.tiff','.webp'):
        raise ValueError('Unexpected file type: '+name)
    return p.parts[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):h.update(block)
    return h.hexdigest()

def verify_part(path):
    expected=PART_HASHES.get(path.name)
    if expected and digest(path)!=expected:
        raise ValueError('SHA-256 mismatch: '+str(path))

def download_parts(base_url, folder):
    if not base_url.startswith(('https://','http://')):
        raise ValueError('Use an HTTP(S) download directory URL.')
    folder.mkdir(parents=True,exist_ok=True)
    for i in range(1,10):
        name=f'HISeg-623_part{i:02d}_of09.zip'
        target=folder/name
        if target.exists():
            verify_part(target)
            print('Using cached:',name,flush=True)
            continue
        url=base_url.rstrip('/')+'/'+name
        print('Downloading:',url,flush=True)
        temp=target.with_suffix('.zip.download')
        req=urllib.request.Request(url,headers={'User-Agent':'HISeg-623-downloader/1.0'})
        try:
            with urllib.request.urlopen(req,timeout=120) as response, temp.open('wb') as f:
                shutil.copyfileobj(response,f,1024*1024)
            expected=PART_HASHES.get(name)
            if expected and digest(temp)!=expected:raise ValueError('Download checksum mismatch: '+name)
            with zipfile.ZipFile(temp) as z:
                if z.testzip() is not None:raise ValueError('Damaged download: '+name)
            temp.replace(target)
        except Exception:
            temp.unlink(missing_ok=True)
            raise

def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images',type=Path,default=HERE/'HISeg-623')
    parser.add_argument('--parts',type=Path,default=HERE/'HISeg-623_parts')
    parser.add_argument('--base-url',help='Published directory URL containing the nine part archives')
    parser.add_argument('--output',type=Path,default=HERE/'HISeg-623.zip')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Output already exists; choose another --output path: '+str(args.output))
    if args.base_url:download_parts(args.base_url,args.parts)
    use_images=args.images.is_dir() and not args.base_url
    parts=[args.parts/f'HISeg-623_part{i:02d}_of09.zip' for i in range(1,10)]
    if not use_images:
        for p in parts:
            if not p.is_file():raise FileNotFoundError('Missing '+str(p)+'. Supply the images, nine parts, or --base-url.')
            verify_part(p)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_name(args.output.name+'.building')
    if temp.exists():raise FileExistsError('Temporary output already exists: '+str(temp))
    counts=dict.fromkeys(EXPECTED,0)
    hashes={}
    def add(z,name,data):
        category=check_name(name)
        if name in hashes:raise ValueError('Duplicate path: '+name)
        counts[category]+=1
        hashes[name]=hashlib.sha256(data).hexdigest()
        z.writestr(name,data)
    try:
        with zipfile.ZipFile(temp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as result:
            if use_images:
                for p in sorted(args.images.rglob('*')):
                    if p.is_file():add(result,'HISeg-623/'+p.relative_to(args.images).as_posix(),p.read_bytes())
            else:
                for p in parts:
                    print('Reading:',p.name,flush=True)
                    with zipfile.ZipFile(p) as source:
                        for entry in source.infolist():
                            if not entry.is_dir():add(result,entry.filename,source.read(entry))
        if counts!=EXPECTED:raise ValueError('Wrong category counts: '+str(counts))
        with zipfile.ZipFile(temp) as z:
            if len(z.infolist())!=623:raise ValueError('Wrong image count')
            for name,expected in hashes.items():
                if hashlib.sha256(z.read(name)).hexdigest()!=expected:raise ValueError('Image verification failed: '+name)
        # Hard-link creation refuses to overwrite an output appearing during the run.
        os.link(temp,args.output)
        temp.unlink()
        print('Verified 623 images:',counts)
        print('Created:',args.output.resolve())
        print('Size: %.1f MiB'%(args.output.stat().st_size/1024**2))
    except Exception:
        temp.unlink(missing_ok=True)
        raise

if __name__=='__main__':
    main()
