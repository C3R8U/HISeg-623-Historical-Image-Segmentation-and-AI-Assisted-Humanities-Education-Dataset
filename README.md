# HISeg-623-Historical-Image-Segmentation-and-AI-Assisted-Humanities-Education-Dataset
HISeg-623
HISeg-623 is a collection of 623 historical image samples organized into three categories:
Category	Folder	Samples
Ming and Qing dynasty historical maps	maps/	215
Han dynasty pictorial stone rubbings	rubbings/	198
Tang dynasty tomb murals	murals/	210
Total		623


The collection includes crops derived from source images. The 623 samples therefore do not represent 623 independent historical objects or source images. Segmentation masks and pixel-level annotations are not included.
Requirements
- Python 3.9 or later
- No third-party Python packages are required
Build from Local Images
Place the script alongside the image directory:
project/
├── build_hiseg623.py
└── HISeg-623/
    ├── maps/
    ├── rubbings/
    └── murals/
Run:
python build_hiseg623.py
The script creates HISeg-623.zip in its own directory.
Build from the Nine ZIP Parts
Alternatively, organize the downloaded parts as follows:
project/
├── build_hiseg623.py
└── HISeg-623_parts/
    ├── HISeg-623_part01_of09.zip
    ├── HISeg-623_part02_of09.zip
    ├── ...
    └── HISeg-623_part09_of09.zip
Run the same command:
python build_hiseg623.py
Each part can also be extracted independently. The script combines all nine parts into one archive while preserving the category folders.
Download and Build Automatically
If the nine parts are published online, provide their download directory:
python build_hiseg623.py --base-url "https://YOUR-HOST/PATH/"
For assets uploaded to a GitHub Release:
python build_hiseg623.py --base-url "https://github.com/OWNER/REPO/releases/download/TAG/"
Replace OWNER, REPO, and TAG with the actual release details. All nine files must retain their original filenames.
These URLs are examples, not active dataset download links. The script requires access to the image files or published ZIP parts; it does not generate historical images from code.
Custom Paths
Build from a specific image directory:
python build_hiseg623.py --images "/path/to/HISeg-623"
Build from a specific parts directory:
python build_hiseg623.py --parts "/path/to/HISeg-623_parts"
Choose a different output filename:
python build_hiseg623.py --output "/path/to/HISeg-623-complete.zip"
Existing output files are not overwritten. Choose another output path if necessary.
Validation
The script checks:
- The expected category counts: 215 maps, 198 rubbings, and 210 murals
- A total of 623 image files
- Duplicate archive paths and unexpected file types
- SHA-256 checksums for the supplied ZIP parts
- Image integrity after creating the final archive
The resulting ZIP is approximately 377 MiB. Archive size may vary with compression settings.
Research Use
For machine-learning experiments, keep crops from the same source image in the same training, validation, or test split to avoid data leakage. Source relationships must be established before splitting; the build script does not perform this step.
Segmentation research requires additional annotation because this release contains images only.
Image Rights
Public availability does not necessarily imply unrestricted reuse. Consult the original image providers for copyright, attribution, and redistribution requirements. This README does not grant a blanket license for the images.


19:59
