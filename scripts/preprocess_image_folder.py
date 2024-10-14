from pathlib import Path
from shutil import copy


base_path = Path(Path().absolute().parent, 'data', 'raw', 'cracks', 'images', 'default', 'db_cracks')
images = [image for image in base_path.rglob('*.JPG')]

for image in images:
    try:
        copy(image, base_path)
    except Exception as e:
        print(e)
