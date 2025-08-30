import subprocess
import shutil
from pathlib import Path
import logging
from dir_guard import DirGuard

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(name)s: %(message)s')
logger = logging.getLogger(__name__)

src = Path('/compute/babel-2-17/yusenh/OpenImagesV7')
dst = Path('/scratch/yusenh/vlr')
file = Path('validation.tar.gz')

with DirGuard(dst) as pg:
    if pg.days_since_creation() > 20:
        # Re-download data
        logger.info(f'Preparing data')

        shutil.rmtree(dst)
        dst.mkdir(parents=True)

        logger.info(f'Copying {src / file} to {dst}')
        copy_archive = [
            'rsync',
            '-ah',
            '--partial',
            '--append-verify',
            '--info=stats',
            str(src / file),
            f'{dst}/',
        ]
        subprocess.run(copy_archive, check=True)

        logger.info(f'Untaring archive')
        untar = ['tar', '-xf', str(dst / file), '-C', str(dst)]
        subprocess.run(untar, check=True)

        pg.update_creation_timestamp()

    else:
        logger.info(f'Using existing data')
