'''
It is the script for calculation of metrics used for SVM training.
Author: Roman CHABAN, University of Geneva, 2021
'''

import argparse
import traceback
from typing import List, Tuple, Union
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from p_tqdm import p_map

from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import jaccard_score as jaccard

import utils


parser = argparse.ArgumentParser()
parser.add_argument("templates", type=Path, help="Path to synchronized templates (for example: /.../templates/sync/dens50)")
parser.add_argument("--bsize", default=684, type=int, help="Size of block used for splitting image on blocks")
parser.add_argument("--debug", default=False, type=bool, help="Whether to use parallelization or not")
parser.add_argument("--cpus", default=6, type=int, help="Number of CPUs used for parallelization")
parser.add_argument("--dens", default=50, type=int, help="Density of CDP")

args = parser.parse_args()

def imread(path: Union[Path, str], *flags):
    return cv.imread(str(path), *flags)

def norm(img: np.ndarray) -> np.ndarray:
    img = img - np.min(img, axis=(0, 1), keepdims=True)
    img = img / np.max(img, axis=(0, 1), keepdims=True)
    return img.astype(np.float32)

def img2blocks(
        img: np.ndarray,
        block_size: Tuple[int, int],
        step: int = 1,
        rows: Union[None, List[int]] = None,
        cols: Union[None, List[int]] = None
) -> List[np.ndarray]:
    def __get_blocks_idxs(
            img_shape: Tuple[int, int], block_size: Tuple[int, int], step: int = 1
    ) -> Tuple[List[int], List[int]]:

        ss = img_shape - np.asarray(block_size) + 1

        img_mat = np.zeros((ss[0], ss[1]))
        img_mat[::step, ::step] = 1
        img_mat[img_mat.shape[0] - 1, ::step] = 1
        img_mat[::step, img_mat.shape[1] - 1] = 1
        img_mat[img_mat.shape[0] - 1, img_mat.shape[1] - 1] = 1

        return np.where(img_mat == 1)

    if not rows or not cols:
        rows, cols = __get_blocks_idxs(img.shape, block_size, step)

    n = len(rows)
    blocks = []
    for i in range(n):
        blocks.append(img[rows[i]:rows[i] + block_size[0], cols[i]:cols[i] + block_size[1]])

    return blocks

def imthresh(img_gray: np.ndarray, thresh_adjust: int = -10, otsu: bool = True) -> np.ndarray:
    thresh_value, image_otsu = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    if otsu:
        return image_otsu.astype(np.uint8)

    if thresh_value + thresh_adjust <= 0:
        thresh_adjust = - thresh_value + 1

    _, image_bw = cv.threshold(img_gray, thresh_value + thresh_adjust, 255, cv.THRESH_BINARY)
    return image_bw.astype(np.uint8)


unfold = lambda x: [item for sublist in x for item in sublist]
load = lambda x: sorted(list(Path(x).glob('??????.tif*')))
read = lambda x: norm(utils.imread(x, -1))
binarize = lambda x: imthresh((x * 255).astype(np.uint8))
toblocks = lambda x: img2blocks(x, (args.bsize, args.bsize), args.bsize)


def calc_single(file: Path):
    try:
        path_t = args.templates / file.name
        img_t = read(path_t)
        img_y = read(file)
        img_yb = binarize(img_y)

        res = []
        bi = 1
        for b_t, b_y, b_yb in zip(toblocks(img_t), toblocks(img_y), toblocks(img_yb)):

            jaccard_val = jaccard(b_t.ravel(), b_yb.ravel())
            ssim_val = ssim(b_t, b_y)
            corr_val = cv.matchTemplate(b_t, b_y, cv.TM_CCORR_NORMED).ravel()[0]
            hamming_val = cv.countNonZero(cv.bitwise_xor(b_t, b_yb)) / b_y.size

            res.append([file.stem, file.parent.parent.name, str(file), bi, jaccard_val, ssim_val, corr_val, hamming_val])
            bi += 1

        return res
    except Exception as e:
        print(e)
        traceback.print_exc()
        return [file.stem, file.parent.parent.name, str(file)]


if __name__ == '__main__':
    files = []

    # Specify here the paths to the directories of synchronized printed CDPs
    dirs = [
        'HPI55_des3_812.8dpi_2400dpi',
        'HPI55_EHPI55_des3_812.8dpi_2400dpi',
        'HPI55_EHPI76_des3_812.8dpi_2400dpi',

        'HPI76_des3_812.8dpi_2400dpi',
        'HPI76_EHPI55_des3_812.8dpi_2400dpi',
        'HPI76_EHPI76_des3_812.8dpi_2400dpi',
    ]

    for dir in dirs:
        loaded = load(dir / f'dens{args.dens}')
        print(f"Loaded {len(loaded)} from {dir}")
        files.extend(loaded)

    if args.debug:
        res = [calc_single(f) for f in files]
    else:
        res = p_map(calc_single, files, num_cpus=args.cpus)

    df = pd.DataFrame(unfold(res))
    df = df.set_axis(
        ['label', 'subset', 'path', 'block', 'jaccard', 'ssim', 'corr', 'hamming'],
        axis='columns', inplace=False
    )
    df.to_csv('metrics.csv')
