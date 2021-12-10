import numpy as np

def getBlocksIndices(image_shape, block_size, step=1):
    ss = image_shape - np.asarray(block_size) + 1
    idxMat = np.zeros((ss[0], ss[1]))
    idxMat[::step, ::step] = 1
    idxMat[idxMat.shape[0] - 1, ::step] = 1
    idxMat[::step, idxMat.shape[1] - 1] = 1
    idxMat[idxMat.shape[0] - 1, idxMat.shape[1] - 1] = 1

    [rows, cols] = np.where(idxMat == 1)

    return rows, cols

def checkInstance(input):
    if not isinstance(input,(np.ndarray)):
        input = np.asarray(input)

    return input

def blocks2Image(blocks, block_size, image_size, step=1, rows=[], cols=[]):
    image = np.zeros((image_size))
    count = np.zeros((image_size))

    if len(rows) == 0 or len(cols) == 0:
        rows, cols = getBlocksIndices(image.shape, block_size, step)

    n = len(rows)
    for i in range(n):
        image[rows[i]:rows[i] + block_size[0], cols[i]:cols[i] + block_size[1]] += blocks[i]
        count[rows[i]:rows[i] + block_size[0], cols[i]:cols[i] + block_size[1]] += 1

    image /= count

    return image


def image2Blocks(image, block_size, step=1, rows=[], cols=[]):

    if len(rows) == 0 or len(cols) == 0:
        rows, cols = getBlocksIndices(image.shape, block_size, step)

    n = len(rows)
    blocks = np.zeros((n, *block_size))
    for i in range(n):
        blocks[i] = image[rows[i]:rows[i] + block_size[0], cols[i]:cols[i] + block_size[1]]

    return blocks
