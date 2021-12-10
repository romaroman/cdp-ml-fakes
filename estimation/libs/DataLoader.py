import numpy as np
import skimage.io
from skimage.exposure import adjust_gamma
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from libs.BaseDataLoader import BaseDataLoader
from libs.image2blocks import image2Blocks, getBlocksIndices

# ======================================================================================================================

class DataLoader(BaseDataLoader):

    def __init__(self, config, args, type="train", is_debug_mode=False):
        super().__init__(config, args, type, is_debug_mode)

        self._is_debug_mode = is_debug_mode
        self.config = config
        self.type   = type if type else "test"
        self.symbol_size = config.dataset['args']["symbol_size"]

        self._seed = args.seed if "seed" in args else -1
        self._indices = []
        self.file_names = []
        self._code_name = args.code_name if "code_name" in args else "%04d.png"

        self._rows = []
        self._cols = []

        self._binary_codes_path = args.templates_path if "templates_path" in args else config.dataset["args"]["templates_path"]
        self._printed_codes_path = args.printed_path if "printed_path" in args else config.dataset["args"]["printed_path"]

        self._printed = []
        self._binary  = []

        self._initDataSet()

    def _initDataSet(self):
        # load data
        self._loadData(self.config.dataset['args'])
        if self.type == "train":
            if self.config.dataset['args']["template_target_size"] == [1, 1]:
                self._binary = np.max(self._binary, axis=(1, 2))
            else:
                self._binary = self.reshapeData(self._binary, self.config.dataset['args']["template_target_size"])
            self._printed = self.reshapeData(self._printed, self.config.dataset['args']["target_size"])
            if "template_target_channels" in self.config.dataset['args']:
                n, h, w, c = self._printed.shape
                k = len(self.config.dataset["args"]["template_target_channels"])

                printed_codes = np.zeros((n, h, w, c*(k+1)))
                for i in range(n):
                    ii = -1
                    for j in range(c):
                        ii += 1
                        printed_codes[i, ..., ii] = self._printed[i, ..., j]
                        for ij in range(k):
                            ii += 1
                            if self.config.dataset["args"]["template_target_channels"][ij] < 0: # negative
                                printed_codes[i, ..., ii] = 1 - adjust_gamma(self._printed[i, ..., j],
                                                                             gamma=abs(self.config.dataset["args"]["template_target_channels"][ij]))
                            elif self.config.dataset["args"]["template_target_channels"][ij] > 0: # gamma correction
                                printed_codes[i, ..., ii] = adjust_gamma(self._printed[i, ..., j],
                                                                         gamma=self.config.dataset["args"]["template_target_channels"][ij])

                del self._printed
                self._printed = printed_codes

        # create data generator
        datagen = ImageDataGenerator(samplewise_center=False, samplewise_std_normalization=False)

        # compile the data generators
        if self.type == "train":
            self.n_batches = self._printed.shape[0] // self.config.batchsize + 1
            self.datagen = datagen.flow(x=self._printed, y=self._binary,
                                        batch_size=self.config.batchsize, shuffle=True)
        else:
            self.n_batches = self._printed.shape[0]
            self.datagen = datagen.flow(x=self._printed, y=self._binary, batch_size=1, shuffle=False)

    def _loadImages(self, args, inds):
        self._indices = inds if not isinstance(inds, str) else self._loadIndices(inds)

        N = len(self._indices)
        if self.type == "train":
            k = self._getNumberofBlocks(args)

            printed = np.zeros((k*N, (*args["target_size"])))
            binary  = np.zeros((k*N, (*args["template_target_size"])))
        else:
            printed = np.zeros((N, (*args["expected_target_size"])))
            binary  = np.zeros((N, (*args["expected_template_target_size"])))

        i = -1
        for ind in self._indices:
            self.file_names.append(self._code_name % ind)
            image_x = skimage.io.imread(self._printed_codes_path + "/" + self._code_name % ind).astype(np.float64)
            if len(image_x.shape) < len(args["target_size"]):
                image_x = image_x.reshape((image_x.shape[0], image_x.shape[1], 1))
            image_y = skimage.io.imread(self._binary_codes_path + "/" + self._code_name % ind).astype(np.float64)

            image_x = self.normaliseDynamicRange(image_x, args)
            image_y = self.normaliseDynamicRange(image_y, args)

            if self.type == "train":
                x_blocks = image2Blocks(image_x,
                                        args["target_size"],
                                        args["augmentation_args"]["sub-block_step"],
                                        self._rows, self._cols)
                y_blocks = image2Blocks(image_y,
                                        args["template_target_size"],
                                        args["augmentation_args"]["sub-block_step"],
                                        self._rows, self._cols)
                for ij in range(x_blocks.shape[0]):
                    i += 1
                    printed[i] = x_blocks[ij]
                    binary[i] = y_blocks[ij]

            else:
                i += 1
                printed[i] = image_x
                binary[i] = image_y

        return printed, binary

    def _getNumberofBlocks(self, args):
        s0 = args["expected_target_size"][0] - args["target_size"][0] + 1
        s1 = args["expected_target_size"][1] - args["target_size"][1] + 1
        step = args["augmentation_args"]["sub-block_step"]

        k0 = len(np.arange(0, s0, step))+1
        k1 = len(np.arange(0, s1, step))+1

        self._rows, self._cols = getBlocksIndices(args["expected_target_size"], args["target_size"], args["augmentation_args"]["sub-block_step"])

        return k0*k1
















