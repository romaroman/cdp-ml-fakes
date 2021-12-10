'''
CDP: Use the trained model to produce the estimations of the digital templates from the printed codes
Author: Olga TARAN, University of Geneva, 2021
'''

from __future__ import print_function
import argparse
import yaml
import cv2

import sys
sys.path.insert(0,'..')

import libs.yaml_utils as yaml_utils
from libs.utils import *

from libs.DataLoader import DataLoader
from libs.EstimatiorModel import TemplateEstimatior
from libs.image2blocks import image2Blocks, blocks2Image

# ======================================================================================================================
parser = argparse.ArgumentParser(description="TEST: the templates estimator model trained wrt Dtt and Dt terms")
parser.add_argument("--config_path", default="./configuration.yml", type=str, help="The config file path")
# datset parameters
parser.add_argument("--symbol_size", default=8, type=int, help="The symbol size of digitized printed codes")
parser.add_argument("--target_symbol_size", default=1, type=int, help="The symbol size in digital template")
# model parameters
parser.add_argument("--type", default="Dtt_Dt", type=str, help="The model estimator configuration")
parser.add_argument("--lr", default=1e-5, type=float, help="Training learning rate")
parser.add_argument("--epoch", default=100, type=int, help="The test epoch")

parser.add_argument("--is_symbol_proc", default=True, type=int, help="Is to apply the symbolwise post-processing")
parser.add_argument("--save_estimatins_to", default="./data/estimations/", type=str, help="The path where to save the estimations")
parser.add_argument("--thr", default=0.5, type=float, help="Binarization threshold")
# log mode
parser.add_argument("--is_debug", default=False, type=int, help="Debug mode")

# ======================================================================================================================
args = parser.parse_args()
set_log_config(args.is_debug)
log.info("PID = %d\n" % os.getpid())

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ======================================================================================================================
def run(args):

    symbol_size = args.symbol_size
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    args.checkpoint_dir = "HP76_%s" % args.type
    args.dir = "HP76%s" % args.type

    log.info("Start Model preparation.....")
    Estimator = TemplateEstimatior(config, args, type=args.type)
    EstimationModel = Estimator.EstimationModel
    EstimationModel.load_weights("%s/EstimationModel_epoch_%d" % (Estimator.checkpoint_dir, args.epoch))

    # --- Data set -----------
    log.info("Start Train Data loading.....")
    DataGen = DataLoader(config, args, type="test", is_debug_mode=args.is_debug)

    # === Testing =================================================================================================
    n_batches = DataGen.n_batches
    file_names = DataGen.file_names

    l = -1
    errors = []
    for x_batch, t_batch in DataGen.datagen:
        l += 1
        if  l >= n_batches:
            break

        x_blocks = image2Blocks(x_batch[0],
                                config.dataset['args']["target_size"],
                                config.dataset['args']["augmentation_args"]["sub-block_step"])

        prediction_blocks = EstimationModel.predict(x_blocks)[0]
        t_predict = blocks2Image(prediction_blocks,
                                 config.dataset['args']["target_size"],
                                 config.dataset['args']["expected_target_size"],
                                 config.dataset['args']["augmentation_args"]["sub-block_step"])

        t_predict = t_predict.reshape((*config.dataset["args"]["expected_template_target_size"]))
        t_batch   = t_batch.reshape((*config.dataset["args"]["expected_template_target_size"]))

        if args.is_symbol_proc: # symbolwise binarization
            t_predict_binary = postProcessingSimbolWise(np.copy(t_predict), symbol_size=symbol_size, thr=args.thr)
        else: # global binarization
            t_predict_binary = np.copy(t_predict)
            t_predict_binary[t_predict_binary <= args.thr] = 0
            t_predict_binary[t_predict_binary != 0] = 1

        dist = np.sum(np.logical_xor(t_batch.reshape((-1)), t_predict_binary.reshape((-1)))) / (symbol_size ** 2)
        errors.append(dist)

        if args.target_symbol_size != args.symbol_size:
           t_predict_binary = postProcessingSimbolWise(np.copy(t_predict), symbol_size=symbol_size,
                                                       thr=args.thr, target_symbol_size=args.target_symbol_size)

        cv2.imwrite(args.save_estimatins_to + "%04d.png" % file_names[l], (t_predict_binary*255).astype(np.uint8))

    mean_error = np.mean(errors)
    total_symbols = (config.dataset["args"]["expected_template_target_size"][0] / symbol_size) ** 2
    mean_error_rate = mean_error / total_symbols
    log.info("mean number of error symbols   = %0.5f\t mean error rate = %0.5f" % (mean_error, mean_error_rate))

# ======================================================================================================================
if __name__ == "__main__":
    run(args)
























































