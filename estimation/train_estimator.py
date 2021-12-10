'''
CDP: digital templates estimator training based on the printed codes
Author: Olga TARAN, University of Geneva, 2021
'''

from __future__ import print_function
import argparse
import yaml

import libs.yaml_utils as yaml_utils
from libs.utils import *

from libs.DataLoader import DataLoader
from libs.EstimatiorModel import TemplateEstimatior

# ======================================================================================================================
parser = argparse.ArgumentParser(description="TRAIN: the templates estimator model trained wrt Dtt and Dt terms")
parser.add_argument("--config_path", default="./configuration.yml", type=str, help="The config file path")
# model parameters
parser.add_argument("--type", default="Dtt_Dt", type=str, help="The model estimator configuration")
parser.add_argument("--lr", default=1e-4, type=float, help="Training learning rate")
parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs")
parser.add_argument("--is_stochastic", default=True, type=int, help="Is to train the stochastic or deterministic model")
# log mode
parser.add_argument("--is_debug", default=True, type=int, help="Is debug mode?")

args = parser.parse_args()
# ======================================================================================================================
set_log_config(args.is_debug)
log.info("PID = %d\n" % os.getpid())

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ======================================================================================================================
def train():

    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    args.checkpoint_dir = "%s" % args.type
    args.dir = "%s" % args.type

    log.info("Start Model preparation.....")
    Estimator = TemplateEstimatior(config, args, type=args.type)
    EstimationModel = Estimator.EstimationModel
    DtModel = Estimator.DiscriminatorModel

    # --- Data set -----------
    log.info("Start Train Data loading.....")
    DataGen = DataLoader(config, args, type="train", is_debug_mode=args.is_debug)

    # === Training =================================================================================================
    for epoch in range(args.epochs):
        Loss = []
        Loss_dt  = []
        batches = 0
        save_each = saveSpeed(epoch)
        for x_batch, y_batch in DataGen.datagen:
            if config["dataset"]["args"]["is_stochastic"]:
                noise = np.random.normal(0, config["dataset"]["args"]["noise_std"], x_batch.size)
                x_batch += noise.reshape(x_batch.shape)

            y_batch = y_batch.reshape((-1, config["dataset"]["args"]["target_size"][0],
                                       config["dataset"]["args"]["target_size"][1], 1))

            # --- Dt -----
            x = np.concatenate((y_batch, EstimationModel.predict(x_batch)[0]))
            # real images label is 1.0
            y = np.ones([2 * y_batch.shape[0], 1])
            # fake images label is 0.0
            y[config.batchsize:, :] = 0.0
            loss = DtModel.train_on_batch(x, y)
            Loss_dt.append(loss)

            # --- estimator -----
            Loss.append(EstimationModel.train_on_batch(x_batch, [y_batch, np.ones([x_batch.shape[0], 1])])[1])

            batches += 1
            if batches >= DataGen.n_batches:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

        log.info(f"epoch : {epoch}, \t"
                 f"Dtt = {np.mean(np.asarray(Loss))}\t "
                 f"Dt = {np.mean(np.asarray(Loss_dt))}")


        # ------------------------------------------------------------------------
        if epoch % save_each == 0 or epoch == args.epochs:
            EstimationModel.save_weights("%s/EstimationModel_epoch_%d" % (Estimator.checkpoint_dir, epoch))

# ======================================================================================================================
if __name__ == "__main__":
    train()





























































