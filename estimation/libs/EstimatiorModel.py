import datetime
from libs.BaseClass import BaseClass
from libs.UNet import *
from libs.Discriminators import ClassicalDiscriminator

# ======================================================================================================================

class TemplateEstimatior(BaseClass):
    def __init__(self, config, args, type="unet"):

        self.config = config
        self.type = type
        self.discriminator_weight = args.discriminator_weight if "discriminator_weight" in args else 1
        self.__layer_normalisation = args.unet_layer_normalisation if "unet_layer_normalisation" in args else self.config.models["unet"]["layer_normalisation"]

        self.__createResDirs(args)

        self.tensor_board = keras.callbacks.TensorBoard(
            log_dir=self.tensor_board_dir,
            histogram_freq=1,
            write_graph=True,
            write_grads=True,
            write_images=True
        )

        self.__initUnetClassicalDiscriminator()
        self.tensor_board.set_model(self.EstimationModel)

    def __initUnetClassicalDiscriminator(self):
        input = Input(shape=(self.config["dataset"]["args"]["target_size"][0],
                             self.config["dataset"]["args"]["target_size"][1],
                             self.config.models["unet"]["input_channels"]))

        input_dt = Input(shape=(self.config["dataset"]["args"]["target_size"][0],
                                self.config["dataset"]["args"]["target_size"][1],
                                self.config["dataset"]["args"]["target_size"][2]))

        # --- init Models -------
        self.UnetModel = UNet(filters=self.config.models["unet"]["filters"],
                                  output_channels=self.config.models["unet"]["output_channels"],
                                  layer_normalisation=self.__layer_normalisation).init(input)
        self.Discriminator = ClassicalDiscriminator(filters=self.config.models["ClassicalDiscriminator"]["filters"]).init(input_dt)

        # --- DiscriminatorModel -------
        optimizer_discr = self.__getOptimizer(self.config.models["ClassicalDiscriminator"]["optimizer"], self.config.models["ClassicalDiscriminator"]["lr"])
        loss_discr = self.__getLoss(self.config.models["ClassicalDiscriminator"]["loss"])

        self.DiscriminatorModel = Model(inputs=input_dt, outputs=self.Discriminator(input_dt), name="discriminator")
        self.DiscriminatorModel.compile(loss=loss_discr, optimizer=optimizer_discr)
        self.Discriminator.trainable = False

        # --- Nested estimator -------
        optimizer = self.__getOptimizer(self.config.models["unet"]["optimizer"], self.config.models["unet"]["lr"])
        loss = self.__getLoss(self.config.models["unet"]["loss"])

        estimation = self.UnetModel(input)
        self.EstimationModel = Model(inputs=[input],
                                     outputs=[estimation, self.Discriminator(estimation)],
                                     name="estimator")

        self.EstimationModel.compile(loss=[loss, self.config.models["ClassicalDiscriminator"]["loss"]],
                                     loss_weights=[self.config.models["unet"]["loss_weight"], self.config.models["ClassicalDiscriminator"]["loss_weight"]],
                                     optimizer=optimizer)


    def __getLoss(self, loss):
        if loss == "binary_crossentropy":
            return tf.keras.losses.binary_crossentropy
        elif loss == "mse":
            return tf.keras.losses.mean_squared_error
        elif loss == "l1":
            return tf.keras.losses.mean_absolute_error

    def __getOptimizer(self, optimazer, lr):
        if optimazer == "adam":
            return tf.keras.optimizers.Adam(learning_rate=lr)


    def __createResDirs(self, args):
        self.checkpoint_dir = self.makeDir(self.config.checkpoint_dir + "/" + args.checkpoint_dir)
        self.results_dir = self.makeDir(self.config.results_dir + "/" + args.dir)
        self.tensor_board_dir = self.makeDir("./TensorBoard/" + args.dir + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.makeDir(self.config.results_dir + "/" + args.dir + "/prediction/")

