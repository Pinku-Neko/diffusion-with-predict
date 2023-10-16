# here the main board
from modules.images.tools import plot_grayscale_image
from modules.models.tools import init_models
from modules.models.IO import load_model
from modules.models.test import evaluate_regression
from modules.train.MLP import train_MLP
from modules.train.ddpm import train_diffusion
from modules.applications.test import get_image
import modules.images.tools as imgtools
import modules.applications.generation as gen
import modules.applications.restoration as restore
import modules.applications.imageblur as blur


def main():
    # get previous diffusion model

    # filename diffusion
    diffusion_name = "diffusion_epoch_0_loss_0.0357.pt"

    # filename regression
    regression_name = "./saved_models/regression_best_256_0.0001.pt"

    # load models
    diffusion, _ = init_models(regression_layer_dim=128)
    # diffusion = load_model(diffusion, diffusion_name)
    # regression = load_model(regression, regression_name)

    # generate image from pure noise
    # anim_normal, anim_predict = gen.double_generate(diffusion,regression)
    # imgtools.compare_grayscale_animations(anim_normal,anim_predict,"comparison.gif")

    # restore images
    # test_image = get_image()
    # noisy_image = blur.get_noisy_image(test_image, 100)
    # anim_normal, anim_predict = restore.double_restore(noisy_image,diffusion,regression)
    # imgtools.compare_grayscale_animations(anim_normal,anim_predict,"restore_comparison.gif")

    # look for noise level, that human can recognize object for weights
    # test_image = get_image()
    # noise_animation = blur.animate_noisy_image(test_image,time=140)
    # imgtools.animate_grayscale_images(noise_animation)

    # do training MLP
    train_MLP(num_epochs=2000, layer_dim=128,
              lr=1e-4, batch_size=128, tolerance=2000)

    # do training diffusion model
    # train_diffusion(num_epochs=2000,lr=1e-4,batch_size=128,tolerance=2000)


    # do test
    # training.test_power_training()

    # evaluate model
    # evaluate_regression(regression=regression)


if __name__ == "__main__":
    main()
