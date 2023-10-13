# here the main board

def main():
    # get previous diffusion model

    # filename diffusion
    diffusion_name = "diffusion_epoch_0_loss_0.0357.pth"

    # filename regression
    regression_name = "./saved_models/regression_epoch_232_loss_0.000642.pth"

    # do training
    from modules.train import training
    training.train_MLP(filename=regression_name,lr = 5e-6, batch_size= 750, tolerance= 30)

    # do test
    # training.test_power_training()

    # evaluate model
    # from modules.models.utils import init_models, load_model
    # _, regression = init_models()
    # regression, _, _ = load_model(regression, regression_name)
    # from modules.models.test import evaluate_regression
    # while True:
    #     evaluate_regression(regression=regression)
    

if __name__ == "__main__":
    main()