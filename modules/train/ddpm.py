

from tqdm.auto import tqdm
from torch import nn, optim, no_grad, randint, clamp
from ..utils import constants as const
from ..noise.diffusion import q_sample
from ..models.model import Unet
from ..models.IO import save_model, load_model
from ..dataset.tools import prepare_dataset
from ..images.transforms import transform

def train_diffusion(filename=None, num_epochs=None, lr=None, batch_size=None, tolerance=None):
    '''
    -filename: name of model if to be loaded for resuming training \n
    -num_epochs: number of epochs to be trained \n
    -lr: learning rate for training \n
    -batch_size: maximal batch size in data loader \n
    -tolerance: how many failures accepted for early stopping
    '''
    # variables for training optimization
    previous_loss = 1024.  # TODO: find better number
    improvement_threshold = 0
    no_improvement_count = 0
    if tolerance is None:
        tolerance = const.default_training_tolerance

    # init model
    model = Unet(
    dim=const.image_size,
    channels=3,
    dim_mults=(1, 2, 4,)
    ).to(const.default_device)

    # init data and test loader for input
    if batch_size is None:
        batch_size = const.default_batch_size
    train_loader, test_loader = prepare_dataset(
        batch_size=batch_size, transform=transform)

    # Define the loss function (Mean Squared Error) and the optimizer (e.g., SGD)
    criterion = nn.MSELoss()
    # criterion = weighted_MSE_loss(const.default_MSE_weights)

    if lr is None:
        lr = const.default_learning_rate
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # load trained model
    old_epoch = 0
    if filename is not None:
        model = load_model(
            model=model, filename=filename)
    best_loss = previous_loss

    if num_epochs is None:
        num_epochs = const.default_training_epochs

    # train loop
    for epoch in tqdm(range(num_epochs), desc=f"diffusion; lr: {lr}"):
        total_train_loss = 0.
        for step, image_samples in enumerate(train_loader):
            # in case each sample has different size
            batch_size = len(image_samples)

            # t = record_time("make batch")

            # sample random t
            t_samples = randint(0, const.timesteps, (batch_size,),
                                device=const.default_device)
            # t = record_time("sample t's",t)

            # move to device
            image_samples, t_samples = image_samples.to(
                const.default_device), t_samples.to(const.default_device)
            # t = record_time("move to device",t)

            # noisy images
            # clamp [-1,1]
            noise_samples = clamp(q_sample(image_samples, t_samples),min=-1,max=1)
            # t = record_time("generate q_samples",t)

            # Forward pass
            predict_denoise = model(noise_samples, t_samples)
            # t = record_time("pass samples to model",t)
            
            # Compute the loss
            loss = criterion(predict_denoise, noise_samples)
            # t = record_time("compute loss",t)

            # Backpropagation and optimization
            optimizer.zero_grad()
            # t = record_time("reset grad",t)
            loss.backward()  # keeps track of outputs to go back to model
            # t = record_time("locate weights in model",t)
            optimizer.step()
            # t = record_time("back propagation",t)

            # measure loss
            total_train_loss += loss
            avg_train_loss = total_train_loss / (step+1)

            # convert MSE back into time step error
            if step % 100 == 0:
                print(
                    f"{step} steps trained. Loss: {avg_train_loss}")

        print(
            f"{step} steps trained. Loss: {avg_train_loss}")
        # evaluate model using test dataset in this epoch
        test_loss = 0.

        # calculate test loss and see performance
        for step, eval_samples in enumerate(test_loader):
            # in case each sample has different size
            batch_size = len(eval_samples)

            # sample random t
            t_samples = randint(0, const.timesteps, (batch_size,),
                                device=const.default_device)

            # move to device
            eval_samples, t_samples = eval_samples.to(
                const.default_device), t_samples.to(const.default_device)

            # noisy images
            eval_noise_samples = clamp(q_sample(eval_samples, t_samples),min=-1,max=1)

            # Forward pass
            with no_grad():
                predict_denoise = model(eval_noise_samples, t_samples)

            # Compute the loss
            loss = criterion(predict_denoise, eval_noise_samples)

            # accumulate loss
            test_loss += loss

        avg_test_loss = test_loss / len(test_loader)
        print(
            f"test performance loss: {avg_test_loss}")
        # compare significance of training
        # early stop, if after 5 epochs the test performance improves less than a certain threshold
        # improvement defined as 100 % when reaching 0
        # [0,1]: improvement; [0,-infty]: bad
        improvement = 1. - (avg_test_loss / previous_loss)
        print(f'improvement: {improvement}')

        # early stopping
        if (improvement < improvement_threshold):
            # no improvement
            no_improvement_count += 1
            print('No significant improvement')
            if no_improvement_count >= tolerance:
                print(
                    f'No significant improvement for {tolerance} epochs. Early stopping.')
                break

        # store best loss
        if epoch != 0 and avg_test_loss < best_loss:
            save_model(model=model, lr=lr)

        # reach desired loss
        if avg_test_loss <= 6.25e-6:
            print("good job! saving and call it a day")
            break

        print(f"best loss: {best_loss}")
        previous_loss = avg_test_loss  # reset loss
        best_loss = min(avg_test_loss, best_loss)

    print('training finished')