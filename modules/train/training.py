'''
training
'''
# for train_MLP
from torch import nn, optim, randint, no_grad, sqrt, arange, clamp
from tqdm.auto import tqdm
from ..utils import constants as const
from ..noise.diffusion import q_sample
from ..models.model import Advanced_Regression
from ..models.IO import save_model, load_model
from ..models.loss import weighted_MSE_loss
from ..dataset.tools import prepare_dataset
from ..images.transforms import transform

# for testing model
from torch.utils.data import DataLoader
from ..dataset.test import Single_Image_Dataset
from ..dataset.init import dataset

# for evaluating performance
from ..utils.helper import record_time


def train_MLP(filename=None, num_epochs=None, layer_dim=None, lr=None, batch_size=None, tolerance=None):
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

    if layer_dim is None:
        layer_dim = const.default_layer_dim

    # init model
    model = Advanced_Regression(layer_dim=layer_dim).to(const.default_device)

    # init data and test loader for input
    if batch_size is None:
        batch_size = const.default_batch_size
    train_loader, test_loader = prepare_dataset(
        batch_size=batch_size, transform=transform)

    # Define the loss function (Mean Squared Error) and the optimizer (e.g., SGD)
    criterion = weighted_MSE_loss(const.default_MSE_weights)

    if lr is None:
        lr = const.default_learning_rate
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # load trained model
    old_epoch = 0
    if filename is not None:
        model, old_epoch, previous_loss = load_model(
            model=model, filename=filename)
    best_loss = previous_loss

    if num_epochs is None:
        num_epochs = const.default_training_epochs

    # train loop
    for epoch in tqdm(range(num_epochs), desc=f"dim: {layer_dim}; lr: {lr}"):
        print(f"layer dim: {layer_dim}. lr: {lr}")
        total_train_loss = 0.
        for step, image_samples in enumerate(train_loader):
            # in case each sample has different size
            batch_size = len(image_samples)

            # t = record_time("make batch")

            # sample random t
            t_samples = randint(0, const.timesteps, (batch_size, 1),
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
            outputs = model(noise_samples)
            # t = record_time("pass samples to model",t)

            # normalize t, because we want normalized output of regression
            t_samples = t_samples / float(const.timesteps)
            # t = record_time("normalize output",t)

            # Compute the loss
            loss = criterion(outputs, t_samples)
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
            error_timestep = round(
                sqrt(avg_train_loss * (const.timesteps ** 2)).item(), ndigits=2)
            if step % 100 == 0:
                print(
                    f"{step} steps trained. Loss in time step: {error_timestep}; True loss: {avg_train_loss}")

        print(
            f"{step} steps trained. Loss in time step: {error_timestep}; True loss: {avg_train_loss}")
        # evaluate model using test dataset in this epoch
        test_loss = 0.

        # calculate test loss and see performance
        with no_grad():
            for step, eval_samples in enumerate(test_loader):
                # in case each sample has different size
                batch_size = len(eval_samples)

                # sample random t
                t_samples = randint(0, const.timesteps, (batch_size, 1),
                                    device=const.default_device)

                # move to device
                eval_samples, t_samples = eval_samples.to(
                    const.default_device), t_samples.to(const.default_device)

                # noisy images
                eval_noise_samples = q_sample(eval_samples, t_samples)

                # Forward pass
                outputs = model(eval_noise_samples)

                # normalize t
                t_samples = t_samples / float(const.timesteps)

                # Compute the loss
                loss = criterion(outputs, t_samples)

                # accumulate loss
                test_loss += loss

        avg_test_loss = test_loss / len(test_loader)
        error_timestep = round(
            sqrt(avg_test_loss * (const.timesteps ** 2)).item(), ndigits=2)
        print(
            f"test performance loss in time step: {error_timestep}; True loss: {avg_test_loss}")
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
        if avg_test_loss < best_loss:
            save_model(model=model, layer_dim=layer_dim, lr=lr)

        # reach desired loss
        if avg_test_loss <= 6.25e-6:
            print("good job! saving and call it a day")
            break

        print(f"best loss: {best_loss}")
        previous_loss = avg_test_loss  # reset loss
        best_loss = min(avg_test_loss, best_loss)

    print('training finished')


# test the power of the model, whether model can overfit on 1 image during training


def test_power_training():
    # variables for training optimization
    previous_loss = 1024.  # TODO: find better number
    improvement_threshold = 0
    no_improvement_count = 0
    tolerance = 10

    # init model
    model = Advanced_Regression().to(const.default_device)

    # input 1 image
    test_image = transform(dataset['train']['image'][0]).to(const.default_device)

    # ? generate all noise levels of image
    batch_size = const.timesteps
    t_samples = arange(const.timesteps).view(const.timesteps, 1).to(const.default_device)
    train_dataset = Single_Image_Dataset(
        image=test_image, num_samples=const.timesteps)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    # as batch, train with Adam on MSE

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    previous_loss = 200.
    # train
    # train loop
    for epoch in tqdm(range(1000000)):
        # init previous loss
        total_loss = 0.
        avg_loss = 0.
        for step, batch in enumerate(train_loader):
            # move batch to device
            batch = batch.to(const.default_device)
            batch = batch.unsqueeze(1)

            # Forward pass
            outputs = model(batch)
            # normalize t, because we want normalized output of regression
            t_samples = t_samples / float(const.timesteps)

            # Compute the loss
            loss = criterion(outputs, t_samples)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()  # keeps track of outputs to go back to model
            optimizer.step()

            total_loss += loss
            avg_loss = total_loss / (step+1)
            # measure loss
            if epoch % 100 == 0:
                error_timestep = round(
                    sqrt(avg_loss * (const.timesteps ** 2)).item(), ndigits=2)
                print(
                    f"{step} step trained. Loss in time step: {error_timestep}; True loss: {avg_loss}")
                # compare significance of training
                # early stop, if after 5 epochs the test performance improves less than a certain threshold
                # improvement defined as 100 % when reaching 0
                # [0,1]: improvement; [0,-infty]: bad
                improvement = 1. - (avg_loss / previous_loss)
                print(
                    f"improvement: {improvement}, avg: {avg_loss}, prev:{previous_loss}")

                # early stopping
                if (improvement < improvement_threshold):
                    # no improvement
                    no_improvement_count += 1
                    print('No significant improvement')
                    if no_improvement_count >= tolerance:
                        print(
                            f'No significant improvement for {tolerance} epochs. Early stopping.')
                        break

                previous_loss = avg_loss  # reset loss
                # no storing model

        # reach desired loss
        if avg_loss <= 6.25e-6:
            print("good job! saving and call it a day")
            break

    save_model(model=model, epoch_number=epoch, loss=loss.item())
    print('training finished')
