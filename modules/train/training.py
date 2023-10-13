'''
training
'''
from torch import nn, optim, randint, no_grad, sqrt, arange
from tqdm.auto import tqdm
from ..utils.constants import default_device, timesteps, default_training_epochs, default_learning_rate, default_batch_size, default_training_tolerance
from ..noise.diffusion import q_sample
from ..models.model import Advanced_Regression
from ..dataset.mydataset import prepare_dataset
from ..models.utils import save_model, load_model
# for testing
from ..utils.helper import record_time

def train_MLP(filename = None, num_epochs = None, lr = None, batch_size = None, tolerance = None):
    '''
    -filename: name of model if to be loaded for resuming training \n
    -num_epochs: number of epochs to be trained \n
    -lr: learning rate for training \n
    -batch_size: maximal batch size in data loader \n
    -tolerance: how many failures accepted for early stopping
    '''
    # variables for training optimization
    previous_loss = 1024. # TODO: find better number
    improvement_threshold = 0
    no_improvement_count = 0
    if tolerance is None:
        tolerance = default_training_tolerance

    # init model
    model = Advanced_Regression().to(default_device)

    # init data and test loader for input
    if batch_size is None:
        batch_size = default_batch_size
    train_loader, test_loader = prepare_dataset(batch_size=batch_size)

    # Define the loss function (Mean Squared Error) and the optimizer (e.g., SGD)
    criterion = nn.MSELoss()

    if lr is None:
        lr = default_learning_rate
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # load trained model
    old_epoch = 0
    if filename is not None:
        model, old_epoch, previous_loss = load_model(model=model,filename=filename)
    best_loss = previous_loss

    if num_epochs is None:
        num_epochs = default_training_epochs
    
    # train loop
    for epoch in tqdm(range(num_epochs)):
        total_train_loss = 0.
        for step, image_samples in enumerate(train_loader):
            # in case each sample has different size
            batch_size = len(image_samples)

            # t = record_time("make batch")

            # sample random t
            t_samples = randint(0, timesteps, (batch_size, 1),
                                    device=default_device)
            # t = record_time("sample t's",t)

            # move to device
            image_samples, t_samples = image_samples.to(
                default_device), t_samples.to(default_device)
            # t = record_time("move to device",t)

            # noisy images
            noise_samples = q_sample(image_samples, t_samples)
            # t = record_time("generate q_samples",t)
            
            # Forward pass
            outputs = model(noise_samples)
            # t = record_time("pass samples to model",t)

            # normalize t, because we want normalized output of regression
            t_samples = t_samples / float(timesteps)
            # t = record_time("normalize output",t)

            # Compute the loss
            loss = criterion(outputs, t_samples)
            # t = record_time("compute loss",t)

            # move back to cpus, if in cuda
            # image_samples, t_samples = image_samples.to(
            #     'cpu'), t_samples.to('cpu')
            # t = record_time("move to cpu",t)

            # Backpropagation and optimization
            optimizer.zero_grad()
            # t = record_time("reset grad",t)
            loss.backward() # keeps track of outputs to go back to model
            # t = record_time("locate weights in model",t)
            optimizer.step()
            # t = record_time("back propagation",t)

            # update scheduler
            # scheduler.step()

            # measure loss
            total_train_loss += loss
            avg_train_loss = total_train_loss / (step+1)

            # convert MSE back into time step error
            error_timestep = round(sqrt(avg_train_loss * (timesteps ** 2)).item(),ndigits=2)
            if step % 100 == 0:
                print(f"{step} step trained. Loss in time step: {error_timestep}; True loss: {avg_train_loss}")

        print(f"{step} step trained. Loss in time step: {error_timestep}; True loss: {avg_train_loss}")
        # evaluate model using test dataset in this epoch
        test_loss = 0.

        # calculate test loss and see performance
        with no_grad():
            for step, eval_samples in enumerate(test_loader):
                # in case each sample has different size
                batch_size = len(eval_samples)

                # sample random t
                t_samples = randint(0, timesteps, (batch_size, 1),
                                        device=default_device)
                
                # move to device
                eval_samples, t_samples = eval_samples.to(
                    default_device), t_samples.to(default_device)
                
                # noisy images
                eval_noise_samples = q_sample(eval_samples, t_samples)

                # Forward pass
                outputs = model(eval_noise_samples) 

                # normalize t
                t_samples = t_samples / float(timesteps)

                # Compute the loss
                loss = criterion(outputs, t_samples)

                # accumulate loss
                test_loss += loss 

        avg_test_loss = test_loss / len(test_loader)
        error_timestep = round(sqrt(avg_test_loss * (timesteps ** 2)).item(),ndigits=2)
        print(f"test performance loss in time step: {error_timestep}; True loss: {avg_test_loss}")
        # compare significance of training
        # early stop, if after 5 epochs the test performance improves less than a certain threshold
        # improvement defined as 100 % when reaching 0
        # [0,1]: improvement; [0,-infty]: bad
        improvement = 1. - (avg_test_loss / previous_loss)
        print(f'improvement: {improvement}')

        # early stopping
        # manual adjust learning rate: no more needed, as there is scheduler
        # if test_loss < 2.5e-3:
        #   optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        # if test_loss < 2.5e-6:
        #   optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        
        if (improvement<improvement_threshold):
            # no improvement
            no_improvement_count += 1
            print('No significant improvement')
            if no_improvement_count >= tolerance:
                print(f'No significant improvement for {tolerance} epochs. Early stopping.')
                break
        
        # store best loss
        if avg_test_loss < best_loss :
            save_model(model=model,epoch_number=old_epoch+epoch,loss=avg_test_loss.item())
        
        # reach desired loss
        if avg_test_loss <= 6.25e-6:
            print("good job! saving and call it a day")
            break
        
        print(f"best loss: {best_loss}")
        previous_loss = avg_test_loss # reset loss
        best_loss = min(avg_test_loss,best_loss)
            
    print('training finished')

from torch.utils.data import DataLoader
from modules.dataset.mydataset import train_images, Single_Image_Dataset
from modules.dataset.imgtools import transform

# test the power of the model, whether model can overfit on 1 image during training
def test_power_training():
    # variables for training optimization
    previous_loss = 1024. # TODO: find better number
    improvement_threshold = 0
    no_improvement_count = 0
    tolerance = 10

    # init model
    model = Advanced_Regression().to(default_device)

    # input 1 image
    test_image = transform(train_images[0]).to(default_device)

    # ? generate all noise levels of image
    batch_size = timesteps
    t_samples = arange(timesteps).view(timesteps, 1).to(default_device)
    train_dataset = Single_Image_Dataset(
        image=test_image, num_samples=timesteps)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    

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
            batch = batch.to(default_device)
            batch = batch.unsqueeze(1)

            # Forward pass
            outputs = model(batch)
            # normalize t, because we want normalized output of regression
            t_samples = t_samples / float(timesteps)

            # Compute the loss
            loss = criterion(outputs, t_samples)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward() # keeps track of outputs to go back to model
            optimizer.step()

            total_loss += loss
            avg_loss = total_loss / (step+1)
            # measure loss
            if epoch % 100 == 0:
                error_timestep = round(sqrt(avg_loss * (timesteps ** 2)).item(),ndigits=2)
                print(f"{step} step trained. Loss in time step: {error_timestep}; True loss: {avg_loss}")
                # if loss < 2.5e-4:
                #   optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                # if loss < 2.5e-5:
                #   optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

                # compare significance of training
                # early stop, if after 5 epochs the test performance improves less than a certain threshold
                # improvement defined as 100 % when reaching 0
                # [0,1]: improvement; [0,-infty]: bad
                improvement = 1. - (avg_loss / previous_loss)
                print(f"improvement: {improvement}, avg: {avg_loss}, prev:{previous_loss}")

                # early stopping
                # manual adjust learning rate: no more needed, as there is scheduler
                # if test_loss < 2.5e-3:
                #   optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                # if test_loss < 2.5e-6:
                #   optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
                
                if (improvement<improvement_threshold):
                    # no improvement
                    no_improvement_count += 1
                    print('No significant improvement')
                    if no_improvement_count >= tolerance:
                        print(f'No significant improvement for {tolerance} epochs. Early stopping.')
                        break

                previous_loss = avg_loss # reset loss
                # no storing model
                    
        # reach desired loss
        if avg_loss <= 6.25e-6:
            print("good job! saving and call it a day")
            break

    save_model(model=model,epoch_number=epoch,loss=loss.item())
    print('training finished')
