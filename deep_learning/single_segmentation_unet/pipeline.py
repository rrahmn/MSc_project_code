import numpy as np
import torch
import torchvision
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision.transforms import GaussianBlur
import random
from random import randrange
from tqdm import tqdm
from gc import collect
from datetime import datetime
import pandas as pd
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import cv2
from pytorch3d.structures import Meshes
from unet import UNet

base_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
training_data_directory = os.path.join(base_directory, 'Rendered_Data','Auto_generated', 'training','images')
training_chessboardcorners_masks_directory = os.path.join(base_directory, 'Rendered_Data','Auto_generated', 'training','images_masks')
training_background_masks_directory = os.path.join(base_directory, 'Rendered_Data','Auto_generated', 'training','background_masks')

validation_data_directory = os.path.join(base_directory, 'Rendered_Data','Auto_generated', 'validation','images')
validation_chessboardcorners_masks_directory = os.path.join(base_directory, 'Rendered_Data','Auto_generated', 'validation','images_masks')
validation_background_masks_directory = os.path.join(base_directory, 'Rendered_Data','Auto_generated', 'validation','background_masks')

testing_data_directory = os.path.join(base_directory, 'Rendered_Data','Auto_generated', 'testing','images')
testing_chessboardcorners_masks_directory = os.path.join(base_directory, 'Rendered_Data','Auto_generated', 'testing','images_masks')
testing_backgrounds_masks_directory = os.path.join(base_directory, 'Rendered_Data','Auto_generated', 'testing','background_masks')


background_directory = os.path.join(base_directory, 'backgrounds')
training_image_info_csv_path = os.path.join(base_directory, 'Rendered_Data','Auto_generated', 'training', 'training_images_info.csv')
validation_image_info_csv_path = os.path.join(base_directory, 'Rendered_Data','Auto_generated', 'validation', 'validation_images_info.csv')
testing_image_info_csv_path = os.path.join(base_directory, 'Rendered_Data','Auto_generated', 'testing','testing_images_info.csv')


class ChessboardDataset(Dataset):
    def __init__(self, data_directory, background_masks_directory, background_directory, chessboardcorners_masks_directory, images_info_db, use_background=True, augment=True):
        self.data_directory = data_directory
        self.background_masks_directory = background_masks_directory
        self.corners_masks_directory = chessboardcorners_masks_directory
        self.augment = augment
        self.use_background = use_background
        self.background_directory = background_directory

        #database with image info
        self.images_info = images_info_db
        #indexing on image_name column
        self.images_info.set_index("image_name", inplace = True)

        self.architecture_image_size = [320,320]

        
        #folder names
        self.folders = sorted(os.listdir(data_directory))
        self.folder_lengths = [len(os.listdir(os.path.join(data_directory, folder))) for folder in self.folders]
        self.cumulative_lengths = [0] + list(np.cumsum(self.folder_lengths))

        # image paths
        self.image_paths = []
        self.background_mask_paths = []
        self.corners_mask_paths = []
        for folder in self.folders:
            self.image_paths.extend(sorted([os.path.join(self.data_directory, folder, img) for img in os.listdir(os.path.join(self.data_directory, folder))]))
            self.background_mask_paths.extend(sorted([os.path.join(self.background_masks_directory, folder, mask) for mask in os.listdir(os.path.join(self.background_masks_directory, folder))]))
            self.corners_mask_paths.extend(sorted([os.path.join(self.corners_masks_directory, folder, mask) for mask in os.listdir(os.path.join(self.corners_masks_directory, folder))]))
        
        # background images
        self.bg_images = sorted([os.path.join(background_directory, img) for img in os.listdir(background_directory)])

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        background_mask_path = self.background_mask_paths[idx]
        corners_mask_path = self.corners_mask_paths[idx]
        
        image = Image.open(image_path).convert('RGB')
        background_mask = Image.open(background_mask_path).convert('L')
        corners_mask = Image.open(corners_mask_path).convert('L')
        
        bg_image_path = np.random.choice(self.bg_images)
        bg_image = Image.open(bg_image_path).convert('RGB')

        
        if self.augment:
            #AUGMENTATIONS
            #to get different coloured chessboards
            switch = random.random()
            if(switch>0):

                #whiteish and blackish colours
                colour1 = randrange(200, 256)
                colour_replacement1 = (colour1, colour1, colour1)
                colour2 = randrange(0, 71)
                colour_replacement2 = (colour2, colour2, colour2)


                image_np = np.array(image)
                white_mask = np.all(image_np == 255, axis=2)
                image_np[white_mask] = colour_replacement1
                black_mask = np.all(image_np == 0, axis=2)
                image_np[black_mask] = colour_replacement2
                image = Image.fromarray(image_np)

                #free up memory
                del image_np
                del white_mask
                del black_mask
                collect()

        

        if self.use_background:
            b = np.random.uniform(0, 0.5)  # brightness
            c = np.random.uniform(0, 0.5)  # contrast
            s = np.random.uniform(0, 0.3)  # saturation
            h = np.random.uniform(0, 0.3) # hue

            #random background augmentations
            background_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(size=image.size, scale=(0.3, 1.0), ratio=(0.75, 1.33)),
            transforms.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h),
            ])

            bg_image = background_transform(bg_image)

            #make sure background consistent shape with image and mask
            bg_image = bg_image.resize(image.size, Image.LANCZOS)


            image_np = np.array(image)
            #for broadcasting mask
            background_mask_np = np.array(background_mask)/255
            background_mask_np = background_mask_np[:,:,np.newaxis]
            bg_image_np = np.array(bg_image)



            #get the image with the background
            image_with_background_np = ( background_mask_np == 1) * image_np + (background_mask_np == 0) * bg_image_np
            image_with_background = Image.fromarray(np.uint8(image_with_background_np))

        else:
            image_with_background = image


        del image_np
        del background_mask_np
        del bg_image_np
        del image
        del bg_image
        collect()

        if self.augment:
            augment_switch = random.random()
            if(augment_switch>0):
                ksize = random.randrange(3, 7, 2)
                sig = (0.01, 4)
                b = np.random.uniform(0, 0.3)  # brightness
                c = np.random.uniform(0, 0.3)  # contrast
                s = np.random.uniform(0, 0.3)  # saturation
                h = np.random.uniform(0, 0.15) # hue



                blur = transforms.Compose([
                    GaussianBlur(kernel_size=ksize, sigma=sig)
                    ])
                
                jitter = transforms.Compose([
                    transforms.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h),
                    ])


                image_with_background = jitter(blur(image_with_background))
        
        to_tensor = transforms.Compose([
                transforms.ToTensor()
                ])

        # image_with_background = image_with_background.convert('L') #converting to grayscale
        image_with_background = to_tensor(image_with_background)

        background_mask_image = to_tensor(background_mask)
        corners_mask_image = to_tensor(corners_mask)
        


        #LOADING FROM IMAGE INFO FILE
        fx, fy, cx, cy, rodrigues0, rodrigues1, rodrigues2, tx, ty, tz, corner1x, corner1y,corner2x, corner2y,corner3x, corner3y,corner4x, corner4y, rows, columns, square_size = self.images_info.loc[os.path.basename(image_path)]

        #ground_truth = torch.tensor([fx,fy,cx,cy, rodrigues0, rodrigues1, rodrigues2, tx, ty, tz])
        main_corners = torch.tensor([corner1x, corner1y,corner2x, corner2y,corner3x, corner3y,corner4x, corner4y])

        
        return image_with_background, background_mask_image, corners_mask_image#, main_corners


#SEGMENTATION LOSS
def dice_loss(pred, target, eps=1e-7, use_dilation=False, dilation_radius=4):
    
    if use_dilation:
        # Binarize predictions and target for dilation
        pred_binary = (pred > 0.5).float()
        target_binary = (target > 0.5).float()

        # Create a dilation kernel of 3x3 
        kernel_size = 2 * dilation_radius + 1
        kernel = torch.ones((1, 1, kernel_size, kernel_size)).to(pred.device)
        
        # If the input doesn't have a batch dimension, add one
        if len(pred_binary.shape) == 3:
            pred_binary = pred_binary.unsqueeze(0)
            target_binary = target_binary.unsqueeze(0)

        # Dilate the predictions and target
        pred_dilated = F.conv2d(pred_binary, kernel, padding=dilation_radius) > 0.5
        target_dilated = F.conv2d(target_binary, kernel, padding=dilation_radius) > 0.5
        
        # Reassign pred and target to their dilated versions
        pred, target = pred_dilated.squeeze(), target_dilated.squeeze()
    
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + eps) / (pred.sum() + target.sum() + eps)

#regression loss for 4 corners
def custom_mse_loss(pred_main_corners, main_corners):
    # Reshape to (batch, 4, 2) to convert alternating coordinates to (x,y) pairs
    

    main_corners = main_corners.view(-1, 4, 2)
    pred_main_corners = pred_main_corners.view(-1, 4, 2)
    
    # Create masks for valid points
    valid_mask = (main_corners >= 0) & (main_corners <= 320)

    # Ensure both x and y coordinates are valid
    valid_mask = valid_mask.all(dim=-1)

    # Expand the mask dimensions to match original shape for filtering
    expanded_mask = valid_mask.unsqueeze(2).expand_as(main_corners)


    # Filter points using the mask
    valid_main_corners = main_corners[expanded_mask].view(-1, 2)
    valid_pred_main_corners = pred_main_corners[expanded_mask].view(-1, 2)

    # Compute the loss only over valid points
    loss = F.mse_loss(valid_pred_main_corners/100, valid_main_corners/100)

    return loss



#TRAINING/EVALUATION/METRICS
def overlay_masks(bg_mask, pred_bg):
    """
    Overlay the predicted mask on top of the ground truth mask using different colors.
    """
    # Masks are gray; for overlay, use green for ground truth, red for predicted.
    bg_mask_colored = torch.cat([torch.zeros_like(bg_mask), bg_mask, torch.zeros_like(bg_mask)], 0)  # green
    pred_bg_colored = torch.cat([pred_bg, torch.zeros_like(pred_bg), torch.zeros_like(pred_bg)], 0)  # red
    return bg_mask_colored + pred_bg_colored  # this will give yellow where both masks overlap

def train_batch(inputs, model, optimizer, writer, current_iter, logging_freq, epoch):
    image_with_background, background_mask_image, corners_mask_image = inputs
    if epoch == 0 and current_iter == 0:
        writer.add_graph(model, image_with_background)

    optimizer.zero_grad()
    pred_backgroundmask, pred_cornersmask = model(image_with_background)
    loss_background = dice_loss(pred_backgroundmask, background_mask_image)
    loss_corners = dice_loss(pred_cornersmask, corners_mask_image)
    #loss_regression = custom_mse_loss(pred_main_corners, main_corners)
    loss = loss_background + loss_corners #+ loss_regression
    loss.backward()
    optimizer.step()

    if ((current_iter+1) % logging_freq == 0 or current_iter==0):
        grids = []
        num_images_to_plot = min(3, image_with_background.size(0))

        
        for idx in range(num_images_to_plot):
            # Ensure the predicted corners fall within the image.
            # pred_main_corners_clipped = torch.clamp(pred_main_corners[idx], 0, 319)
            # image_with_points = image_with_background[idx].clone().detach()
            # for i in range(0, len(pred_main_corners_clipped), 2):
            #     x, y = int(pred_main_corners_clipped[i].item()), int(pred_main_corners_clipped[i+1].item())
            #     # Draw a red point for each corner.
            #     image_with_points[0, y-2:y+2, x-2:x+2] = 1.0  # Red channel
            #     image_with_points[1, y-2:y+2, x-2:x+2] = 0.0  # Green channel
            #     image_with_points[2, y-2:y+2, x-2:x+2] = 0.0  # Blue channel

            bg_mask_overlay = overlay_masks(background_mask_image[idx], pred_backgroundmask[idx])
            corners_mask_overlay = overlay_masks(corners_mask_image[idx], pred_cornersmask[idx])
            

            concatenated_images = torch.cat([
                image_with_background[idx].unsqueeze(0),
                bg_mask_overlay.unsqueeze(0),
                corners_mask_overlay.unsqueeze(0)
            ], -1)
            
            grids.append(concatenated_images)

        final_grid = torch.cat(grids, 2)
        writer.add_image('combined_images', final_grid[0], global_step=current_iter)
    
    return loss_background.item(), loss_corners.item(), loss.item()




def eval_batch(inputs, model):
    with torch.no_grad():
        image_with_background, background_mask_image, corners_mask_image= inputs
        pred_backgroundmask, pred_cornersmask= model(image_with_background)
        
        loss_background = dice_loss(pred_backgroundmask, background_mask_image)
        loss_corners = dice_loss(pred_cornersmask, corners_mask_image)
        #loss_regression = custom_mse_loss(pred_main_corners, main_corners)
        loss = loss_background + loss_corners
        
        return loss_background.item(), loss_corners.item(), loss.item()

def log_to_tensorboard(writer, losses, step, prefix=""):
    loss_background, loss_corners, loss = losses
    writer.add_scalar(f'{prefix}loss_background', loss_background, step)
    writer.add_scalar(f'{prefix}loss_corners', loss_corners, step)
    writer.add_scalar(f'{prefix}total_loss', loss, step)


def main():

    #read images info csv to pandas db
    training_images_info_db = pd.read_csv(training_image_info_csv_path, header = 0)
    validation_images_info_db = pd.read_csv(validation_image_info_csv_path, header = 0)
    testing_images_info_db = pd.read_csv(testing_image_info_csv_path, header = 0)
    
    #shufffle rows
    training_images_info_db= training_images_info_db.sample(frac=1).reset_index(drop=True)


    train_dataset = ChessboardDataset(training_data_directory, training_background_masks_directory, background_directory, training_chessboardcorners_masks_directory, training_images_info_db, use_background=True, augment=True)
    validation_dataset = ChessboardDataset(validation_data_directory, validation_background_masks_directory, background_directory, validation_chessboardcorners_masks_directory, validation_images_info_db, use_background=True, augment=True)
    test_dataset = ChessboardDataset(testing_data_directory, testing_backgrounds_masks_directory, background_directory, testing_chessboardcorners_masks_directory, testing_images_info_db, use_background=True, augment=False)

    batch_size  = 12
    evaluation_batch_size = 30

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=evaluation_batch_size, shuffle=True, num_workers=8)

    #load  unet model
    model = UNet()

    #move to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    model = model.to(device)

    #loss and optimiser
    learning_rate = 1e-1
    momentum = 0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    optimizer_type = type(optimizer).__name__
    #number of epochs
    num_epochs = 7

    #creating tensorboard writer
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_name = f"{'experiment'}_{current_time}"
    writer = SummaryWriter(f'runs/{experiment_name}')


    writer.add_text(f'batch_size', str(batch_size))
    writer.add_text(f'learning_rate', str(learning_rate))
    writer.add_text(f'momentum', str(momentum))
    writer.add_text(f'epochs', str(num_epochs))
    writer.add_text(f'iterations_per_epoch', str(len(train_loader)))
    writer.add_text(f'optimizer_type', str(optimizer_type))



    logging_freq = 200
    m = 1_000  # Size of random validation subset

    for epoch in range(num_epochs):
        model.train()

        # Initialize accumulators for 3 losses + total loss
        accumulated_losses = [0.0, 0.0, 0.0]
        
        for i, data in tqdm(enumerate(train_loader, 0), desc="Processing", total=len(train_loader)):
            inputs = [d.to(device) for d in data]
            
            loss_vals = train_batch(inputs, model, optimizer, writer, epoch * len(train_loader) + i, logging_freq, epoch)

            # Accumulate the training losses and total loss for averaging
            accumulated_losses = [accumulated_losses[j] + loss_vals[j] for j in range(3)]
            
            if (i+1) % logging_freq == 0:
                # Average the accumulated losses
                avg_train_losses = [loss / logging_freq for loss in accumulated_losses]
                log_to_tensorboard(writer, avg_train_losses, epoch * len(train_loader) + i, prefix="train/")

                # Reset the accumulators for the next interval
                accumulated_losses = [0.0, 0.0, 0.0]
                
                # Generate a new random validation subset
                subset, _ = random_split(validation_dataset, [m, len(validation_dataset)-m])
                subset_loader = DataLoader(subset, batch_size=evaluation_batch_size, shuffle=True, num_workers=8)
                
                # Evaluate on validation subset
                model.eval()
                val_losses = [0.0, 0.0, 0.0]
                for val_data in subset_loader:
                    val_inputs = [d.to(device) for d in val_data]
                    batch_losses = eval_batch(val_inputs, model)
                    
                    val_losses = [val_losses[j] + batch_losses[j] for j in range(3)]
                
                avg_val_losses = [l/len(subset_loader) for l in val_losses]
                log_to_tensorboard(writer, avg_val_losses, epoch * len(train_loader) + i, prefix="val/")
                
                model.train()
            
        # Save the model weights at the end of each epoch
        torch.save(model.state_dict(), f'runs/{experiment_name}/model_weights_epoch_{epoch}.pth')

    # Evaluate on the entire test dataset
    model.eval()
    test_losses = [0.0, 0.0, 0.0]
    for test_data in test_loader:
        test_inputs = [d.to(device) for d in test_data]
        batch_losses = eval_batch(test_inputs, model)
        
        test_losses = [test_losses[j] + batch_losses[j] for j in range(3)]

    avg_test_losses = [l/len(test_loader) for l in test_losses]
    log_to_tensorboard(writer, avg_test_losses, num_epochs * len(train_loader), prefix="test/")

    writer.close()

if __name__ == '__main__':
    main()