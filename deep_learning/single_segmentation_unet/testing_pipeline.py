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
from gc import collect
import pandas as pd
import torch.nn.functional as F
from unet import UNet
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboardX import SummaryWriter
from tqdm import tqdm


base_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))

testing_data_directory = os.path.join(base_directory, 'Rendered_Data','Auto_generated', 'testing','images')
testing_chessboardcorners_masks_directory = os.path.join(base_directory, 'Rendered_Data','Auto_generated', 'testing','images_masks')
testing_backgrounds_masks_directory = os.path.join(base_directory, 'Rendered_Data','Auto_generated', 'testing','background_masks')


background_directory = os.path.join(base_directory, 'backgrounds')
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




#TRAINING/EVALUATION/METRICS

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
    testing_images_info_db = pd.read_csv(testing_image_info_csv_path, header = 0)
    
    test_dataset = ChessboardDataset(testing_data_directory, testing_backgrounds_masks_directory, background_directory, testing_chessboardcorners_masks_directory, testing_images_info_db, use_background=True, augment=False)
    evaluation_batch_size = 40
    test_loader = DataLoader(test_dataset, batch_size=evaluation_batch_size, shuffle=True, num_workers=8)

    #load  unet model
    save_path = os.path.join('runs', 'experiment_2023-09-07_18-32-43')
    weights_path = os.path.join('runs', 'experiment_2023-09-07_18-32-43', 'model_weights_epoch_3.pth')
    model = UNet()
    model.load_state_dict(torch.load(weights_path))

    #move to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    model = model.to(device)

    writer = SummaryWriter(f'runs/experiment_2023-09-07_18-32-43')

    # Evaluate on the entire test dataset
    model.eval()
    test_losses = [0.0, 0.0, 0.0]

    for i, test_data in tqdm(enumerate(test_loader, 0), desc="Processing", total=len(test_loader)):
        test_inputs = [d.to(device) for d in test_data]
        batch_losses = eval_batch(test_inputs, model)
        
        test_losses = [test_losses[j] + batch_losses[j] for j in range(3)]

    avg_test_losses = [l/len(test_loader) for l in test_losses]

    log_to_tensorboard(writer, avg_test_losses, 0, prefix="test/")

    writer.close()

if __name__ == '__main__':
    main()