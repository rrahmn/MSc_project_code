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
from torchvision.utils import make_grid, save_image
from torch.utils.data import Subset



if not os.path.exists("rendered_data_results"):
    os.makedirs("rendered_data_results")

if not os.path.exists("real_data_results"):
    os.makedirs("real_data_results")


base_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))

testing_data_directory = os.path.join(base_directory, 'Rendered_Data','Auto_generated', 'testing','images')
testing_chessboardcorners_masks_directory = os.path.join(base_directory, 'Rendered_Data','Auto_generated', 'testing','images_masks')
testing_backgrounds_masks_directory = os.path.join(base_directory, 'Rendered_Data','Auto_generated', 'testing','background_masks')


background_directory = os.path.join(base_directory, 'backgrounds')
testing_image_info_csv_path = os.path.join(base_directory, 'Rendered_Data','Auto_generated', 'testing','testing_images_info.csv')



#directory for real world images
real_data_directory = os.path.join(base_directory, 'Data')


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

            
        
        to_tensor = transforms.Compose([
                transforms.ToTensor()
                ])

        image_with_background = to_tensor(image_with_background)

        background_mask_image = to_tensor(background_mask)
        corners_mask_image = to_tensor(corners_mask)


        
        return image_with_background, background_mask_image, corners_mask_image


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
        loss = loss_background + loss_corners
        
        return loss_background.item(), loss_corners.item(), loss.item()

def overlay_predictions_on_image(image, pred_corners, pred_chessboard):
    """
    Overlay the predicted masks on top of the original image.
    """
    
    # Convert predictions to binary masks
    binary_corners = (pred_corners > 0.5).float()
    binary_chessboard = (pred_chessboard > 0.5).float()
    
    # Overlay transparent yellow chessboard
    yellow = torch.tensor([1.0, 1.0, 0.0], device=image.device).unsqueeze(-1).unsqueeze(-1)
    combined_overlay = image.clone()
    combined_overlay += binary_chessboard * yellow * 0.2
    
    # Overlay red corners on top
    red = torch.tensor([1.0, 0.0, 0.0], device=image.device).unsqueeze(-1).unsqueeze(-1)
    for c in range(3):  # Iterate through color channels
        combined_overlay[c, binary_corners[0] == 1] = red[c]
    
    # Clamp values to [0, 1] to ensure they are in valid range
    combined_overlay.clamp_(0.0, 1.0)

    return combined_overlay




color_jitter_params_grid = [
    {"brightness": 0, "contrast": 0, "saturation": 0, "hue": 0},
    {"brightness": 0.15, "contrast": 0.15, "saturation": 0.15, "hue": 0.07},
    {"brightness": 0.3, "contrast": 0.3, "saturation": 0.3, "hue": 0.15},
    {"brightness": 0.5, "contrast": 0.5, "saturation": 0.5, "hue": 0.5}
]

gaussian_blur_params_grid = [
    {"kernel_size": 3, "sigma": 0.01},
    {"kernel_size": 5, "sigma": 2},
    {"kernel_size": 7, "sigma": 4},
    {"kernel_size": 11, "sigma": 6}
]


def main():

    #read images info csv to pandas db
    testing_images_info_db = pd.read_csv(testing_image_info_csv_path, header = 0)
    
    test_dataset = ChessboardDataset(testing_data_directory, testing_backgrounds_masks_directory, background_directory, testing_chessboardcorners_masks_directory, testing_images_info_db, use_background=True, augment=True)
    evaluation_batch_size = 40
    test_loader = DataLoader(test_dataset, batch_size=evaluation_batch_size, shuffle=True, num_workers=8)

    #load  unet models
    weights_path_no_noise = os.path.join(base_directory, 'CSML_MSc_Project_Work', 'deep_learning', 'single_segmentation_unet', 'runs','experiment_2023-09-09_07-06-30', 'model_weights_epoch_3.pth')
    weights_path_with_noise = os.path.join(base_directory, 'CSML_MSc_Project_Work', 'deep_learning', 'single_segmentation_unet', 'runs','experiment_2023-09-10_02-17-19', 'model_weights_epoch_6.pth')





    #to hold results
    columns = ['Jitter Level', 'Blur Level', 'Loss 1', 'Loss 2', 'Loss 3']
    df_batch_losses_no_noise = pd.DataFrame(columns=columns)
    df_batch_losses_with_noise = pd.DataFrame(columns=columns)

    model_no_noise = UNet()
    model_no_noise.load_state_dict(torch.load(weights_path_no_noise))
    model_with_noise = UNet()
    model_with_noise.load_state_dict(torch.load(weights_path_with_noise))

    #move to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    model_no_noise = model_no_noise.to(device)
    model_with_noise = model_with_noise.to(device)



    # Evaluate on the rendered test dataset
    model_no_noise.eval()
    model_with_noise.eval()


    # Initialize arrays to store average losses for each model
    # avg_losses_no_noise_model = np.zeros((4, 4, 3))
    # avg_losses_with_noise_model = np.zeros((4, 4, 3))

    overlay_grid_no_noise = torch.zeros([4, 4, 3, 320, 320], dtype=torch.float32)
    overlay_grid_with_noise = torch.zeros([4, 4, 3, 320, 320], dtype=torch.float32)

    with torch.no_grad():
        #stepping throough blur and jitter levels
        for jitter_index, jitter_params in enumerate(color_jitter_params_grid):
            for blur_index, blur_params in enumerate(gaussian_blur_params_grid):
                # test_losses_no_noise_model = [0.0, 0.0, 0.0]
                # test_losses_with_noise_model = [0.0, 0.0, 0.0]

                jitter_transform = transforms.ColorJitter(
                        brightness=jitter_params['brightness'],
                        contrast=jitter_params['contrast'],
                        saturation=jitter_params['saturation'],
                        hue=jitter_params['hue']
                    )

                blur_transform = GaussianBlur(
                        kernel_size=blur_params['kernel_size'],
                        sigma=blur_params['sigma'])
                
                
                subset_length = int(0.3*len(test_dataset))
                subset_indices = torch.randperm(len(test_dataset))[:subset_length]
                subset_test_dataset = Subset(test_dataset, subset_indices)
                test_loader = DataLoader(subset_test_dataset, batch_size=evaluation_batch_size, shuffle=True, num_workers=8)
                
                
                for i, test_data in tqdm(enumerate(test_loader, 0), desc="Processing", total=len(test_loader)):
                    test_inputs = [d.to(device) for d in test_data]
                    test_inputs[0] = jitter_transform(blur_transform(test_inputs[0]))

                    batch_losses_no_noise_model = eval_batch(test_inputs, model_no_noise)
                    batch_losses_with_noise_model = eval_batch(test_inputs, model_with_noise)

                    #store results
                    row_no_noise = pd.Series({
                            'Jitter Level': jitter_index, 
                            'Blur Level':blur_index, 
                            'Loss 1': batch_losses_no_noise_model[0], 
                            'Loss 2': batch_losses_no_noise_model[1], 
                            'Loss 3':batch_losses_no_noise_model[2]
                        })
                    row_with_noise = pd.Series({
                            'Jitter Level': jitter_index, 
                            'Blur Level':blur_index, 
                            'Loss 1': batch_losses_with_noise_model[0], 
                            'Loss 2': batch_losses_with_noise_model[1], 
                            'Loss 3':batch_losses_with_noise_model[2]
                        })


                    df_batch_losses_no_noise = pd.concat([df_batch_losses_no_noise, pd.DataFrame([row_no_noise])], ignore_index=True)

                    df_batch_losses_with_noise = pd.concat([df_batch_losses_with_noise, pd.DataFrame([row_with_noise])], ignore_index=True)



                #visualisation saves
                images, _, _= test_inputs


                predicted_chessboard_mask_no_noise, predicted_corners_mask_no_noise = model_no_noise(images)
                predicted_chessboard_mask_with_noise, predicted_corners_mask_with_noise = model_with_noise(images)
                images = images[0]
                
                predicted_corners_mask_no_noise = predicted_corners_mask_no_noise[0]
                predicted_chessboard_mask_no_noise = predicted_chessboard_mask_no_noise[0]
                predicted_corners_mask_with_noise = predicted_corners_mask_with_noise[0]
                predicted_chessboard_mask_with_noise = predicted_chessboard_mask_with_noise[0]



                overlay_no_noise = overlay_predictions_on_image(images, predicted_corners_mask_no_noise, predicted_chessboard_mask_no_noise)
                overlay_with_noise = overlay_predictions_on_image(images, predicted_corners_mask_with_noise, predicted_chessboard_mask_with_noise)
                overlay_grid_no_noise[jitter_index, blur_index] = overlay_no_noise
                overlay_grid_with_noise[jitter_index, blur_index] = overlay_with_noise

                del test_inputs, batch_losses_no_noise_model, batch_losses_with_noise_model



                
        
        # np.save(os.path.join("rendered_data_results", "avg_losses_no_noise_model.npy"), avg_losses_no_noise_model)
        # np.save(os.path.join("rendered_data_results", "avg_losses_with_noise_model.npy"), avg_losses_with_noise_model)
        final_grid_no_noise = make_grid(overlay_grid_no_noise.view(-1, 3, 320, 320), nrow=4)
        final_grid_with_noise = make_grid(overlay_grid_with_noise.view(-1, 3, 320, 320), nrow=4)
        
        #saving the quantitative results for later
        df_batch_losses_no_noise.to_csv(os.path.join("rendered_data_results", "batch_losses_no_noise_model.csv"), index=False)
        df_batch_losses_with_noise.to_csv(os.path.join("rendered_data_results", "batch_losses_with_noise_model.csv"), index=False)

        # Save the final grids
        save_image(final_grid_no_noise, os.path.join("rendered_data_results", "final_grid_no_noise.png"))
        save_image(final_grid_with_noise, os.path.join("rendered_data_results", "final_grid_with_noise.png"))
        


        resize_transform = transforms.Resize((320, 320))
        to_tensor_transform = transforms.ToTensor()

        # Loading the real images
        image_extensions = ('.jpeg', '.jpg', '.png', '.bmp', '.tif', '.tiff')
        image_filenames = [f for f in os.listdir(real_data_directory) if f.lower().endswith(image_extensions)]

        real_images = [Image.open(os.path.join(real_data_directory, fname)) for fname in image_filenames]




        overlay_grid_real_no_noise = torch.zeros([4, 4, 3, 320, 320], dtype=torch.float32)  # Assuming the images have 3 color channels
        overlay_grid_real_with_noise = torch.zeros([4, 4, 3, 320, 320], dtype=torch.float32)

        for jitter_index, jitter_params in enumerate(color_jitter_params_grid):
            for blur_index, blur_params in enumerate(gaussian_blur_params_grid):
                random_img_idx = random.choice(range(len(real_images)))

                jitter_transform = transforms.ColorJitter(
                    brightness=jitter_params['brightness'],
                    contrast=jitter_params['contrast'],
                    saturation=jitter_params['saturation'],
                    hue=jitter_params['hue']
                )

                blur_transform = GaussianBlur(
                    kernel_size=blur_params['kernel_size'],
                    sigma=blur_params['sigma']
                )

                for img_idx, img in tqdm(enumerate(real_images), desc="Processing Real Images"):
                    # Apply transformations
                    transformed_img = to_tensor_transform(blur_transform(jitter_transform(resize_transform(img)))).unsqueeze(0).to(device)
                    predicted_chessboard_mask_no_noise, predicted_corners_mask_no_noise = model_no_noise(transformed_img)
                    predicted_chessboard_mask_with_noise, predicted_corners_mask_with_noise = model_with_noise(transformed_img)

                    # Overlay the predictions
                    overlay_no_noise = overlay_predictions_on_image(transformed_img[0], predicted_corners_mask_no_noise[0], predicted_chessboard_mask_no_noise[0])
                    overlay_with_noise = overlay_predictions_on_image(transformed_img[0], predicted_corners_mask_with_noise[0], predicted_chessboard_mask_with_noise[0])

                    # Save the overlays
                    save_image(overlay_no_noise, os.path.join("real_data_results", f'overlay_no_noise_jitter_{jitter_index}_blur_{blur_index}_img_{img_idx}.png'))
                    save_image(overlay_with_noise, os.path.join("real_data_results", f'overlay_with_noise_jitter_{jitter_index}_blur_{blur_index}_img_{img_idx}.png'))

                    # Store the overlay for the random image in the grid
                    if img_idx == random_img_idx:
                        overlay_grid_real_no_noise[jitter_index, blur_index] = overlay_no_noise
                        overlay_grid_real_with_noise[jitter_index, blur_index] = overlay_with_noise

        # Once you've populated the grid, use make_grid and save_image to save the 4x4 images
        final_grid_real_no_noise = make_grid(overlay_grid_real_no_noise.view(-1, 3, 320, 320), nrow=4)
        final_grid_real_with_noise = make_grid(overlay_grid_real_with_noise.view(-1, 3, 320, 320), nrow=4)

        # Save the final grids
        save_image(final_grid_real_no_noise, os.path.join("real_data_results", "final_grid_real_no_noise.png"))
        save_image(final_grid_real_with_noise, os.path.join("real_data_results", "final_grid_real_with_noise.png"))

                


if __name__ == '__main__':
    main()