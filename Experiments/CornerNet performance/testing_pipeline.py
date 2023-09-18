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
from architectures import CornerNet
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
real_data_directory = os.path.join(base_directory, 'MyData')





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

        # image_with_background = image_with_background.convert('L') #converting to grayscale
        image_with_background = to_tensor(image_with_background)

        


        #LOADING FROM IMAGE INFO FILE
        fx, fy, cx, cy, rodrigues0, rodrigues1, rodrigues2, tx, ty, tz, corner1x, corner1y,corner2x, corner2y,corner3x, corner3y,corner4x, corner4y, rows, columns, square_size = self.images_info.loc[os.path.basename(image_path)]

        #ground_truth = torch.tensor([fx,fy,cx,cy, rodrigues0, rodrigues1, rodrigues2, tx, ty, tz])
        main_corners = torch.tensor([corner1x, corner1y,corner2x, corner2y,corner3x, corner3y,corner4x, corner4y])

        
        return image_with_background, main_corners







#TRAINING/EVALUATION/METRICS

def eval_batch(inputs, model):
    with torch.no_grad():
        image_with_background, main_corners = inputs
        pred_main_corners = model(image_with_background)
        
        loss = torch.mean(torch.norm(main_corners - pred_main_corners, dim=1))
        
        return loss.item()
    

def draw_predicted_corners_on_image(image_with_background, pred_main_corners):
    """
    Draw predicted corners on the first image in the batch.

    Parameters:
    - image_with_background: torch.Tensor - The image tensor.
    - pred_main_corners: torch.Tensor - The predicted corner coordinates.

    Returns:
    - torch.Tensor - Image tensor with drawn corners.
    """

    pred_main_corners_clipped = torch.clamp(pred_main_corners[0], 0, 319)
    image_with_points = image_with_background[0].clone().detach()

    for i in range(0, len(pred_main_corners_clipped), 2):
        x, y = int(pred_main_corners_clipped[i].item()), int(pred_main_corners_clipped[i+1].item())

        # Draw a red point for each predicted corner.
        image_with_points[0, y-2:y+2, x-2:x+2] = 1.0  # Red channel
        image_with_points[1, y-2:y+2, x-2:x+2] = 0.0  # Green channel
        image_with_points[2, y-2:y+2, x-2:x+2] = 0.0  # Blue channel

    return image_with_points






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
    evaluation_batch_size = 20
    test_loader = DataLoader(test_dataset, batch_size=evaluation_batch_size, shuffle=True, num_workers=8)

    #load model
    weights_path = os.path.join(base_directory, 'CSML_MSc_Project_Work', 'deep_learning', 'cornerNet', "runs/experiment_2023-09-18_01-18-06/model_weights_epoch_1.pth")
    




    #to hold results
    columns = ['Jitter Level', 'Blur Level', 'Loss']
    df_batch_losses = pd.DataFrame(columns=columns)

    model = CornerNet()
    model.load_state_dict(torch.load(weights_path))


    #move to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    model = model.to(device)



    # Evaluate on the rendered test dataset
    model.eval()



    overlay_grid = torch.zeros([4, 4, 3, 320, 320], dtype=torch.float32)


    with torch.no_grad():
        #stepping throough blur and jitter levels
        # for jitter_index, jitter_params in enumerate(color_jitter_params_grid):
        #     for blur_index, blur_params in enumerate(gaussian_blur_params_grid):

        #         jitter_transform = transforms.ColorJitter(
        #                 brightness=jitter_params['brightness'],
        #                 contrast=jitter_params['contrast'],
        #                 saturation=jitter_params['saturation'],
        #                 hue=jitter_params['hue']
        #             )

        #         blur_transform = GaussianBlur(
        #                 kernel_size=blur_params['kernel_size'],
        #                 sigma=blur_params['sigma'])
                
                
        #         subset_length = int(0.3*len(test_dataset))
        #         subset_indices = torch.randperm(len(test_dataset))[:subset_length]
        #         subset_test_dataset = Subset(test_dataset, subset_indices)
        #         test_loader = DataLoader(subset_test_dataset, batch_size=evaluation_batch_size, shuffle=True, num_workers=8)
                
                
        #         for i, test_data in tqdm(enumerate(test_loader, 0), desc="Processing", total=len(test_loader)):
        #             test_inputs = [d.to(device) for d in test_data]
        #             test_inputs[0] = jitter_transform(blur_transform(test_inputs[0]))

        #             batch_losses = eval_batch(test_inputs, model)


        #             #store results
        #             row = pd.Series({
        #                     'Jitter Level': jitter_index, 
        #                     'Blur Level':blur_index, 
        #                     'Loss': batch_losses, 

        #                 })



        #             df_batch_losses = pd.concat([df_batch_losses, pd.DataFrame([row])], ignore_index=True)




        #         #visualisation saves
        #         images, _,= test_inputs


        #         pred_main_corners = model(images)

        #         # images = images[0]
                
        #         # pred_main_corners_no_noise = pred_main_corners_no_noise[0]
        #         # pred_main_corners_with_noise = pred_main_corners_with_noise[0]



        #         overlay = draw_predicted_corners_on_image(images, pred_main_corners)
        #         overlay_grid[jitter_index, blur_index] = overlay




                
        

        # final_grid = make_grid(overlay_grid.view(-1, 3, 320, 320), nrow=4)

        
        # #saving the quantitative results for later
        # df_batch_losses.to_csv(os.path.join("rendered_data_results", "batch_losses.csv"), index=False)


        # # Save the final grids
        # save_image(final_grid, os.path.join("rendered_data_results", "final_grid.png"))
        


        resize_transform = transforms.Resize((320, 320))
        to_tensor_transform = transforms.ToTensor()

        # Loading the real images
        image_extensions = ('.jpeg', '.jpg', '.png', '.bmp', '.tif', '.tiff')
        image_filenames = [f for f in os.listdir(real_data_directory) if f.lower().endswith(image_extensions)]

        real_images = [Image.open(os.path.join(real_data_directory, fname)) for fname in image_filenames]




        overlay_grid_real = torch.zeros([4, 4, 3, 320, 320], dtype=torch.float32)  # Assuming the images have 3 color channels

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
                    pred_main_corners = model(transformed_img)


                    # Overlay the predictions
                    overlay = draw_predicted_corners_on_image(transformed_img, pred_main_corners)



                    # Save the overlays
                    save_image(overlay, os.path.join("real_data_results", f'overlay_jitter_{jitter_index}_blur_{blur_index}_img_{img_idx}.png'))


                    # Store the overlay for the random image in the grid
                    if img_idx == random_img_idx:
                        overlay_grid_real[jitter_index, blur_index] = overlay


        # Once you've populated the grid, use make_grid and save_image to save the 4x4 images
        final_grid_real = make_grid(overlay_grid_real.view(-1, 3, 320, 320), nrow=4)


        # Save the final grids
        save_image(final_grid_real, os.path.join("real_data_results", "final_grid_real_no_noise.png"))


                


if __name__ == '__main__':
    main()