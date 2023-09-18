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
import argparse
import pandas as pd

from tensorboardX import SummaryWriter
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform, Textures, RasterizationSettings, MeshRenderer, MeshRasterizer, 
    HardPhongShader, BlendParams
)
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.transforms import so3_exp_map

import cv2
from architecture import CombinedModel


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

        ground_truth = torch.tensor([fx,fy,cx,cy, rodrigues0, rodrigues1, rodrigues2, tx, ty, tz])

        
        return square_size, image_with_background, ground_truth, rows, columns



#REPROJECTION ERROR LOSS FUNC
def get_camera_parameters_from_tensors(params_batch):
    """
    Convert a batch of torch tensors containing fx, fy, cx, cy, 
    rodrigues1, rodrigues2, rodrigues3, tx, ty, tz into intrinsic and extrinsic parameters.
    """
    device = params_batch.device  # get the device of the input tensor

    fx, fy, cx, cy = params_batch[:, 0], params_batch[:, 1], params_batch[:, 2], params_batch[:, 3]
    rodrigues_params = params_batch[:, 4:7]
    tvec = params_batch[:, 7:]

    # Convert Rodrigues parameters to rotation matrices
    R = so3_exp_map(rodrigues_params)

    # Construct camera matrix (intrinsic parameters)
    batch_size = params_batch.shape[0]
    zeros = torch.zeros((batch_size,)).to(device)  # set to the correct device
    ones = torch.ones((batch_size,)).to(device)  # set to the correct device
    
    camera_matrix = torch.stack([fx, zeros, cx, zeros, fy, cy, zeros, zeros, ones], dim=-1).reshape(batch_size, 3, 3)

    return camera_matrix, R, tvec


def mape_loss(output, target, eps=1e-8):
    return torch.mean(torch.abs((target - output) / (target + eps))) * 100


def projection_losses(corners, predicted, ground):
    corners = corners.float()
    predicted = predicted.float()
    ground = ground.float()
    image_size = torch.tensor([[320, 320]], dtype=torch.float32)
    true_camera, trueR, trueT = get_camera_parameters_from_tensors(ground)


    pred_camera, predR, predT = get_camera_parameters_from_tensors(predicted)
    cameras_ground = cameras_from_opencv_projection(
        R=trueR,
        tvec=trueT,
        camera_matrix=true_camera,
        image_size=image_size
    )

    cameras_pred = cameras_from_opencv_projection(
        R=predR,
        tvec=predT,
        camera_matrix=pred_camera,
        image_size=image_size
    )

    true_projected_corners_with_depth = cameras_ground.transform_points_screen(corners)
    true_projected_corners = true_projected_corners_with_depth[:,:,:2]


    pred_projected_corners_with_depth = cameras_pred.transform_points_screen(corners)
    pred_projected_corners = pred_projected_corners_with_depth[:,:,:2]

    #2d error
    pairwise_distances = torch.norm(true_projected_corners - pred_projected_corners, dim=2)
    avg_distance = torch.mean(pairwise_distances, dim=1)

    twod_error = torch.mean(avg_distance)

    #3d error
    projected_corners_with_depth_ndc_space = cameras_ground.transform_points(corners)
    estimated_threed_coordinates = cameras_pred.unproject_points(projected_corners_with_depth_ndc_space, world_coordinates=True, from_ndc=True)
    true_threed_coordinates = cameras_ground.unproject_points(projected_corners_with_depth_ndc_space, world_coordinates=True, from_ndc=True)
    threed_error = torch.mean(torch.norm(true_threed_coordinates - estimated_threed_coordinates, dim=2))


    return twod_error, threed_error

def selected_loss_function(twod_projection_error, threed_projection_error, mape, whichloss):
    """Return the specified loss based on whichloss argument"""
    if whichloss == "mape":
        return mape
    elif whichloss == "twod":
        return twod_projection_error
    elif whichloss == "threed":
        return threed_projection_error
    elif whichloss == "projection_combined":
        return (twod_projection_error + threed_projection_error)/2
    else:
        raise ValueError(f"Invalid loss name: {whichloss}")




#chessboard vertices for reprojection error
def generate_vertices(square_sizes, rows, columns):
    
    device = square_sizes.device  # Gets the device of the 'square_sizes' tensor
    
    batch_size = square_sizes.size(0)
    max_rows = int(rows.max().item())
    max_columns = int(columns.max().item())
    
    square_sizes_exp = square_sizes.view(batch_size, 1, 1).expand(batch_size, max_rows, max_columns)
    
    # Generating coordinates
    row_coords = torch.arange(max_rows, dtype=torch.float32, device=device).view(1, max_rows, 1).expand(batch_size, max_rows, max_columns)
    col_coords = torch.arange(max_columns, dtype=torch.float32, device=device).view(1, 1, max_columns).expand(batch_size, max_rows, max_columns)
    
    vertices_x = row_coords * square_sizes_exp
    vertices_y = col_coords * square_sizes_exp
    vertices_z = torch.zeros_like(vertices_x)  # Z-coordinates set to 0
    
    vertices = torch.stack((vertices_x, vertices_y, vertices_z), dim=3)
    
    # Reshape to [batch_size, points, 3]
    vertices = vertices.view(batch_size, -1, 3)
    
    return vertices


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#RENDERING FUCNTIONS
def create_3d_chessboard(r, c, sq=1.0):
    """ function to create chessboard mesh
        input: chessboard rows r, chessboard columns c and square size sq
        output: chessboard mesh, and all corner coordinates
     """
    verts, faces, vert_colors = [], [], []

    corners_list = []
    corner_set = set()
    
    for i in range(r):
        for j in range(c):
            z_offset = 0.0 
            square_verts = [
                [i*sq, j*sq, z_offset],
                [(i+1)*sq, j*sq, z_offset],
                [(i+1)*sq, (j+1)*sq, z_offset],
                [i*sq, (j+1)*sq, z_offset]
            ]
            verts.extend(square_verts)

            for sv in square_verts:
                corner_tuple = tuple(sv)
                if corner_tuple not in corner_set:
                    corner_set.add(corner_tuple)
                    corners_list.append(sv)
            
            color = [2.0, 2.0, 2.0] if (i+j) % 2 == 0 else [0.0, 0.0, 0.0]
            vert_colors.extend([color, color, color, color])
            
            base_idx = len(verts) - 4
            square_faces = [
                [base_idx, base_idx+1, base_idx+2],
                [base_idx, base_idx+3, base_idx+2]
            ]
            faces.extend(square_faces)

    main_corners = [
        [0, 0, 0],                    # Top-left
        [0, c * sq, 0],               # Top-right
        [r * sq, 0, 0],               # Bottom-left
        [r * sq, c * sq, 0]           # Bottom-right
    ]
    for mc in main_corners:
        corners_list.remove(mc)


    inlying_corners = main_corners + [corner for corner in corners_list if 0 < corner[0] < r*sq and 0 < corner[1] < c*sq]

    # Convert list to torch tensor of appropriate shape
    corners = torch.tensor([inlying_corners], dtype=torch.float32)  # Shape: [1, number_of_corners, 3]
    
    verts = torch.tensor(verts, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.int64)
    vert_colors = torch.tensor(vert_colors, dtype=torch.float32).unsqueeze(0)
    textures = Textures(verts_rgb=vert_colors)
    meshes = Meshes(verts=[verts], faces=[faces], textures=textures)

    return meshes, corners


def render_mesh(mesh, camera_matrix, R, T, bg=(0.0, 1.0, 0.0)):
    "function to render a bunch of images of the chessboard given a batch of intrinsic and extrinsic parameters"
    image_size = torch.tensor([[320, 320]])
    #creating a batch of PerspectiveCameras using the provided function
    cameras = cameras_from_opencv_projection(
        R=R,
        tvec=T,
        camera_matrix=camera_matrix,
        image_size=image_size
    )
    raster_settings = RasterizationSettings(image_size=(320, 320))
    blend_params = BlendParams(background_color=bg)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras, blend_params=blend_params)
        #shader = SoftPhongShader(device=device, cameras=cameras, blend_params=blend_params)
    ).to(device)

    return renderer(mesh)


def train_batch(inputs, model, optimizer, writer, current_iter, logging_freq, epoch, training_stds, training_means, whichloss):
    square_size, image_with_background, ground_truth, rows, columns = inputs
    if epoch == 0 and current_iter == 0:
        writer.add_graph(model, (image_with_background, square_size))

    optimizer.zero_grad()
    prediction = model(image_with_background, square_size)

    chessboard_verts = generate_vertices(square_size, rows, columns)
    

   
    twod_projection_error, threed_projection_error = projection_losses(chessboard_verts, training_stds*prediction + training_means, training_stds*ground_truth + training_means)
 
    mape = mape_loss(training_stds*prediction + training_means, training_stds*ground_truth + training_means)


    loss = selected_loss_function(twod_projection_error, threed_projection_error, mape, whichloss)
    print("#################################")
    print(loss)
    print(mape)
    print(twod_projection_error)
    print(threed_projection_error)
    print("#################################")
    loss.backward()
    optimizer.step()

    
    with torch.no_grad():
        if ((current_iter+1) % logging_freq == 0 or current_iter==0):
            grids = []
            num_images_to_plot = min(3, image_with_background.size(0))

            
            
            for idx in range(num_images_to_plot):
            #RENDER CHESSBOARD WITH FOUND INTRINSICS AND TRUE EXTRINSICS
                chessboard_mesh,_ = create_3d_chessboard(int(rows[idx].item()), int(columns[idx].item()), sq=square_size[idx].item())
                rodrigues = (training_stds*prediction + training_means)[idx, 4:7].cpu().numpy()
                R, _ = cv2.Rodrigues(rodrigues)
                T = (training_stds*prediction + training_means)[idx, 7:]
                R = torch.tensor(R, dtype=torch.float32)
                camera_matrix = torch.eye(3, dtype=torch.float32).unsqueeze(0)
                camera_matrix[:, 0, 0] = (training_stds*prediction + training_means)[idx, 0]
                camera_matrix[:, 1, 1] =(training_stds*prediction + training_means)[idx, 1]
                camera_matrix[:, 0, 2] = (training_stds*prediction + training_means)[idx, 2]
                camera_matrix[:, 1, 2] = (training_stds*prediction + training_means)[idx, 3]
                chessboard_mesh = chessboard_mesh.to(device)
                camera_matrix = camera_matrix.to(device)
                R = R.unsqueeze(0)
                T = T.unsqueeze(0)
                R = R.to(device)
                T = T.to(device)

                rendered_image = render_mesh(chessboard_mesh, camera_matrix, R, T)
                rendered_image = rendered_image[:,:,:, :3]
                rendered_image = rendered_image.permute(0, 3, 1, 2)
                

                grid_for_sample = torch.cat([
                                rendered_image,
                                image_with_background[idx].unsqueeze(0),
                            ], -1)  # concatenate over width
                grids.append(grid_for_sample)

            final_grid = torch.cat(grids, 2)
            writer.add_image('combined_images', final_grid[0], global_step=current_iter)
    
    return [twod_projection_error.item(), threed_projection_error.item(), mape.item(), loss.item()]


def eval_batch(inputs, model, training_stds, training_means, whichloss):
    with torch.no_grad():
        square_size, image_with_background, ground_truth, rows, columns = inputs
        prediction = model(image_with_background, square_size)
        
        chessboard_verts = generate_vertices(square_size, rows, columns)
        twod_projection_error, threed_projection_error = projection_losses(chessboard_verts, training_stds*prediction + training_means, training_stds*ground_truth + training_means)
        mape = mape_loss(training_stds*prediction + training_means, training_stds*ground_truth + training_means)
        loss = selected_loss_function(twod_projection_error, threed_projection_error, mape, whichloss)
        return [twod_projection_error.item(), threed_projection_error.item(), mape.item(), loss.item()]

def log_to_tensorboard(writer, losses, step, prefix=""):
    twod_projection_error, threed_projection_error, mape, loss = losses

    # Logging total loss
    writer.add_scalar(f'{prefix}total_loss', loss, step)

    # Logging 2D projection error
    writer.add_scalar(f'{prefix}twod_projection_error', twod_projection_error, step)

    # Logging 3D projection error
    writer.add_scalar(f'{prefix}threed_projection_error', threed_projection_error, step)

    # Logging mean squared error
    writer.add_scalar(f'{prefix}mape', mape, step)

def main():
    parser = argparse.ArgumentParser(description='Select model and loss settings for training.')
    parser.add_argument('--resnet', choices=['resnet18', 'resnet50', 'resnet152'], required=True, 
                        help='Choices: resnet18, resnet50, resnet152')
    parser.add_argument('--use_unet', action='store_true', help='Use UNet model?')
    parser.add_argument('--loss', choices=['mape', 'twod', 'threed', 'projection_combined'], required=True, 
                        help='Choices: mape, twod, threed')
    args = parser.parse_args()





    #read images info csv to pandas db
    training_images_info_db = pd.read_csv(training_image_info_csv_path, header = 0)
    validation_images_info_db = pd.read_csv(validation_image_info_csv_path, header = 0)
    testing_images_info_db = pd.read_csv(testing_image_info_csv_path, header = 0)
    
    #shufffle rows
    training_images_info_db= training_images_info_db.sample(frac=1).reset_index(drop=True)


    #normalising ground truths
    training_means = training_images_info_db.iloc[:, 1:11].mean()
    training_stds = training_images_info_db.iloc[:, 1:11].std()
    training_images_info_db.iloc[:, 1:11] = (training_images_info_db.iloc[:, 1:11] - training_means) / training_stds
    validation_images_info_db.iloc[:, 1:11] = (validation_images_info_db.iloc[:, 1:11] - training_means) / training_stds
    testing_images_info_db.iloc[:, 1:11] = (testing_images_info_db.iloc[:, 1:11] - training_means) / training_stds

    train_dataset = ChessboardDataset(training_data_directory, training_background_masks_directory, background_directory, training_chessboardcorners_masks_directory, training_images_info_db, use_background=True, augment=True)
    validation_dataset = ChessboardDataset(validation_data_directory, validation_background_masks_directory, background_directory, validation_chessboardcorners_masks_directory, validation_images_info_db, use_background=True, augment=True)
    test_dataset = ChessboardDataset(testing_data_directory, testing_backgrounds_masks_directory, background_directory, testing_chessboardcorners_masks_directory, testing_images_info_db, use_background=True, augment=False)

    batch_size  = 30
    evaluation_batch_size = 30

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)


    weights_path = os.path.join(os.path.dirname(os.getcwd()), 'single_segmentation_unet', 'runs', 'experiment_2023-09-10_02-17-19', 'model_weights_epoch_6.pth')
    unet_weights = torch.load(weights_path)

    model = CombinedModel(resnet_variant=args.resnet, use_unet=args.use_unet)

    # Load the pretrained weights into UNet only if use_unet is true
    if args.use_unet:
        unet_weights = torch.load(weights_path)
        model.unet.load_state_dict(unet_weights)

    #move to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    training_stds = torch.tensor(training_stds, dtype=torch.float32).to(device)
    training_means = torch.tensor(training_means, dtype=torch.float32).to(device)
    model = model.to(device)

    #loss and optimiser
    learning_rate = 1e-4
    momentum = 0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    optimizer_type = type(optimizer).__name__
    #number of epochs
    num_epochs = 10

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
    writer.add_text('resnet_variant', args.resnet)
    writer.add_text('use_unet', str(args.use_unet))
    writer.add_text('loss', args.loss)


    
    
    logging_freq = 400
    m = 2_000  # Size of random validation subset

    for epoch in range(num_epochs):
        model.train()

        # Initialize accumulators for 4 losses
        accumulated_losses = [0.0, 0.0, 0.0, 0.0]
        
        for i, data in tqdm(enumerate(train_loader, 0), desc="Processing", total=len(train_loader)):
            inputs = [d.to(device) for d in data]
            
            loss_vals = train_batch(inputs, model, optimizer, writer, epoch * len(train_loader) + i, logging_freq, epoch, training_stds, training_means, args.loss)

            # Accumulate the training losses and total loss for averaging
            accumulated_losses = [accumulated_losses[j] + loss_vals[j] for j in range(4)]
            
            if (i+1) % logging_freq == 0:
                # Average the accumulated losses
                avg_train_losses = [loss / logging_freq for loss in accumulated_losses]
                log_to_tensorboard(writer, avg_train_losses, epoch * len(train_loader) + i, prefix="train/")

                # Reset the accumulators for the next interval
                accumulated_losses = [0.0, 0.0, 0.0, 0.0]
                
                # Generate a new random validation subset
                subset, _ = random_split(validation_dataset, [m, len(validation_dataset)-m])
                subset_loader = DataLoader(subset, batch_size=evaluation_batch_size, shuffle=True, num_workers=8)
                
                # Evaluate on validation subset
                model.eval()
                val_losses = [0.0, 0.0, 0.0, 0.0]
                for val_data in subset_loader:
                    val_inputs = [d.to(device) for d in val_data]
                    batch_losses = eval_batch(val_inputs, model, training_stds, training_means, args.loss)
                    
                    val_losses = [val_losses[j] + batch_losses[j] for j in range(4)]
                
                avg_val_losses = [l/len(subset_loader) for l in val_losses]
                log_to_tensorboard(writer, avg_val_losses, epoch * len(train_loader) + i, prefix="val/")
                
                model.train()
            
        # Save the model weights at the end of each epoch
        torch.save(model.state_dict(), f'runs/{experiment_name}/model_weights_epoch_{epoch}.pth')

    # Evaluate on the entire test dataset
    model.eval()
    test_losses = [0.0, 0.0, 0.0, 0.0]
    for test_data in test_loader:
        test_inputs = [d.to(device) for d in test_data]
        batch_losses = eval_batch(test_inputs, model, training_stds, training_means, args.loss)
        
        test_losses = [test_losses[j] + batch_losses[j] for j in range(4)]

    avg_test_losses = [l/len(test_loader) for l in test_losses]
    log_to_tensorboard(writer, avg_test_losses, num_epochs * len(train_loader), prefix="test/")

    writer.close()

if __name__ == '__main__':
    main()