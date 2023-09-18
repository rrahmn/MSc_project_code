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
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
from torch.utils.data import Subset
import cv2
import pandas as pd
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.transforms import so3_exp_map
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform, Textures, RasterizationSettings, MeshRenderer, MeshRasterizer, 
    HardPhongShader, BlendParams
)
from architectures import CornerNet

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
    def __init__(self, data_directory, background_masks_directory, background_directory, chessboardcorners_masks_directory, 
                 images_info_db, augment_colours = False, use_background=True, augment=True, jitter_params=None, blur_params=None,num_images=15):
        self.data_directory = data_directory
        self.background_masks_directory = background_masks_directory
        self.corners_masks_directory = chessboardcorners_masks_directory
        self.augment = augment
        self.augment_colours = augment_colours
        self.use_background = use_background
        self.background_directory = background_directory
        self.images_info = images_info_db
        self.images_info.set_index("image_name", inplace=True)
        self.architecture_image_size = [320, 320]
        self.folders = sorted(os.listdir(data_directory))
        self.num_images = num_images
        # image paths grouped by folder
        self.image_paths_by_folder = []
        self.background_mask_paths_by_folder = []
        self.corners_mask_paths_by_folder = []
        for folder in self.folders:
            self.image_paths_by_folder.append(sorted([os.path.join(self.data_directory, folder, img) for img in os.listdir(os.path.join(self.data_directory, folder))]))
            self.background_mask_paths_by_folder.append(sorted([os.path.join(self.background_masks_directory, folder, mask) for mask in os.listdir(os.path.join(self.background_masks_directory, folder))]))
            self.corners_mask_paths_by_folder.append(sorted([os.path.join(self.corners_masks_directory, folder, mask) for mask in os.listdir(os.path.join(self.corners_masks_directory, folder))]))
        self.bg_images = sorted([os.path.join(background_directory, img) for img in os.listdir(background_directory)])

        self.jitter_params = jitter_params or {"brightness": 0, "contrast": 0, "saturation": 0, "hue": 0}
        self.blur_params = blur_params or {"kernel_size": 3, "sigma": 0.01}
        self.jitter_transform = transforms.ColorJitter(
            brightness=self.jitter_params["brightness"],
            contrast=self.jitter_params["contrast"],
            saturation=self.jitter_params["saturation"],
            hue=self.jitter_params["hue"]
        )
        self.blur_transform = GaussianBlur(
            kernel_size=self.blur_params["kernel_size"],
            sigma=self.blur_params["sigma"]
        )

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        
        selected_indices = np.random.choice(len(self.image_paths_by_folder[idx]), self.num_images, replace=False)
        
        image_paths = [self.image_paths_by_folder[idx][i] for i in selected_indices]
        background_mask_paths = [self.background_mask_paths_by_folder[idx][i] for i in selected_indices]
        corners_mask_paths = [self.corners_mask_paths_by_folder[idx][i] for i in selected_indices]

        images = []
        background_masks = []
        corners_masks = []
        images_with_backgrounds = []
        
        for i in range(self.num_images):
            image = Image.open(image_paths[i]).convert('RGB')
            background_mask = Image.open(background_mask_paths[i]).convert('L')
            corners_mask = Image.open(corners_mask_paths[i]).convert('L')
            images.append(image)
            background_masks.append(background_mask)
            corners_masks.append(corners_mask)

            bg_image_path = np.random.choice(self.bg_images)
            bg_image = Image.open(bg_image_path).convert('RGB')
            if self.augment_colours:
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
            if self.use_background:
                b = np.random.uniform(0, 0.5)
                c = np.random.uniform(0, 0.5)
                s = np.random.uniform(0, 0.3)
                h = np.random.uniform(0, 0.3)
                background_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(20),
                    transforms.RandomResizedCrop(size=image.size, scale=(0.3, 1.0), ratio=(0.75, 1.33)),
                    transforms.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h),
                ])
                bg_image = background_transform(bg_image)
                bg_image = bg_image.resize(image.size, Image.LANCZOS)
                image_np = np.array(image)
                background_mask_np = np.array(background_mask) / 255
                background_mask_np = background_mask_np[:, :, np.newaxis]
                bg_image_np = np.array(bg_image)
                image_with_background_np = (background_mask_np == 1) * image_np + (background_mask_np == 0) * bg_image_np
                image_with_background = Image.fromarray(np.uint8(image_with_background_np))
                to_tensor = transforms.Compose([
                transforms.ToTensor()
                ])
                if self.augment:
                    image_with_background = self.jitter_transform(self.blur_transform(image_with_background))
                image_with_background = to_tensor(image_with_background)
                images_with_backgrounds.append(image_with_background)
            else:
                to_tensor = transforms.Compose([
                transforms.ToTensor()
                ])
                image_with_background = to_tensor(image_with_background)
                images_with_backgrounds.append(image_with_background)
                #images_with_backgrounds.append(image)
            

            

        
        # Adjusting the part for image info
        ground_truths = []
        for i in range(self.num_images):
            fx, fy, cx, cy, rodrigues0, rodrigues1, rodrigues2, tx, ty, tz, _, _,_, _,_, _,_, _, rows, columns, square_size = self.images_info.loc[os.path.basename(image_paths[i])]
            ground_truths.append(torch.tensor([fx,fy,cx,cy, rodrigues0, rodrigues1, rodrigues2, tx, ty, tz]))

        return square_size, images_with_backgrounds, ground_truths, rows, columns
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
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


def render_mesh(mesh, camera_matrix, R, T, bg=(0.0, 1.0, 0.0), size=[320,320]):
    "function to render a bunch of images of the chessboard given a batch of intrinsic and extrinsic parameters"
    image_size = torch.tensor([[size[1], size[0]]])
    #creating a batch of PerspectiveCameras using the provided function
    cameras = cameras_from_opencv_projection(
        R=R,
        tvec=T,
        camera_matrix=camera_matrix,
        image_size=image_size
    )
    raster_settings = RasterizationSettings(image_size=(size[1], size[0]))
    blend_params = BlendParams(background_color=bg)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras, blend_params=blend_params)
        #shader = SoftPhongShader(device=device, cameras=cameras, blend_params=blend_params)
    ).to(device)

    return renderer(mesh)


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

    true_projected_corners_with_depth = cameras_ground.transform_points_screen(corners,with_xyflip=True)
    true_projected_corners = true_projected_corners_with_depth[:,:,:2]



    pred_projected_corners_with_depth = cameras_pred.transform_points_screen(corners,with_xyflip=True)
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



criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
#calibration funcs
def calibrate_camera_with_corner_net(images, square_size, rows, columns, model):
    rows = int(rows.item())
    columns = int(columns.item())
    square_size = square_size.item()
    top_left = [rows, 0, 0]
    top_right = [0, 0, 0]
    bottom_left = [rows, columns, 0]
    bottom_right = [0, columns, 0]

    corners = [top_right, bottom_right, top_left, bottom_left]
    objp = np.array(corners, dtype=np.float32) * square_size

    # objp[:,:2] = np.mgrid[1:rows, columns-1:0:-1].T.reshape(-1,2) * square_size
    #objp = objp[::-1] #reversing due to pytorch opencv hand and array indxing convention differences


    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    successful_indices = []  # This will store the indices of images where corners were detected
    
    imgs_with_corners_list = []


    
    #distortion free
    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
    for idx, img_tensor in enumerate(images):
        # Convert tensor to numpy array and adjust dimensions
        points = model(img_tensor)
        points = points.view(-1, 2)


        img_np = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        
        # Convert range from [0,1] to [0,255] if required
        if img_np.max() <= 1:
            img_np = (img_np * 255).astype(np.uint8)

        # # Convert the image to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        corners = points.squeeze(0).unsqueeze(1).cpu().numpy()




        # # Find chessboard corners
        # ret, corners = cv2.findChessboardCorners(gray, (rows-1, columns-1), None)

        if True:
            objpoints.append(objp)
            successful_indices.append(idx)  # add index to the successful list

            # Refine the corners
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # # # Draw and display the corners on the image
            img_with_corners = cv2.drawChessboardCorners(img_np.copy(), (2, 2), corners2, True)
            imgs_with_corners_list.append(img_with_corners)
            # # Using OpenCV to display the image
            # cv2.imshow('Detected Corners', img_with_corners)
            # cv2.waitKey(0)  # waits until a key is pressed
            # cv2.destroyAllWindows()


    if len(objpoints) == 0:
        return None
    else:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=flags)
        

        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        mean_error = mean_error / len(objpoints)
    return mean_error, mtx, rvecs, tvecs, successful_indices,imgs_with_corners_list    # Also return the successful indices list



#function to render and save images
def render_and_save_images(predicted_params_batch, rows, columns, square_size, device, camera_name, imgs_with_corners_list, directory,  succesfull_indices, size=[320,320], render_all=False):
    #create 3D chessboard mesh
    chessboard_mesh, _ = create_3d_chessboard(int(rows.item()), int(columns.item()), sq=square_size.item())

    
    if render_all:
        for batch_idx in range(predicted_params_batch.shape[0]):
            #convert Rodrigues vector to rotation matrix
            rodrigues = predicted_params_batch[batch_idx, 4:7].cpu().numpy()
            R, _ = cv2.Rodrigues(rodrigues)
            R = torch.tensor(R, dtype=torch.float32).unsqueeze(0).to(device)
            
            
            #extract translation and camera matrix
            T = predicted_params_batch[batch_idx, 7:].unsqueeze(0).to(device)
            camera_matrix = torch.eye(3, dtype=torch.float32).unsqueeze(0)
            camera_matrix[:, 0, 0] = predicted_params_batch[0, 0]
            camera_matrix[:, 1, 1] = predicted_params_batch[0, 1]
            camera_matrix[:, 0, 2] = predicted_params_batch[0, 2]
            camera_matrix[:, 1, 2] = predicted_params_batch[0, 3]
            camera_matrix = camera_matrix.to(device)

            #rennder mesh
            rendered_image = render_mesh(chessboard_mesh.to(device), camera_matrix, R, T, size=size)
            rendered_image = rendered_image[:, :, :, :3]

            #save the images
            camera_dir = os.path.join(os.getcwd(), directory, camera_name)
            os.makedirs(camera_dir, exist_ok=True)
            

            
            image_index = succesfull_indices[batch_idx]
            render_img_path = os.path.join(camera_dir, f"rendered_image_{image_index}.png")
            rendered_image_permuted = rendered_image.permute(0, 3, 1, 2)

            save_image(rendered_image_permuted, render_img_path)
        
        for idx, image in enumerate(imgs_with_corners_list):
                corner_img_path = os.path.join(camera_dir, f"image_with_corners_{idx}.png")
                cv2.imwrite(corner_img_path, image)


    else:
        #convert Rodrigues vector to rotation matrix
        rodrigues = predicted_params_batch[-1, 4:7].cpu().numpy()
        R, _ = cv2.Rodrigues(rodrigues)
        R = torch.tensor(R, dtype=torch.float32).unsqueeze(0).to(device)
        
        
        #extract translation and camera matrix
        T = predicted_params_batch[-1, 7:].unsqueeze(0).to(device)
        camera_matrix = torch.eye(3, dtype=torch.float32).unsqueeze(0)
        camera_matrix[:, 0, 0] = predicted_params_batch[0, 0]
        camera_matrix[:, 1, 1] = predicted_params_batch[0, 1]
        camera_matrix[:, 0, 2] = predicted_params_batch[0, 2]
        camera_matrix[:, 1, 2] = predicted_params_batch[0, 3]
        camera_matrix = camera_matrix.to(device)

        #rennder mesh
        rendered_image = render_mesh(chessboard_mesh.to(device), camera_matrix, R, T, size=size)
        rendered_image = rendered_image[:, :, :, :3]

        #save the images
        camera_dir = os.path.join(os.getcwd(), directory, camera_name)
        os.makedirs(camera_dir, exist_ok=True)
        

        for idx, image in enumerate(imgs_with_corners_list):
            corner_img_path = os.path.join(camera_dir, f"image_with_corners_{idx}.png")
            cv2.imwrite(corner_img_path, image)

        image_index = succesfull_indices[-1]
        render_img_path = os.path.join(camera_dir, f"rendered_image_{image_index}.png")
        rendered_image_permuted = rendered_image.permute(0, 3, 1, 2)

        save_image(rendered_image_permuted, render_img_path)



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
    
    
    evaluation_batch_size = 1


    #read images info csv to pandas db
    testing_images_info_db = pd.read_csv(testing_image_info_csv_path, header = 0)
    test_dataset = ChessboardDataset(testing_data_directory, testing_backgrounds_masks_directory, background_directory, testing_chessboardcorners_masks_directory, testing_images_info_db, use_background=True, augment=False, jitter_params=None, blur_params=None, num_images=15)
    subset_length = int(0.3*len(test_dataset))
    subset_indices = torch.randperm(len(test_dataset))[:subset_length]
    subset_test_dataset = Subset(test_dataset, subset_indices)
    test_loader = DataLoader(subset_test_dataset, batch_size=evaluation_batch_size, shuffle=True, num_workers=8)

    model = CornerNet()
    model_weights = torch.load('model_weights_epoch_1.pth')
    model.load_state_dict(model_weights)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    rendereing_calibration_results = pd.DataFrame(columns=[
    "Camera Name", 
    "Requested Images Used", 
    "Actual Images Used", 
    "Calibration Status", 
    "OpenCV Predicted Error", 
    "2D Projection Error", 
    "3D Reprojection Error", 
    "MAPE"
])

    with torch.no_grad():
        for camera, test_data in tqdm(enumerate(test_loader, 0), desc="Processing", total=len(test_loader)):
            square_size, images_with_backgrounds_list, ground_truths_list, rows, columns = [d for d in test_data]


            for images_used in [5, 10, 15]:
                images = images_with_backgrounds_list[:images_used]

                calibration_output = calibrate_camera_with_corner_net(images, square_size, rows, columns, model)

                if calibration_output is None:
                    # Append data to the DataFrame when calibration fails
                    new_row = pd.Series({
                        "Camera Name": "None",
                        "Requested Images Used": images_used,
                        "Actual Images Used": "None",
                        "Calibration Status": "Failed",
                        "OpenCV Predicted Error": "None",
                        "2D Projection Error": "None",
                        "3D Reprojection Error": "None",
                        "MAPE": "None"
                    })
                    rendereing_calibration_results = pd.concat([rendereing_calibration_results, pd.DataFrame([new_row])], ignore_index=True)
                    
                else:
                    mean_error, mtx, rvecs, tvecs, successful_indices, imgs_with_corners_list   = calibration_output
                    #number of images actually used during calibration
                    num_images_used = len(successful_indices)


                    # Filtering ground truths by successful indices
                    ground_truths_tensor = torch.cat(ground_truths_list, dim=0)
                    ground_truths_tensor = ground_truths_tensor[successful_indices]

                    # Extract intrinsic parameters from mtx
                    fx = mtx[0, 0]
                    fy = mtx[1, 1]
                    cx = mtx[0, 2]
                    cy = mtx[1, 2]

                    # Replicate the intrinsic parameters to match the batch size
                    batch_size = len(rvecs)
                    fx_tensor = torch.full((batch_size,), fx).float()
                    fy_tensor = torch.full((batch_size,), fy).float()
                    cx_tensor = torch.full((batch_size,), cx).float()
                    cy_tensor = torch.full((batch_size,), cy).float()


                    rvecs_tensor = torch.tensor([rvec.ravel() for rvec in rvecs]).float()
                    tvecs_tensor = torch.tensor([tvec.ravel() for tvec in tvecs]).float()
                    predicted_params_batch = torch.stack([fx_tensor, fy_tensor, cx_tensor, cy_tensor, 
                                     rvecs_tensor[:, 0], rvecs_tensor[:, 1], rvecs_tensor[:, 2], 
                                     tvecs_tensor[:, 0], tvecs_tensor[:, 1], tvecs_tensor[:, 2]], dim=1)


                    
                    # Calculating MAPE
                    mape = mape_loss(predicted_params_batch, ground_truths_tensor)

                    # Calculating 2D and 3D errors
                    chessboard_verts = generate_vertices(square_size, rows, columns)

                    twod_error, threed_error = projection_losses(chessboard_verts, predicted_params_batch, ground_truths_tensor)
                    #identifier for results                    
                    camera_name = f"{twod_error:.2f}_camera_{fx:.2f}_{fy:.2f}_{cx:.2f}_{cy:.2f}"

                    #rendering chessboard with predicte params for comparison
                    render_and_save_images(predicted_params_batch, rows, columns, square_size, device, camera_name, imgs_with_corners_list  ,directory=os.path.join("rendered_data_results", "number_of_images"), succesfull_indices=successful_indices)
                    #save results
                    rendereing_calibration_results.loc[len(rendereing_calibration_results)] = [camera_name, images_used, num_images_used, "Successful", mean_error, twod_error.item(), threed_error.item(), mape.item()]
                    

                    
                    new_row = pd.Series({
                        "Camera Name": camera_name,
                        "Requested Images Used": images_used,
                        "Actual Images Used": num_images_used,
                        "Calibration Status": "Successful",
                        "OpenCV Predicted Error": mean_error,
                        "2D Projection Error": twod_error.item(),
                        "3D Reprojection Error": threed_error.item(),
                        "MAPE": mape.item()
                    })
                    rendereing_calibration_results = pd.concat([rendereing_calibration_results, pd.DataFrame([new_row])], ignore_index=True)
                    

        #save to csv
        rendereing_calibration_results.to_csv(os.path.join("rendered_data_results", "number_of_images","calibration_results.csv"), index=False)



        #testing on rendered set with different levels of noise and blurring
        noise_calibration_df = pd.DataFrame(columns=[
            "Camera Name", 
            "Requested Images Used", 
            "Actual Images Used", 
            "Calibration Status", 
            "OpenCV Predicted Error", 
            "2D Projection Error", 
            "3D Reprojection Error", 
            "MAPE",
            "Jitter",
            "Blur"
        ])

        #stepping throough blur and jitter levels
        for jitter_index, jitter_params in enumerate(color_jitter_params_grid):
            for blur_index, blur_params in enumerate(gaussian_blur_params_grid):
                testing_images_info_db = pd.read_csv(testing_image_info_csv_path, header = 0)
                test_dataset = ChessboardDataset(testing_data_directory, testing_backgrounds_masks_directory, background_directory, testing_chessboardcorners_masks_directory, testing_images_info_db, augment_colours=False, use_background=True, augment=True, 
                                                 jitter_params=jitter_params, blur_params=blur_params, num_images=15)
                #test_loader = DataLoader(test_dataset, batch_size=evaluation_batch_size, shuffle=True, num_workers=8)
                subset_length = int(0.3*len(test_dataset))
                subset_indices = torch.randperm(len(test_dataset))[:subset_length]
                subset_test_dataset = Subset(test_dataset, subset_indices)
                test_loader = DataLoader(subset_test_dataset, batch_size=evaluation_batch_size, shuffle=True, num_workers=8)

                for camera, test_data in tqdm(enumerate(test_loader, 0), desc="Processing", total=len(test_loader)):
                    square_size, images_with_backgrounds_list, ground_truths_list, rows, columns = [d for d in test_data]

                    images = images_with_backgrounds_list

                    calibration_output = calibrate_camera_with_corner_net(images, square_size, rows, columns, model)

                    if calibration_output is None:
                            new_row = pd.Series({
                            "Camera Name": "None",
                            "Requested Images Used": 15,
                            "Actual Images Used": "None",
                            "Calibration Status": "Failed",
                            "OpenCV Predicted Error": "None",
                            "2D Projection Error": "None",
                            "3D Reprojection Error": "None",
                            "MAPE": "None", 
                            "Jitter": jitter_index, 
                            "Blur": blur_index, 
                            })
                            noise_calibration_df = pd.concat([noise_calibration_df, pd.DataFrame([new_row])], ignore_index=True)
                    else:
                        mean_error, mtx, rvecs, tvecs, successful_indices, imgs_with_corners_list   = calibration_output
                        #number of images actually used during calibration
                        num_images_used = len(successful_indices)
                        # Filtering ground truths by successful indices
                        ground_truths_tensor = torch.cat(ground_truths_list, dim=0)
                        ground_truths_tensor = ground_truths_tensor[successful_indices]

                        # Extract intrinsic parameters from mtx
                        fx = mtx[0, 0]
                        fy = mtx[1, 1]
                        cx = mtx[0, 2]
                        cy = mtx[1, 2]

                        # Replicate the intrinsic parameters to match the batch size
                        batch_size = len(rvecs)
                        fx_tensor = torch.full((batch_size,), fx).float()
                        fy_tensor = torch.full((batch_size,), fy).float()
                        cx_tensor = torch.full((batch_size,), cx).float()
                        cy_tensor = torch.full((batch_size,), cy).float()


                        rvecs_tensor = torch.tensor([rvec.ravel() for rvec in rvecs]).float()
                        tvecs_tensor = torch.tensor([tvec.ravel() for tvec in tvecs]).float()
                        predicted_params_batch = torch.stack([fx_tensor, fy_tensor, cx_tensor, cy_tensor, 
                                        rvecs_tensor[:, 0], rvecs_tensor[:, 1], rvecs_tensor[:, 2], 
                                        tvecs_tensor[:, 0], tvecs_tensor[:, 1], tvecs_tensor[:, 2]], dim=1)


                        
                        # Calculating MAPE
                        mape = mape_loss(predicted_params_batch, ground_truths_tensor)

                        # Calculating 2D and 3D errors
                        chessboard_verts = generate_vertices(square_size, rows, columns)

                        twod_error, threed_error = projection_losses(chessboard_verts, predicted_params_batch, ground_truths_tensor)
                        #identifier for results                    
                        camera_name = f"{twod_error:.2f}_camera_{fx:.2f}_{fy:.2f}_{cx:.2f}_{cy:.2f}"

                        #rendering chessboard with predicte params for comparison
                        render_and_save_images(predicted_params_batch, rows, columns, square_size, device, camera_name, imgs_with_corners_list  ,directory=os.path.join("rendered_data_results", "jitterblur", f"jitter_{jitter_index}_blur_{blur_index}"), succesfull_indices=successful_indices)
                        #save results
                        rendereing_calibration_results.loc[len(rendereing_calibration_results)] = [camera_name, 15, num_images_used, "Successful", mean_error, twod_error.item(), threed_error.item(), mape.item()]
                        

                        
                        new_row = pd.Series({
                            "Camera Name": camera_name,
                            "Requested Images Used": 15,
                            "Actual Images Used": num_images_used,
                            "Calibration Status": "Successful",
                            "OpenCV Predicted Error": mean_error,
                            "2D Projection Error": twod_error.item(),
                            "3D Reprojection Error": threed_error.item(),
                            "MAPE": mape.item(),
                            "Jitter": jitter_index, 
                            "Blur": blur_index, 
                        })
                        noise_calibration_df = pd.concat([noise_calibration_df, pd.DataFrame([new_row])], ignore_index=True)
                        

        #save to csv
        noise_calibration_df.to_csv(os.path.join("rendered_data_results", "jitterblur","calibration_results.csv"), index=False)



        #real data results
        # Loading the real images
        image_extensions = ('.jpeg', '.jpg', '.png', '.bmp', '.tif', '.tiff')
        image_filenames = [f for f in os.listdir(real_data_directory) if f.lower().endswith(image_extensions)]

        real_images = [Image.open(os.path.join(real_data_directory, fname)) for fname in image_filenames]



        real_images_results_df = pd.DataFrame(columns=[
            "Requested Images Used", 
            "Actual Images Used", 
            "Calibration Status", 
            "OpenCV Predicted Error", 
            "Jitter",
            "Blur"
        ])

        to_tensor_transform = transforms.ToTensor()

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

                transformed_images_list = []
                for img_idx, img in tqdm(enumerate(real_images), desc="Processing Real Images"):
                    # Apply transformations

                    transformed_img = (torch.tensor(to_tensor_transform(blur_transform(jitter_transform(img))))).unsqueeze(0)
                    transformed_images_list.append(transformed_img)



                    
                calibration_output = calibrate_camera_with_corner_net(transformed_images_list, torch.tensor(2.5), torch.tensor(10), torch.tensor(7), model)

                if calibration_output is None:
                    # Append data to the DataFrame when calibration fails
                    new_row = pd.Series({
                        "Camera Name": "None",
                        "Requested Images Used": 50,
                        "Actual Images Used": "None",
                        "Calibration Status": "Failed",
                        "OpenCV Predicted Error": "None",
                        "Jitter":jitter_index,
                        "Blur":blur_index
                    })
                    real_images_results_df = pd.concat([real_images_results_df, pd.DataFrame([new_row])], ignore_index=True)
                    print("fail")
                
                else:
                    mean_error, mtx, rvecs, tvecs, successful_indices, imgs_with_corners_list   = calibration_output
                    #number of images actually used during calibration
                    num_images_used = len(successful_indices)

                    # Extract intrinsic parameters from mtx
                    fx = mtx[0, 0]
                    fy = mtx[1, 1]
                    cx = mtx[0, 2]
                    cy = mtx[1, 2]

                    # Replicate the intrinsic parameters to match the batch size
                    batch_size = len(rvecs)
                    fx_tensor = torch.full((batch_size,), fx).float()
                    fy_tensor = torch.full((batch_size,), fy).float()
                    cx_tensor = torch.full((batch_size,), cx).float()
                    cy_tensor = torch.full((batch_size,), cy).float()


                    rvecs_tensor = torch.tensor([rvec.ravel() for rvec in rvecs]).float()
                    tvecs_tensor = torch.tensor([tvec.ravel() for tvec in tvecs]).float()
                    predicted_params_batch = torch.stack([fx_tensor, fy_tensor, cx_tensor, cy_tensor, 
                                    rvecs_tensor[:, 0], rvecs_tensor[:, 1], rvecs_tensor[:, 2], 
                                    tvecs_tensor[:, 0], tvecs_tensor[:, 1], tvecs_tensor[:, 2]], dim=1)
                    
                   
                    camera_name = f"{mean_error:.2f}_camera_{fx:.2f}_{fy:.2f}_{cx:.2f}_{cy:.2f}"

                    #rendering chessboard with predicte params for comparison
                    render_and_save_images(predicted_params_batch, torch.tensor(10), torch.tensor(7), torch.tensor(2.5), device, camera_name, imgs_with_corners_list, succesfull_indices=successful_indices, directory=os.path.join("real_data_results", "jitterblur", f"jitter_{jitter_index}_blur_{blur_index}"), size=[320,320], render_all=True)
                    #save results
                    new_row = pd.Series({
                        "Requested Images Used": 50,
                        "Actual Images Used": num_images_used,
                        "Calibration Status": "Successful",
                        "OpenCV Predicted Error": mean_error,
                        "Jitter":jitter_index,
                        "Blur":blur_index
                    })
                    real_images_results_df = pd.concat([real_images_results_df, pd.DataFrame([new_row])], ignore_index=True)
                    

        #save to csv
        real_images_results_df.to_csv(os.path.join("real_data_results", "jitterblur","calibration_results.csv"), index=False)




if __name__ == '__main__':
    main()