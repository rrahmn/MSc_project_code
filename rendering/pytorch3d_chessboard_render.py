import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform, Textures, RasterizationSettings, MeshRenderer, MeshRasterizer, 
    HardPhongShader, BlendParams, SoftPhongShader
)
import matplotlib.pyplot as plt
from pytorch3d.structures import join_meshes_as_batch
import numpy as np
import os
import argparse
from tqdm import tqdm
from pytorch3d.utils import cameras_from_opencv_projection
from datetime import datetime
import uuid
import cv2
import pandas as pd
from torchvision import transforms
from torchvision.transforms import GaussianBlur

def create_save_directory(save_directory, intrinsic_params):
    """ function to check and create save directory """
    output_dir = os.path.join(save_directory, f'fx_{intrinsic_params[0]}_fy_{intrinsic_params[1]}_cx_{intrinsic_params[2]}_cy_{intrinsic_params[3]}')
    #create the directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


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


def render_mesh(mesh, corners, camera_matrix, R, T, bg=(0.0, 1.0, 0.0)):
    "function to render a bunch of images of the chessboard given a batch of intrinsic and extrinsic parameters"
    
    #creating a batch of PerspectiveCameras using the provided function
    image_size = torch.tensor([[320, 320]])
    cameras = cameras_from_opencv_projection(
        R=R,
        tvec=T,
        camera_matrix=camera_matrix,
        image_size=image_size
    )   
    projected_corners = cameras.transform_points_screen(corners)
    projected_corners = projected_corners[:,:,:2]
    
    
    
    #rendering in higher res for antialiasing purposes
    image_size = torch.tensor([[960, 960]])
    sf = 960/320
    camera_matrix[:, 0, 0] = camera_matrix[:, 0, 0]*sf
    camera_matrix[:, 1, 1] =  camera_matrix[:, 1, 1]*sf
    camera_matrix[:, 0, 2] = camera_matrix[:, 0, 2]*sf
    camera_matrix[:, 1, 2] = camera_matrix[:, 1, 2]*sf
    
    #creating a batch of PerspectiveCameras using the provided function
    cameras = cameras_from_opencv_projection(
        R=R,
        tvec=T,
        camera_matrix=camera_matrix,
        image_size=image_size
    )
    raster_settings = RasterizationSettings(image_size=(960, 960))
    blend_params = BlendParams(background_color=bg)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras, blend_params=blend_params)
        #shader = SoftPhongShader(device=device, cameras=cameras, blend_params=blend_params)
    ).to(device)


    

    return renderer(mesh), projected_corners

# def render_mask_mesh(mesh, camera_matrix, R, T, bg=(0.0, 0.0, 0.0)):
#     "function to render a bunch of images of the chessboard given a batch of intrinsic and extrinsic parameters"
#     image_size = torch.tensor([[320, 320]])
#     #creating a batch of PerspectiveCameras using the provided function
#     cameras = cameras_from_opencv_projection(
#         R=R,
#         tvec=T,
#         camera_matrix=camera_matrix,
#         image_size=image_size
#     )
#     raster_settings = RasterizationSettings(image_size=(320, 320))
#     blend_params = BlendParams(background_color=bg)

#     renderer = MeshRenderer(
#         rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
#         shader=HardPhongShader(device=device, cameras=cameras, blend_params=blend_params)
#     ).to(device)

#     return renderer(mesh)

# def anti_alias(image):
#     "function to take a super sampled image and size it down as an antialiasing measure"
#     image = image[:,:,:,:3]
#     image = image.permute(0, 3, 1, 2)

#     blur = transforms.Compose([
#                     GaussianBlur(kernel_size=(3,3), sigma=0.7)
#                     ])
#     #samlpling down
#     #desired_size = (256, 256)
#     #image_resized = torch.nn.functional.interpolate(image, size=desired_size, mode='area')
#     image_resized = blur(image)
#     return image_resized

def anti_alias(image):
    "function to take a super sampled image and size it down as an antialiasing measure"
    image = image[:,:,:,:3]
    image = image.permute(0, 3, 1, 2)
    #samlpling down
    desired_size = (320, 320)
    image_resized = torch.nn.functional.interpolate(image, size=desired_size, mode='area')
    return image_resized


def resize_mesh(mesh, scale_factor):
    """ function to resize square size of  chessboard. 
     input:chessboard mesh, scale factor
      output: scaled chessboard mesh
     """
    verts = mesh.verts_list()[0] * scale_factor
    faces = mesh.faces_list()[0]
    textures = mesh.textures
    return Meshes(verts=[verts], faces=[faces], textures=textures)




def roll_rotation_about_axis(angle_degrees, axis):
    """Compute a rotation matrix about the given axis."""
    
    rvec = np.deg2rad(angle_degrees) * axis.numpy()
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    return torch.Tensor(rotation_matrix)


def generate_extrinsics(num_meshes, centre, square, move_back_scaling, chessboard_rows, chessboard_columns):

    """ function to generate batched extrinsics """
    Rs = []
    Ts = []
    chessboard_length = square*chessboard_columns
    #https://en.wikipedia.org/wiki/Spherical_coordinate_system
    for _ in range(num_meshes):
        elevation = (np.random.rand() * 110) - 55
        if -3 <= elevation <= 3:
            elevation += 3 if elevation >= 0 else -3

        azimuth = (np.random.rand() * 110) - 55
        if -3 <= azimuth <= 3:
            azimuth += 3 if azimuth >= 0 else -3
        dist = np.random.uniform(1.2*chessboard_length + chessboard_length*(move_back_scaling - 1)/1.5, 2.3*chessboard_length + 1.5*chessboard_length*(move_back_scaling - 1))
        
        #instead of looking at the centre of  the chessboard we  will  look at a random point sampled from a 2d normal with centre as mean
        covariance_matrix = np.eye(2) 
        covariance_matrix[0,0] = 0.25*chessboard_rows*square/2
        covariance_matrix[1,1] = 0.25*chessboard_columns*square/2
        sampled_point = np.random.multivariate_normal(centre[:2], covariance_matrix)
        at_point = (sampled_point[0], sampled_point[1], 0)

    
        R, T = look_at_view_transform(dist, elevation, azimuth, at=(at_point,))

        # Apply roll rotation
        optical_axis = R @ torch.Tensor([0, 0, 1])
        optical_axis = optical_axis / torch.norm(optical_axis)
        roll_angle = np.random.uniform(-20, 20)  # random roll between -20 and 20 degrees
        roll_matrix = roll_rotation_about_axis(roll_angle, optical_axis)
        R = torch.Tensor(R @ roll_matrix)  # Apply the roll after the current transformations
        
        Rs.append(R)
        Ts.append(T)

    Rs_batched = torch.cat(Rs, dim=0)
    Ts_batched = torch.cat(Ts, dim=0)

    return Rs_batched, Ts_batched



# def create_3d_chessboard_and_vertex_crosses(r, c, sq=1.0, cross_size=0.5):
#     """ 
#     Function to create chessboard mesh and vertex crosses.
#     input: chessboard rows r, chessboard columns c, square size sq, cross size
#     output: chessboard mesh, vertex crosses mesh 
#     """
#     verts, faces, vert_colors = [], [], []
#     cross_verts, cross_faces = [], []
    
#     for i in range(r):
#         for j in range(c):
#             z_offset = 0.0 
#             square_verts = [
#                 [i*sq, j*sq, z_offset],
#                 [(i+1)*sq, j*sq, z_offset],
#                 [(i+1)*sq, (j+1)*sq, z_offset],
#                 [i*sq, (j+1)*sq, z_offset]
#             ]
#             verts.extend(square_verts)
            
#             color = [2.0, 2.0, 2.0] if (i+j) % 2 == 0 else [0.0, 0.0, 0.0]
#             vert_colors.extend([color, color, color, color])
            
#             base_idx = len(verts) - 4
#             square_faces = [
#                 [base_idx, base_idx+1, base_idx+2],
#                 [base_idx, base_idx+3, base_idx+2]
#             ]
#             faces.extend(square_faces)

#             #detecting corners only
#             if(i==0 or j==0):
#                 continue
#             # Cross creation for vertex
#             half_size = cross_size / 2.0
#             x, y = i*sq, j*sq
#             c_verts = [
#                 [x - half_size, y, z_offset],
#                 [x + half_size, y, z_offset],
#                 [x, y - half_size, z_offset],
#                 [x, y + half_size, z_offset]
#             ]
#             base_idx_cross = len(cross_verts)
#             c_faces = [
#                 [base_idx_cross, base_idx_cross+1, base_idx_cross+2],
#                 [base_idx_cross+1, base_idx_cross+2, base_idx_cross+3],
#                 [base_idx_cross+2, base_idx_cross+3, base_idx_cross],
#                 [base_idx_cross+3, base_idx_cross, base_idx_cross+1]
#             ]
#             cross_verts.extend(c_verts)
#             cross_faces.extend(c_faces)
    
#     # Create chessboard mesh
#     verts = torch.tensor(verts, dtype=torch.float32)
#     faces = torch.tensor(faces, dtype=torch.int64)
#     vert_colors = torch.tensor(vert_colors, dtype=torch.float32).unsqueeze(0)
#     textures = Textures(verts_rgb=vert_colors)
#     chessboard_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    
#     # Create vertex crosses mesh
#     cross_verts = torch.tensor(cross_verts, dtype=torch.float32)
#     cross_faces = torch.tensor(cross_faces, dtype=torch.int64)
#     cross_vert_colors = 2.0*torch.ones_like(cross_verts).unsqueeze(0)
#     cross_textures = Textures(verts_rgb=cross_vert_colors)
#     vertex_crosses_mesh = Meshes(verts=[cross_verts], faces=[cross_faces], textures=cross_textures)
    
#     return chessboard_mesh, vertex_crosses_mesh


def generate_corners_mask(coordinates, img_size=(320, 320), radius=3):
    """
    Generates a grayscale image with circles plotted at given 2D coordinates. 
    The circles are brightest in the center and fade away as we move from the center.

    :param coordinates: Numpy array of shape [num_points, 2] containing 2D coordinates.
    :param img_size: Tuple indicating the size of the output grayscale image.
    :param radius: The radius of the circle within which the fading effect will be applied.
    :return: Numpy array representing the image.
    """
    
    # Create a blank grayscale image
    img = np.zeros((img_size[0], img_size[1]), dtype=np.float32)

    for coord in coordinates:
        x_center, y_center = int(coord[0]), int(coord[1])

        # Define the bounding box around the circle to reduce computations
        x_min, x_max = max(0, x_center - radius), min(img_size[1], x_center + radius + 1)
        y_min, y_max = max(0, y_center - radius), min(img_size[0], y_center + radius + 1)
        
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                distance = np.sqrt((x - x_center)**2 + (y - y_center)**2)
                
                if distance <= radius:
                    intensity = (1 - (distance / radius)) * 255
                    img[y, x] = max(img[y, x], intensity)  # Take the brighter value in case of overlap

    return img.astype(np.uint8)

def main():
    INSTRINSIC_SETS = {
    "training": "intrinsic_set_training.npy",
    "validation": "intrinsic_set_validation.npy",
    "testing": "intrinsic_set_testing.npy"
    }
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Train, val or test")
    parser.add_argument("--dataset_type", type=str, choices=["training", "validation", "testing"], default="training", help="Type of dataset to use (training, validation, testing)")
    parser.add_argument("--num_images", type=int, default=50, help="Number of images to generate per camera and chessboard")
    args = parser.parse_args()
    #load in intrinsic parameters
    intrinsic_sets = np.load(INSTRINSIC_SETS[args.dataset_type])

    #set up directories
    projectd_irectory = os.path.dirname(os.path.dirname(os.getcwd()))

    save_directory = os.path.join(projectd_irectory, "Rendered_Data", "Auto_generated", args.dataset_type)

    image_save_directory = os.path.join(save_directory,"images")
    mask_save_directory = os.path.join(save_directory,"images_masks")
    background_masks_directory = os.path.join(save_directory,"background_masks")


    #to store info
    rendered_images_info = []
    
    #go through intrinsics
    for _, intrinsic_params in tqdm(enumerate(intrinsic_sets), desc="Processing", total=len(intrinsic_sets)):
        #create directory where images and masks locating corners get saved
        output_directory_images =  create_save_directory(image_save_directory, intrinsic_params)
        output_directory_masks_corners = create_save_directory(mask_save_directory, intrinsic_params)
        output_directory_masks_chessboard = create_save_directory(background_masks_directory, intrinsic_params)

        #randomly select a chessboard shape, chessboard square size(between 1 and 5) and create mesh. the longer side of the chessboard will be the columns
        size1, size2 = np.random.choice(range(5, 11), 2, replace=False)
        rows = min(size1, size2)
        columns = max(size1, size2)
        square_size = np.random.uniform(1, 5)
        #mesh = create_3d_chessboard(rows, columns, square_size)
        #mesh, mask_mesh = create_3d_chessboard_and_vertex_crosses(rows, columns, square_size, cross_size=0.3)
        mesh, chessboard_corners = create_3d_chessboard(rows, columns, square_size)
        chessboard_corners = chessboard_corners.to(device)




        
        
        total_images_processed = 0
        batch_size = min(args.num_images, 5)
        current_mesh = mesh.extend(batch_size).to(device)

        
        #to go through images in batches
        while total_images_processed < args.num_images:
            #select batch size and batch mesh
            current_batch_size = min(args.num_images - total_images_processed, 5)
            total_images_processed += current_batch_size

            if(current_batch_size!=batch_size):
                current_mesh = mesh.extend(current_batch_size).to(device)

            

            #generate extrinsics and batch
            Rs,Ts = generate_extrinsics(current_batch_size, (rows*square_size/2, columns*square_size/2, 0), square = square_size, move_back_scaling=max(intrinsic_params[0], intrinsic_params[1])/200, chessboard_rows=rows, chessboard_columns=columns)
            Rs = Rs.to(device)
            Ts = Ts.to(device)

            #create batched camera matrix for all the views
            camera_matrix = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(current_batch_size,1,1)
            #render in 960 and downsample to 320 for antialiasing
            camera_matrix[:, 0, 0] = intrinsic_params[0]
            camera_matrix[:, 1, 1] = intrinsic_params[1]
            camera_matrix[:, 0, 2] = intrinsic_params[2]
            camera_matrix[:, 1, 2] = intrinsic_params[3]

            camera_matrix = camera_matrix.to(device)

            #render images and  anti alias
            images, projected_corners = render_mesh(current_mesh, chessboard_corners, camera_matrix, Rs, Ts)
            processed_imgs_batch = anti_alias(images)


            #save images
            for i, processed_imgs in enumerate(processed_imgs_batch):
                processed_img = processed_imgs.permute(1, 2, 0)
                processed_img = processed_img.clamp(0, 1)


                corner_mask = generate_corners_mask(projected_corners[i,4:,:])



                # corner_mask = torch.tensor(corner_mask).to(device)
                # red_channel = processed_img[:, :, 0]
                # red_channel_with_mask = (1 - corner_mask) * red_channel + corner_mask
                # overlayed_image = torch.clone(processed_img)
                # overlayed_image[:, :, 0] = red_channel_with_mask

                # plt.imshow(overlayed_image.cpu().numpy())
                # plt.show()

                green_threshold = 0.15

                #find greenish pixels
                is_greenish = (processed_img[..., 1] - processed_img[..., 0] > green_threshold) & (processed_img[..., 1] - processed_img[..., 2] > green_threshold)

                #binary mask: 1 for non-greenish, 0 for greenish
                chessboard_mask = torch.ones_like(processed_img[..., 0])
                chessboard_mask[is_greenish] = 0

                #generating unique names for images
                filename = f"chessboard_{str(uuid.uuid4())}.png"
                output_path = os.path.join(output_directory_images, filename)
                corners_mask_output_path = os.path.join(output_directory_masks_corners, filename)
                chessboard_mask_output_path =  os.path.join(output_directory_masks_chessboard, filename)
                plt.imsave(output_path, processed_img.cpu().numpy())
                cv2.imwrite(corners_mask_output_path, corner_mask)
                plt.imsave(chessboard_mask_output_path, chessboard_mask.cpu().numpy(), cmap='gray')


                R = Rs[i].cpu().numpy()
                T = Ts[i].cpu().numpy()
                rvec, _ = cv2.Rodrigues(R)
                tvec = T
                rendered_images_info.append({
                "image_name": filename,
                "fx": (intrinsic_params[0]).item(),
                "fy": (intrinsic_params[1]).item(),
                "cx": (intrinsic_params[2]).item(),
                "cy": (intrinsic_params[3]).item(),
                "rodrigues0": rvec[0, 0],
                "rodrigues1": rvec[1, 0],
                "rodrigues2": rvec[2, 0],
                "tx": tvec[0],
                "ty": tvec[1],
                "tz": tvec[2],
                "corner1x": (projected_corners[i,0,0]).item(),
                "corner1y": (projected_corners[i,0,1]).item(),
                "corner2x": (projected_corners[i,1,0]).item(),
                "corner2y": (projected_corners[i,1,1]).item(),
                "corner3x": (projected_corners[i,2,0]).item(),
                "corner3y": (projected_corners[i,2,1]).item(),
                "corner4x": (projected_corners[i,3,0]).item(),
                "corner4y": (projected_corners[i,3,1]).item(),
                "rows": rows,
                "columns": columns,
                "square_size": square_size
            })
    dataframe = pd.DataFrame(rendered_images_info)
    csv_name = f"{args.dataset_type}_images_info.csv"
    csv_path = os.path.join(save_directory, csv_name)
    if not os.path.exists(csv_path):
        dataframe.to_csv(csv_path, mode='w', index=False)
    else:
        # If it exists, append without writing the header
        dataframe.to_csv(csv_path, mode='a', header=False, index=False)

           


if __name__ == "__main__":
    main()



