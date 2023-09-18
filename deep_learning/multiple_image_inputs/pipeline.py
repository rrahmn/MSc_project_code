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
import torch.nn as nn
import pandas as pd

base_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
data_directory = os.path.join(base_directory, 'Rendered_Data','Auto_generated', 'images')
masks_directory = os.path.join(base_directory, 'Rendered_Data','Auto_generated', 'images_masks')
background_directory = os.path.join(base_directory, 'backgrounds')
image_info_csv_path = os.path.join(base_directory, 'CSML_MSc_Project_Work','rendering', 'rendered_images_info.csv')



class ChessboardDataset(Dataset):
    def __init__(self, data_directory, masks_directory, background_directory, images_info_db, stack_size=5):
        self.data_directory = data_directory
        self.masks_directory = masks_directory
        self.background_directory = background_directory

        #database with image info
        self.images_info = images_info_db
        #indexing on image_name column
        self.images_info.set_index("image_name", inplace = True)

        self.architecture_image_size = [320,240]

        
        #folder names
        self.folders = sorted(os.listdir(data_directory))
        self.folder_lengths = [len(os.listdir(os.path.join(data_directory, folder))) for folder in self.folders]
        self.cumulative_lengths = [0] + list(np.cumsum(self.folder_lengths))

        # image paths
        self.image_paths = []
        self.mask_paths = []
        for folder in self.folders:
            self.image_paths.extend(sorted([os.path.join(self.data_directory, folder, img) for img in os.listdir(os.path.join(self.data_directory, folder))]))
            self.mask_paths.extend(sorted([os.path.join(self.masks_directory, folder, mask) for mask in os.listdir(os.path.join(self.masks_directory, folder))]))
        
        # background images
        self.bg_images = sorted([os.path.join(background_directory, img) for img in os.listdir(background_directory)])
        self.stack_size = stack_size

    def __getitem__(self, idx):
        folder_idx = np.digitize(idx, self.cumulative_lengths) - 1
        start_idx = idx - self.cumulative_lengths[folder_idx]

        all_indices = np.delete(np.arange(self.folder_lengths[folder_idx]), start_idx)


        random_indices = np.random.choice(all_indices, size=self.stack_size-1, replace=False)


        selected_indices = np.insert(random_indices, 0, start_idx)


        selected_image_paths = np.array(self.image_paths)[self.cumulative_lengths[folder_idx] + selected_indices]
        selected_mask_paths = np.array(self.mask_paths)[self.cumulative_lengths[folder_idx] + selected_indices]

        images_list = [self.process_image(image_path, mask_path)[0] for image_path, mask_path in zip(selected_image_paths, selected_mask_paths)]
        ground_truth, scaling = self.process_image(selected_image_paths[0], selected_mask_paths[0])[1:]

        return images_list, ground_truth, scaling

    def process_image(self, image_path, mask_path):
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        bg_image_path = np.random.choice(self.bg_images)
        bg_image = Image.open(bg_image_path).convert('RGB')

        #AUGMENTATIONS
        #to get different coloured chessboards
        switch = random.random()
        if(switch>0.95):
            colour_replacement1 = (randrange(256), randrange(256), randrange(256))
            colour_replacement2 = (randrange(256), randrange(256), randrange(256))

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

        
        
        resize = transforms.Compose([
            transforms.Resize((self.architecture_image_size[1], self.architecture_image_size[0]), interpolation=Image.LANCZOS)
            ])

        
        
        
        #make sure background consistent shape with image and mask
        bg_image = bg_image.resize(image.size, Image.ANTIALIAS)
        image_np = np.array(image)
        #for broadcasting mask
        mask_np = np.array(mask)/255
        mask_np = mask_np[:,:,np.newaxis]
        bg_image_np = np.array(bg_image)



        #get the image with the background
        image_with_background_np = ( mask_np == 0) * image_np + (mask_np == 1) * bg_image_np
        image_with_background = Image.fromarray(np.uint8(image_with_background_np))
        #resize to architecture needed shape
        image_with_background = resize(image_with_background)

        del image_np
        del mask_np
        del bg_image_np
        del image
        del mask
        del bg_image
        collect()

        
        augment_switch = random.random()
        if(augment_switch>0.95):
            ksize = random.randrange(3, 11, 2)
            sig = (0.1, 2.0)
            b = np.random.beta(0.3, 4)
            c = np.random.beta(0.3, 4)
            s = np.random.beta(0.3, 4)
            h = 0.5*np.random.beta(0.3, 4)



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


        image_with_background = to_tensor(image_with_background)
        


        #LOADING FROM IMAGE INFO FILE
        fx, fy, cx, cy, rodrigues0, rodrigues1, rodrigues2, tx, ty, tz, rows, columns, width, height = self.images_info.loc[os.path.basename(image_path)]
        scale_factor_x = self.architecture_image_size[0]/width
        scale_factor_y = self.architecture_image_size[1]/height

        fx_scaled = fx*scale_factor_x
        fy_scaled = fy*scale_factor_y
        cx_scaled = cx*scale_factor_x
        cy_scaled = cy*scale_factor_y
        ground_truth = torch.tensor([fx_scaled,fy_scaled,cx_scaled,cy_scaled])
        scaling = torch.tensor([scale_factor_x, scale_factor_y,scale_factor_x, scale_factor_y])
        return image_with_background, ground_truth, scaling

    def __len__(self):
        adjusted_lengths = [(l + self.stack_size - 1) // self.stack_size for l in self.folder_lengths]
        return sum(adjusted_lengths)


class MultipleImageResNet(nn.Module):
    def __init__(self):
        super(MultipleImageResNet, self).__init__() 
        resnet50_backbone = torchvision.models.resnet50(weights=None)
        last_layer = list(resnet50_backbone.children())[-1]
        num_features = last_layer.in_features


        self.image_stream = nn.Sequential(*list(resnet50_backbone.children())[:-1], nn.Flatten())
        self.fc1 = nn.Linear(num_features, num_features//2) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_features//2, 4) 
        
    def forward(self, x_list):
        stream_outputs = [self.image_stream(x) for x in x_list]
        # while len(stream_outputs) < self.max_images:
        #     stream_outputs.append(torch.zeros_like(stream_outputs[0]))

        
        # merged = torch.cat(stream_outputs, dim=1)
        # merged = self.relu(merged)
        # x = self.fc1(merged)

        x = torch.mean(torch.stack(stream_outputs), dim=0)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def main():
    #read images info csv to pandas db
    images_info_db = pd.read_csv(image_info_csv_path)
    train_dataset = ChessboardDataset(data_directory, masks_directory, background_directory, images_info_db, 5)

    # #train, val, test splits
    # total_length = len(dataset)
    # train_length = int(total_length * 0.6)
    # val_length = int(total_length * 0.2)
    # test_length = total_length - train_length - val_length
    batch_size  = 12

    # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_length, val_length, test_length])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    #architecture
    model = MultipleImageResNet()

    #move to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)


    #loss and optimiser
    #criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)



    num_epochs = 3

    losses = []

    
    #Training
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_rightscale_loss = 0.0
        for i, data in tqdm(enumerate(train_loader, 0), desc="Processing", total=len(train_loader)):
            
            inputs, labels, scaling = data
            inputs = [input_tensor.to(device) for input_tensor in inputs]

            

            labels = labels.to(device)
            scaling = scaling.to(device)


            
            # with torch.no_grad():
            #     #augment by addign backgrounds
            #     masks_rep = torch.cat([masks]*3, dim=1)
            #     image_portion = torch.mul(inputs, (1.0 - masks_rep))
            #     bg_portion = torch.mul(bg_images, masks_rep)
            #     combined_images = torch.add(image_portion, bg_portion)
            



            # num_samples_to_show = 1  
            # for j in range(min(num_samples_to_show, batch_size)):
            #     # plt.figure(figsize=(12, 4))

            #     # Original Image
            #     # plt.subplot(1, 3, 1)
            #     plt.imshow(inputs[j].permute(1, 2, 0).cpu().numpy())
            #     plt.title("Original Image")

            #     # # Background Image
            #     # plt.subplot(1, 3, 2)
            #     # plt.imshow(bg_images[j].permute(1, 2, 0).cpu().numpy())
            #     # plt.title("Background Image")

            #     # # Combined Image
            #     # plt.subplot(1, 3, 3)
            #     # plt.imshow(combined_images[j].permute(1, 2, 0).cpu().numpy())
            #     # plt.title("Combined Image")

            #     plt.tight_layout()
            #     plt.show()
            # #del inputs, masks, bg_images, masks_rep, image_portion, bg_portion
            
            # torch.cuda.empty_cache()
            # collect()

            
            optimizer.zero_grad()   #zero the gradiuents
            
            outputs = model(inputs)   
            #loss = criterion(outputs, labels)  
            loss = torch.mean((outputs - labels)**2, axis=0)
            #loss = torch.mean(torch.log((outputs/labels)**2), axis=0)

            for n, component in enumerate(loss):
                if(n==len(loss) - 1):
                    component.backward()
                else:
                    component.backward(retain_graph=True)
            
            #loss.backward()   
            optimizer.step()  
            
            
            
            with torch.no_grad():
                if (i + 1) % 50 == 0:
                    print(outputs)
                    print(labels)
                    print(loss)
                torch.cuda.empty_cache()
                collect()

                scaling = scaling.to(device)
                right_scale_loss = torch.mean(((outputs - labels)/scaling) ** 2, axis=0)
                running_rightscale_loss += torch.mean(right_scale_loss).item()
                running_loss += torch.mean(loss).item()
            
            
                # Print average loss every 5 iterations
                if (i + 1) % 50 == 0:
                    avg_rightscale_loss = running_rightscale_loss / 50
                    avg_loss = running_loss / 50
                    print(f"Iteration {i+1}, Average right scale mse Loss: {avg_rightscale_loss}")
                    print(f"Iteration {i+1}, Average Loss: {avg_loss}")

                    losses.append(avg_rightscale_loss)
                    running_loss = 0.0
                    running_rightscale_loss = 0.0
        
        #scheduler.step()  #update learning rate
        
        


    # Save the trained model weights
    torch.save(model.state_dict(), f"resnet50_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")

    # Plot the losses
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Iterations (every 5th)")
    plt.ylabel("Loss")
    plt.show()
    print('Finished Training')



if __name__ == '__main__':
    main()