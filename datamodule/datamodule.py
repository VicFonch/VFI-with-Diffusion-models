from imports.common_imports import *

class TargetImageDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        subfolders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

        self.image_paths = []
        for folder in subfolders:
            image_folder_path = os.path.join(data_dir, folder)
            image_paths = os.path.join(image_folder_path, 'It.png')
            self.image_paths.append(image_paths)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize([0.8750041, 0.8435287, 0.8396906], [0.2153176, 0.2438267, 0.2413682])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB') 
        images = self.transform(image)
        return images

class TripletImagesDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        # Obtener la lista de carpetas en el directorio de datos
        subfolders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

        self.image_paths = []
        for folder in subfolders:
            image_folder_path = os.path.join(data_dir, folder)
            I0 = os.path.join(image_folder_path, 'I0.png')
            It = os.path.join(image_folder_path, 'It.png')
            I1 = os.path.join(image_folder_path, 'I1.png')
            self.image_paths.append([I0, It, I1])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_paths = self.image_paths[index]
        
        images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
        
        transform_img = Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            #transforms.Normalize([0.8750041, 0.8435287, 0.8396906], [0.2153176, 0.2438267, 0.2413682])
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        images = [transform_img(image) for image in images]

        images = torch.stack(images)

        return images
    
def delete_invalid_images(base_dir, begin_from=0):
    with tqdm(total=len(os.listdir(base_dir))) as pbar:
        for folder in os.listdir(base_dir):
            if pbar.n < begin_from:
                pbar.update(1)
                continue
            folder_dir = os.path.join(base_dir, folder)
            I0_dir = os.path.join(folder_dir, 'I0.png')
            It_dir = os.path.join(folder_dir, 'It.png')
            I1_dir = os.path.join(folder_dir, 'I1.png')
            try:
                _ = Image.open(I0_dir).convert('RGB')
                _ = Image.open(It_dir).convert('RGB')
                _ = Image.open(I1_dir).convert('RGB')
            except Exception as e:
                print(f"Borrando imagen no válida: {folder_dir} por error: ({e})")
                shutil.rmtree(folder_dir)
            pbar.update(1)

