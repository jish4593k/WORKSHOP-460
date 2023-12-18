import torch
import torchvision.transforms as transforms
from tkinter import Tk, filedialog, Canvas, Button, PhotoImage
from PIL import Image, ImageTk

class ImageProcessor:
    def __init__(self):
        self.path = ''
        self.original_image = None
        self.processed_image = None

   
    def edge_detector(self):
        if self.original_image is not None:
          
            transform = transforms.Compose([transforms.ToTensor()])
            img_tensor = transform(self.original_image).unsqueeze(0)

        
            sobel_filter = torch.tensor([[[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype=torch.float32)
            edges = torch.nn.functional.conv2d(img_tensor, sobel_filter.unsqueeze(1))

           
            self.processed_image = transforms.ToPILImage()(edges.squeeze(0))
            self.display_image(self.processed_image)

    def display_image(self, img):
        root = Tk()
        root.title("Image Processor")

        img_tk = ImageTk.PhotoImage(img)
        canvas = Canvas(root, width=img.width, height=img.height)
        canvas.pack()
        canvas.create_image(0, 0, anchor="nw", image=img_tk)

        button = Button(root, text="Edge Detection", command=self.edge_detector)
        button.pack()

        root.mainloop()

if __name__ == "__main__":
    processor = ImageProcessor()
    processor.img_input()
