import cv2
import easygui
import matplotlib.pyplot as plt
import numpy as np

import tkinter as tk
from tkinter import filedialog , Button, TOP
from PIL import ImageTk, Image

from KmeansColorQuantization import kmeans_quantize

def cartoonify(ImagePath , k=10):
    if ImagePath is None:
        print("No image selected.")
        return
    

    originalImage = cv2.imread(ImagePath)

    #Checking if the image is loaded properly
    if originalImage is None:
        print("Cannot find any image, choose appropriate file.")
        return
    

    originalImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)  
   
    height, width = originalImage.shape[:2]
    if width > 1200 or height > 800:
        scale = min(1200/width, 800/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        originalImage = cv2.resize(originalImage, (new_width, new_height))

     # Quantize colors
    quantized_img = kmeans_quantize(originalImage, k)
    
    grayScaleImage = cv2.cvtColor(originalImage , cv2.COLOR_RGB2GRAY)
    
    smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)

    getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 8)
   
   

    kernel = np.ones((2,2), np.uint8)
    getEdge = cv2.dilate( getEdge, kernel, iterations=1)
    getEdge = cv2.erode( getEdge, kernel, iterations=1)

    #applying bilateral filter to remove the noise and keep edges sharp
    smooth = cv2.bilateralFilter(quantized_img, 15, 50, 50)

    edges_colored = cv2.cvtColor(getEdge, cv2.COLOR_GRAY2RGB)

    cartoonImage = cv2.bitwise_and(smooth, smooth, mask=getEdge)

    cartoonImage = cv2.addWeighted(cartoonImage, 0.8, smooth, 0.2, 0)

    return originalImage, cartoonImage



def display_before_after(original, cartoon):
   
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    
    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=16, fontweight='bold', pad=20)
    axes[0].axis('off')
    
    
    axes[1].imshow(cartoon)
    axes[1].set_title('Cartoonified Image', fontsize=16, fontweight='bold', pad=20)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, wspace=0.1)
    plt.show()



def upload():
    try:
        ImagePath = easygui.fileopenbox(title= "Select an image file", filetypes = ["*.jpg", "*.png", "*.bmp", "*.tiff", "*.webp"])
        
        if ImagePath:
            print("Processing image...")
            original, cartoon = cartoonify(ImagePath, k=10)
            display_before_after(original, cartoon)
            print("Cartoonification complete!")

    except Exception as e:
        print(f"Error: {e}")


#Creating main window
top = tk.Tk()
top.geometry("800x600")
top.title("Cartoonify App")
top.configure(bg='lightgray')

title_label = tk.Label(top, text="🎨 Image Cartoonifier", font=('Arial',26,'bold'), bg='lightgray', fg='darkblue')
title_label.pack(pady=40)


desc_label = tk.Label(top,  text="Transform your photos into stunning cartoon-style artwork\nusing advanced K-means color quantization",font=('Arial', 12), bg='lightgray', fg='gray')
desc_label.pack(pady=10)


upload_btn= Button(top, text="Cartoonify an Image",command=upload,padx=50,pady=10)
upload_btn.configure(background='blue',foreground='white',font=('arial', 12, 'bold'), cursor='hand2')
upload_btn.pack(pady=50)


features_frame = tk.Frame(top, bg='lightgray')
features_frame.pack(pady=20)


instructions = tk.Label(top, text="Choose an image and see the cartoonification process!",font=('arial', 10), bg='lightgray', fg='gray')
instructions.pack(pady=20)


top.mainloop()