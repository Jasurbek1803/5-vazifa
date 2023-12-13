#Shodmonov Jasurbek
import concurrent.futures
import cv2
import numpy as np

def apply_gaussian_filter(image, kernel_size=(5, 5), sigma=1.0):
    return cv2.GaussianBlur(image, kernel_size, sigma)

def parallel_filter_images(images, filter_function, *args, **kwargs):
    filtered_images = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
      
        futures = [executor.submit(filter_function, image, *args, **kwargs) for image in images]

    
        concurrent.futures.wait(futures)

       
        filtered_images = [future.result() for future in futures]

    return filtered_images


image1 = cv2.imread('khvicha.jpg')
image2 = cv2.imread('messi.jpg')


filtered_images = parallel_filter_images([image1, image2], apply_gaussian_filter, kernel_size=(5, 5), sigma=1.0)

for i, filtered_image in enumerate(filtered_images):
    cv2.imshow(f"Filtered Image {i + 1}", filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
