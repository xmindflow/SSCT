import numpy as np
import cv2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_image(img_file, img_size):
    im = cv2.imread(img_file)
    im = cv2.resize(im, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32')/255.]))
    
    return data


def compute_sobel_gradients(img_rgb):
    """
    Compute Sobel gradients on the given RGB image.

    :param img_rgb: Input RGB image.
    :type img_rgb: numpy.ndarray
    :return: Gradients on X, Y, and both X and Y axes.
    :rtype: torch.Tensor, torch.Tensor, torch.Tensor
    """
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)/255.
    
    sobelx = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    sobely = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    sobelxy = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)
    
    sub_y_x = torch.tensor(sobely - sobelx).to(device)
    sub_xy_x = torch.tensor(sobelxy - sobelx).to(device)
    sub_xy_y = torch.tensor(sobelxy - sobely).to(device)
    
    return sub_y_x, sub_xy_x, sub_xy_y


def create_mask(pred, GT):
    
    kernel = np.ones((5, 5), np.uint8) 
    dilated_GT = cv2.dilate(GT, kernel, iterations = 2)

    mult = pred * GT        
    unique, count = np.unique(mult[mult !=0], return_counts=True)
    cls= unique[np.argmax(count)]
    
    lesion = np.where(pred==cls, 1, 0) * dilated_GT
    
    return lesion


def dice_metric(A, B):
    intersect = np.sum(A * B)
    fsum = np.sum(A)
    ssum = np.sum(B)
    dice = (2 * intersect ) / (fsum + ssum)
    
    return dice    


def hm_metric(A, B):
    intersection = A * B
    union = np.logical_or(A, B)
    hm_score = (np.sum(union) - np.sum(intersection)) / np.sum(union)
    
    return hm_score


def xor_metric(A, GT):
    intersection = A * GT
    union = np.logical_or(A, GT)
    xor_score = (np.sum(union) - np.sum(intersection)) / np.sum(GT)
    
    return xor_score