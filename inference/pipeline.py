import torch
import cv2
import numpy as np
from models.detection.utils import resize_and_pad, transform, four_point_transform
from utils.visualization import imshow, bi_imshow
from doctr.models import db_mobilenet_v3_large, db_resnet50

from models.recognition.utils import transform as recognition_transform
from PIL import Image

class Task:
    detection = 'text_detection'
    recognition = 'text_regconition'
    visualize = 'visualize_attention_weights'
class Model:
    my_model = 'my_model'
    db_resnet50 = 'db_resnet50'
    db_mobilenet_v3_large = 'db_mobilenet_v3_large'

class Pipe:
    def __init__(self, task):
        self.task = task
    
    def pipeline(self, **kwargs):
        if self.task == Task.detection:
            return self.detection(**kwargs)
        if self.task == Task.recognition:
            return self.recognition(**kwargs)
        if self.task == Task.visualize:
            return self.visualize(**kwargs)

    def detection(self, model, image_path=None, polyon=True, **kwargs):
        if polyon:
            return self.__polygon_detector(model, image_path, **kwargs)
        else:
            return self.__dbnet(model, image_path, **kwargs)

    def recognition(self, model, image=None, vocab=None, loader=None):
        if image:
            return self.__generate_text(model, image, vocab)
        if loader:
            return self.__generate_text_random(model, vocab, loader)

    def visualize(self, model, image=None, vocab=None, loader=None):
       if image:
           return self.__visualize_attention(model, image, vocab)
       else:
            self.__visualize_attention_random(model, vocab, loader)
    # text detection
 
    def __polygon_detector(self, model, image_path, show=False, expand_ratio=0.1, thresh_hold=0.2):
        origin_image = cv2.imread(image_path)
        origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
        origin_image, ratio = resize_and_pad(origin_image)
        image = transform(origin_image.copy()).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = torch.unsqueeze(image, 0)

        model.eval()
        model.to('cpu')

        with torch.no_grad():
            score = model(image)
        score = score[0].squeeze(0).numpy()
        score = (score > thresh_hold).astype(np.uint8) * 255

        contours, _ = cv2.findContours(score, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        expanded_polygons = []
        for contour in contours:
            if cv2.contourArea(contour) < 50:
                continue
            rect = cv2.minAreaRect(contour)
            polygons = cv2.boxPoints(rect)
            polygons = np.intp(polygons) 

            center = np.mean(polygons, axis=0)
            expanded_polygon = []
            for point in polygons:
                vector = point - center
                expanded_point = center + vector / (1 - expand_ratio)
                expanded_polygon.append(expanded_point)
            expanded_polygons.append(expanded_polygon)
        expanded_polygons = np.array(expanded_polygons).astype(int)

        if show:
            show_image = origin_image.copy()
            cv2.polylines(show_image, expanded_polygons, True, (0, 0, 255), 2)
            bi_imshow(score)
            imshow(show_image)

        sub_images = []
        for polygon in expanded_polygons:
            sub_image = four_point_transform(origin_image.copy(), polygon)
            sub_images.append(sub_image)
        return sub_images

    def __rect_to_polygon(self, pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2

        # Đảm bảo thứ tự điểm theo chiều kim đồng hồ (bắt đầu từ top-left)
        top_left = (min(x1, x2), min(y1, y2))
        top_right = (max(x1, x2), min(y1, y2))
        bottom_right = (max(x1, x2), max(y1, y2))
        bottom_left = (min(x1, x2), max(y1, y2))

        return np.array([top_left, top_right, bottom_right, bottom_left])

    def __dbnet(self, model, image_path, show=False, thresh_hold=0.2):
        origin_image = cv2.imread(image_path)
        origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
        origin_image, ratio = resize_and_pad(origin_image, (1024, 1024))
        image = torch.from_numpy(origin_image.copy()).permute(2, 0, 1)
        image = image.type(torch.float32)
        image = torch.unsqueeze(image, 0)

        # model.eval()
        model.to('cpu')
        with torch.no_grad():
            out = model(image)

        rects = [(points[:-1] * 1024).astype(int) for points in out['preds'][0]['words'] if points[-1] > thresh_hold]
        polygons = []

        for rect in rects:
            x1,y1,x2,y2 = rect
            polygon  = self.__rect_to_polygon((x1,y1),(x2,y2))
            # area = cv2.contourArea(polygon)
            # if area < 100:
            #     continue
            polygons.append(polygon)
        polygons = np.array(polygons).astype(int)
        if show:
            show_image = origin_image.copy()
            cv2.polylines(show_image, polygons, True, (0, 0, 255), 2)
            imshow(show_image) 

        sub_images = []
        for polygon in polygons:
            sub_image = four_point_transform(origin_image.copy(), polygon)
            sub_images.append(sub_image)
        return sub_images

    # text regconition
    def __generate_text(self, model, image, vocab):
        image = recognition_transform(image)
    
        model = model.to('cpu')
        image = image.to('cpu')

        pred_txt, _ ,_  = model.generate_text(image.unsqueeze(0), vocab)
        return pred_txt[0], None, None
    
    def __generate_text_random(self, model, vocab, loader):
        img, txt, path = loader.get_random_image()

        model = model.to('cpu')
        img = img.to('cpu')

        pred_txt, _ ,_  = model.generate_text(img.unsqueeze(0), vocab)
        return pred_txt[0], txt, path

    def __visualize_attention(self, model, image, vocab):
        image = recognition_transform(image)

        model = model.to('cpu')
        image = image.to('cpu')

        pred_txt, f ,a  = model.generate_text(image.unsqueeze(0), vocab)
        model.visualize_attention(image, '', pred_txt[0], f, a)
        return image, '', pred_txt[0], f, a
    
    def __visualize_attention_random(self, model, vocab, loader):
        img, txt, path = loader.get_random_image()

        model = model.to('cpu')
        img = img.to('cpu')

        pred_txt, f ,a  = model.generate_text(img.unsqueeze(0), vocab)
        model.visualize_attention(img, txt, pred_txt[0], f, a)
        return img, txt, pred_txt[0], f, a