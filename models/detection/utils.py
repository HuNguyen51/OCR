import numpy as np
import cv2

from configs.detection_config import shrink_ratio, target_size, threshold

def order_points(pts):
    """
    Sắp xếp 4 điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left.
    Thuật toán:
      - Top-left có tổng (x+y) nhỏ nhất.
      - Bottom-right có tổng lớn nhất.
      - Top-right có hiệu (x-y) nhỏ nhất.
      - Bottom-left có hiệu lớn nhất.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Tổng của tọa độ
    s = pts.sum(axis=1)
    # Hiệu tọa độ
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]  # top-left
    br = pts[np.argmax(s)]  # bottom-right
    tr = pts[np.argmin(diff)]  # top-right
    bl = pts[np.argmax(diff)]  # bottom-left

    if tl[1] < tr[1] and (np.linalg.norm(tl-tr) < np.linalg.norm(tl-bl)): # tl < tr
        # chéo sắc
        tr = pts[np.argmin(s)] # 0
        bl = pts[np.argmax(s)]   # 2
        br = pts[np.argmin(diff)]  # 1
        tl = pts[np.argmax(diff)] # 3

    elif tl[1] > tr[1] and (np.linalg.norm(tl-tr) < np.linalg.norm(tl-bl)): # tl > tr
        # chéo quyền
        bl = pts[np.argmin(s)] # 0
        tr = pts[np.argmax(s)]   # 2
        tl = pts[np.argmin(diff)]  # 1
        br = pts[np.argmax(diff)] # 3
    
    rect[0] = tl
    rect[1] = tr
    rect[2] = br
    rect[3] = bl
    return rect

def four_point_transform(image, pts):
    """
    Biến đổi phối cảnh: từ 4 điểm của một polyline trong ảnh gốc chuyển thành một ảnh
    hình chữ nhật chuẩn.
    
    :param image: Ảnh gốc
    :param pts: Mảng numpy chứa 4 điểm (dạng [[x1, y1], [x2, y2], ...])
    :return: Ảnh đã được biến đổi (warped) với dạng hình chữ nhật.
    """
    # Bước 1: Sắp xếp lại các điểm
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Bước 2: Tính toán kích thước mới của ảnh (chiều rộng và chiều cao)
    # Chiều rộng: khoảng cách giữa bottom-right & bottom-left và giữa top-right & top-left
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    
    # Chiều cao: khoảng cách giữa top-right & bottom-right và giữa top-left & bottom-left
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    # Bước 3: Định nghĩa các điểm đích cho ảnh mới (kích thước mới)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Bước 4: Tính ma trận chuyển đổi phối cảnh
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # Bước 5: Áp dụng phép biến đổi lên ảnh gốc để thu được ảnh chữ nhật
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def generate_quad(image_size, text_polygons, shrink_ratio=shrink_ratio):
    """
    Generate QUAD ground truth (score map and geometry map)
    
    Args:
        image_size: tuple (h, w)
        text_polygons: list of polygons, each polygon is a list of points (should be 4 points)
    
    Returns:
        score_map: binary mask (1 for text regions, 0 for non-text regions)
        geo_map: geometry map with 8 channels (x,y coordinates of 4 corners)
    """
    h, w = image_size
    score_map = np.zeros((h, w), dtype=np.uint8)
    geo_map = np.zeros((8, h, w), dtype=np.float32)
    training_mask = np.zeros(image_size, dtype=np.uint8)
    
    for polygon in text_polygons:
        # Calculate shrinked polygon to ignore boundary pixels
        polygon = np.array(polygon).reshape(-1, 2)
        # Find the center of the polygon
        center = np.mean(polygon, axis=0)
        # Shrink the polygon points toward the center
        shrinked_polygon = []
        for point in polygon:
            vector = point - center
            shrinked_point = center + vector * (1 - shrink_ratio)
            shrinked_polygon.append(shrinked_point)
        # Convert to numpy array
        shrinked_polygon = np.array(shrinked_polygon, dtype=np.int32)

        area = cv2.contourArea(polygon)

        quad = np.array(polygon).reshape(4, 2)
        
        # Sort the quadrilateral points in clockwise order starting from top-left
        # This is important for consistent representation
        center = np.mean(quad, axis=0)
        angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        quad = quad[sorted_indices]

        cv2_polygon = quad.reshape((-1, 1, 2)).astype(np.int32)
        # Fill the polygon in the score map
        cv2.fillPoly(score_map, [shrinked_polygon], 1)

        geo_matrix = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(geo_matrix, [cv2_polygon], 1)
        
        # Calculate geometry map values
        # For each pixel in the text polygon
        points = np.argwhere(geo_matrix == 1)
        for point in points:
            y, x = point
            # Store the coordinates of the quadrilateral
            for i in range(4):
                geo_map[i*2, y, x] = quad[i, 0] - x    # x-coordinate offset
                geo_map[i*2+1, y, x] = quad[i, 1] - y  # y-coordinate offset
        
        # -- trainning mask -- 
        if area > 200:
            cv2.fillPoly(training_mask, [polygon], 1)
    return score_map, geo_map, training_mask

def decode_quad(score_map, geo_map, threshold=threshold):
    """
    Simpler approach to decode based on finding contours in the score map
    """
    polygons = []
    
    # Threshold the score map
    binary = (score_map > threshold).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Create mask for this contour
        mask = np.zeros_like(score_map)
        cv2.drawContours(mask, [contour], -1, 1, -1)
        # Get points within the contour
        points = np.argwhere(mask > 0)

        if len(points) == 0:
            continue
        # Calculate average quad for this contour
        quad = np.zeros((4, 2), dtype=np.float32)
        count = 0
        for y, x in points:
            for i in range(4):
                quad[i, 0] += geo_map[i*2, y, x] + x
                quad[i, 1] += geo_map[i*2+1, y, x] + y
            count += 1

        if count > 0:
            quad /= count
            polygons.append(quad.reshape(-1).tolist())
    
    return polygons, contours

def resize_and_pad(image, target_size=target_size):
    # Calculate ratio
    h, w = image.shape[:2]
    ratio = min(target_size[0] / h, target_size[1] / w)
    
    # Resize image
    new_h, new_w = int(h * ratio), int(w * ratio)
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create padding
    if len(image.shape)==2:
        padded = np.zeros((target_size[0], target_size[1]), dtype=np.uint8)
    else:
        padded = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    
    # Return image and scale factor
    return padded, ratio

def transform(image):
  # Chuyển đổi thủ công
  image = image.astype(float) / 255.0  # Chuyển từ [0, 255] sang [0, 1]
  mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
  std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
  return (image - mean) / std  # Chuẩn hóa


def generate_score_mask(image_size, text_polygons, shrink_ratio=shrink_ratio):
    h, w = image_size
    score_map = np.zeros((h, w), dtype=np.uint8)
    training_mask = np.zeros(image_size, dtype=np.uint8)
    
    for polygon in text_polygons:
        # Calculate shrinked polygon to ignore boundary pixels
        polygon = np.array(polygon).reshape(-1, 2)
        # Find the center of the polygon
        center = np.mean(polygon, axis=0)
        # Shrink the polygon points toward the center
        shrinked_polygon = []
        for point in polygon:
            vector = point - center
            shrinked_point = center + vector * (1 - shrink_ratio)
            shrinked_polygon.append(shrinked_point)
        # Convert to numpy array
        shrinked_polygon = np.array(shrinked_polygon, dtype=np.int32)
        area = cv2.contourArea(polygon)

        # sort points
        quad = np.array(polygon).reshape(4, 2)
        center = np.mean(quad, axis=0)
        angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        quad = quad[sorted_indices]
        cv2_polygon = quad.reshape((-1, 1, 2)).astype(np.int32)

        # Fill the polygon in the score map
        cv2.fillPoly(score_map, [shrinked_polygon], 1)
    
        # -- trainning mask -- 
        if area > 200:
            cv2.fillPoly(training_mask, [cv2_polygon], 1)
    return score_map, training_mask