import cv2
import numpy as np

# Is the image grey?
def graycheck(image):
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    return gray

# If the line is direct the function makes it stright
def strighter(data):

    non_zero_data = [x for x in data if x > 0]
    zero_indices = [i for i, x in enumerate(data) if x == 0]

    if non_zero_data:
        values, counts = np.unique(non_zero_data, return_counts=True)
        most_frequent_value = values[np.argmax(counts)]
        threshold = 0.7 * len(non_zero_data)

        new_non_zero_data = []
        for x in non_zero_data:
            if abs(x - most_frequent_value) <= 1:
                new_non_zero_data.append(most_frequent_value)
            else:
                new_non_zero_data.append(x)

        non_zero_data = new_non_zero_data        

        values, counts = np.unique(non_zero_data, return_counts=True) # Вторичное обновление

        # if the line dots are the same in > 90% cases
        if np.max(counts) >= threshold:
            # its line with only this one dot
            non_zero_data = [most_frequent_value] * len(non_zero_data)
            
    # Renew the line
    new_list = []
    non_zero_index = 0
    for i in range(len(data)):
        if i in zero_indices:
            new_list.append(0)
        else:
            if non_zero_index < len(non_zero_data):
                new_list.append(non_zero_data[non_zero_index])
                non_zero_index += 1

    return new_list

# Contour to the normilized curve
def get_normalized_curve(img):
    height, width = img.shape
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=lambda c: cv2.arcLength(c, True))

    perimeter = 2 * width + 2 * height
    distances_to_contour = np.zeros(perimeter, dtype=np.int32)

    # The distance between points
    def calculate_distance(point1, point2):
        return np.sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2))

    # Find the first intersection
    def find_line_intersection(start_point, end_point, contour):
    
        x1, y1 = start_point
        x2, y2 = end_point

        # Where we move?
        dx = 1 if x2 > x1 else -1 if x2 < x1 else 0
        dy = 1 if y2 > y1 else -1 if y2 < y1 else 0

        x, y = x1, y1

        # Checks
        if dx != 0:  # For vertical
            while x1 <= x <= x2 or x2 <= x <= x1:  
                if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                    return (x, y)
                x += dx 
                y += dy # if diagonal
        else:  # Horizontal
            while y1 <= y <= y2 or y2 <= y <= y1:
                if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                    return (x, y)
                x += dx # if diagonal
                y += dy

        return None

    i = 0

    # Top side
    for x in range(width):
        start_point = (x, 0)
        end_point = (x, height - 1)
        intersection_point = find_line_intersection(start_point, end_point, cnt)
        if intersection_point:
            distances_to_contour[i] = calculate_distance(start_point, intersection_point)
        else:
            distances_to_contour[i] = 0  # No intersections
        i += 1

    # Right side
    for y in range(height):
        start_point = (width - 1, y)
        end_point = (0, y)
        intersection_point = find_line_intersection(start_point, end_point, cnt)
        if intersection_point:
            distances_to_contour[i] = calculate_distance(start_point, intersection_point)
        else:
            distances_to_contour[i] = 0  # No intersections
        i += 1

    # Bottom side
    for x in range(width - 1, -1, -1):
        start_point = (x, height - 1)
        end_point = (x, 0)
        intersection_point = find_line_intersection(start_point, end_point, cnt)
        if intersection_point:
            distances_to_contour[i] = calculate_distance(start_point, intersection_point)
        else:
            distances_to_contour[i] = 0  # No intersections
        i += 1

    # Left side
    for y in range(height - 1, -1, -1): 
        start_point = (0, y)
        end_point = (width - 1, y)
        intersection_point = find_line_intersection(start_point, end_point, cnt)
        if intersection_point:
            distances_to_contour[i] = calculate_distance(start_point, intersection_point)
        else:
            distances_to_contour[i] = 0  # No intersections
        i += 1

    #  Normalize

    # 4 parts of contour
    top = distances_to_contour[:width]
    right = distances_to_contour[width:width + height]
    bottom = distances_to_contour[width + height:2 * width + height]    
    left = distances_to_contour[2 * width + height:]    # The only side we need to normilize for the height
    # because we measure the distance from the bottom side to the contour
    
    # Normalize 4 parts
    top = (top / (height - 1) * 100).astype(np.int32)
    top = strighter(top)
    right = (right / (width - 1) * 100).astype(np.int32)
    right = strighter(right)
    bottom = (bottom / (height - 1) * 100).astype(np.int32)    
    bottom = strighter(bottom)
    left = (left / (height - 1) * 100).astype(np.int32)
    left = strighter(left)

    # Re-union
    normalized_curve = np.concatenate((top, right, bottom, left))

    return normalized_curve

# Form-factor of shape
def get_ff(image):

    gray = graycheck(image)
    
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = contours[0]
        area1 = cv2.contourArea(contour)
        perimeter1 = cv2.arcLength(contour, True)
        form_factor = area1 / (perimeter1 ** 2)
    else:
        form_factor = None

    return form_factor

# Pixel dynamics for shape
def width_pixel_dynamics(canvas):
    # Получение высоты изображения
    height = canvas.shape[0]

    # Disassemble the shape for lines
    segments = np.array_split(canvas, height, axis=0)

    dynamics = []
    prev_min, prev_max = None, None  # Для сглаживания
    prev_width = 0  # Для хранения предыдущей ширины
    direction = 0  # 1 - расширение, -1 - сужение, 0 - неопределено
    first_zero = None
    first_non_zero = None    

    for segment_index, segment in enumerate(segments):
        # Non-empty lines
        non_zero_pixels = np.nonzero(segment)

        # Check for empty lines
        if non_zero_pixels[0].size == 0:
            # Smoothing
            if prev_min is not None and prev_max is not None:
                if direction == 1:  # Inflation <>
                    max_x = prev_max + 1
                    min_x = prev_min
                elif direction == -1:  # Compression ><
                    max_x = prev_max
                    min_x = prev_min - 1
                else:  # Not clear, just repeat the line
                    max_x = prev_max
                    min_x = prev_min
            else:
                max_x = 0
                min_x = 0
                # Last zero
            if first_zero is None:
                first_zero = segment_index                
        else:
            # First non zero
            if first_non_zero is None and prev_min == 0 and prev_max == 0: first_non_zero = segment_index            
            # Renew last zero because we found non zero
            first_zero = None
        
            # Min and max
            min_x = np.min(non_zero_pixels[1])
            max_x = np.max(non_zero_pixels[1])

            # Direction
            if prev_width > 0:
                direction = 1 if max_x - min_x > prev_width else -1 if max_x - min_x < prev_width else 0

        # How wide the segmenth
        width = max_x - min_x
        dynamics.append(width)

        prev_min, prev_max, prev_width = min_x, max_x, width

    dynamics = dynamics[first_non_zero:first_zero]
    dynamics = strighter(dynamics)

    return dynamics

# Pixel dynamics for shape (the same but for height)

def height_pixel_dynamics(canvas):
    width = canvas.shape[1]

    segments = np.array_split(canvas, width, axis=1)

    dynamics = []
    prev_min, prev_max = None, None
    prev_height = 0
    direction = 0  # 1 - inflation, -1 - compression, 0 - not clear
    first_zero = None
    first_non_zero = None    

    for segment_index, segment in enumerate(segments):
        non_zero_pixels = np.nonzero(segment)

        if non_zero_pixels[0].size == 0:

            if prev_min is not None and prev_max is not None:
                if direction == 1:
                    max_y = prev_max + 1
                    min_y = prev_min
                elif direction == -1:
                    max_y = prev_max
                    min_y = prev_min - 1
                else:
                    max_y = prev_max
                    min_y = prev_min
            else:
                max_y = 0
                min_y = 0

            if first_zero is None:
                first_zero = segment_index                
        else:

            if first_non_zero is None and prev_min == 0 and prev_max == 0: first_non_zero = segment_index            
            first_zero = None
        
            min_y = np.min(non_zero_pixels[0]) 
            max_y = np.max(non_zero_pixels[0])

            # Определение направления
            if prev_height > 0:
                direction = 1 if max_y - min_y > prev_height else -1 if max_y - min_y < prev_height else 0

        height = max_y - min_y  
        dynamics.append(height) 

        prev_min, prev_max, prev_height = min_y, max_y, height

    dynamics = dynamics[first_non_zero:first_zero]
    dynamics = strighter(dynamics)

    return dynamics