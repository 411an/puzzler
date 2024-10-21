import numpy as np
import cv2

# Shapes

def create_canvas(size):
    return np.zeros((size, size), dtype=np.uint8)

def draw_circle(size):
    canvas = create_canvas(size)
    center = size // 2
    radius = size // 3
    cv2.circle(canvas, (center, center), radius, 255, thickness=1)
    return canvas

def draw_square(size):
    canvas = create_canvas(size)
    side = size // 2
    top_left = (size - side) // 2
    bottom_right = (top_left + side, top_left + side)
    cv2.rectangle(canvas, (top_left, top_left), bottom_right, 255, thickness=1)
    return canvas

def draw_triangle(size):
    canvas = create_canvas(size)
    pts = np.array([[size // 2, size // 4], [size // 4, 3 * size // 4], [3 * size // 4, 3 * size // 4]], np.int32)
    cv2.polylines(canvas, [pts], isClosed=True, color=255, thickness=1)
    return canvas

def draw_star(size):
    canvas = create_canvas(size)
    center = size // 2
    radius_outer = size // 3
    radius_inner = radius_outer // 2
    points = []
    for i in range(5):
        outer_x = center + radius_outer * np.cos(2 * np.pi * i / 5 - np.pi / 2)
        outer_y = center + radius_outer * np.sin(2 * np.pi * i / 5 - np.pi / 2)
        points.append([outer_x, outer_y])
        inner_x = center + radius_inner * np.cos(2 * np.pi * (i + 0.5) / 5 - np.pi / 2)
        inner_y = center + radius_inner * np.sin(2 * np.pi * (i + 0.5) / 5 - np.pi / 2)
        points.append([inner_x, inner_y])
    points = np.array(points, np.int32)
    cv2.polylines(canvas, [points], isClosed=True, color=255, thickness=1)
    return canvas

def draw_hexagon(size):
    canvas = create_canvas(size)
    center = size // 2
    radius = size // 3
    points = []
    for i in range(6):
        x = center + radius * np.cos(2 * np.pi * i / 6)
        y = center + radius * np.sin(2 * np.pi * i / 6)
        points.append([x, y])
    points = np.array(points, np.int32)
    cv2.polylines(canvas, [points], isClosed=True, color=255, thickness=1)
    return canvas

def draw_cross(size):
    canvas = create_canvas(size)
    center = size // 2
    arm_length = size // 2
    arm_width = size // 4
    cv2.rectangle(canvas, (center - arm_width // 2, center - arm_length // 2), (center + arm_width // 2, center + arm_length // 2), 255, thickness=-1)
    cv2.rectangle(canvas, (center - arm_length // 2, center - arm_width // 2), (center + arm_length // 2, center + arm_width // 2), 255, thickness=-1)
    return canvas

def draw_turned_cross(size):
    canvas = create_canvas(size)
    center = size // 2
    arm_length = size // 2
    arm_width = size // 4
    cv2.rectangle(canvas, (center - arm_width // 2, center - arm_length // 2), (center + arm_width // 2, center + arm_length // 2), 255, thickness=-1)
    cv2.rectangle(canvas, (center - arm_length // 2, center - arm_width // 2), (center + arm_length // 2, center + arm_width // 2), 255, thickness=-1)
    image = rotation(canvas)
    return image

def draw_turned_square(size):
    canvas = create_canvas(size)
    side = size // 2
    top_left = (size - side) // 2
    bottom_right = (top_left + side, top_left + side)
    cv2.rectangle(canvas, (top_left, top_left), bottom_right, 255, thickness=1)
    image = rotation(canvas)
    return image

def draw_heart(size):
    canvas = create_canvas(size)
    scale = size / 4  # Size
    points = []
    for x in range(-int(2*scale), int(2*scale)):
        for y in range(-int(2*scale), int(2*scale)):
            xf = x / scale
            yf = y / scale
            if (xf**2 + yf**2 - 1)**3 - xf**2 * yf**3 <= 0:
                points.append((int(x + size//2), int(y + size//2)))

    if points:
        pts = np.array(points, np.int32)
        cv2.polylines(canvas, np.int32([pts]), isClosed=True, color=255, thickness=1)
        # cv2.fillPoly(canvas, np.int32([pts]), 255)
    image = cv2.rotate(canvas, cv2.ROTATE_180)
    return image

def rotation(img):
    (h, w) = img.shape[:2]
    center = (int(w / 2), int(h / 2))
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h))
    return rotated

def get_line_shapes():
    contours_count = 6
    contours_list = []
    for i in range(1, contours_count + 1):
        canvas = np.zeros((50, 50), dtype=np.uint8)
        for j in range(i):
            start_point = (0, j * (50 // contours_count))
            end_point = (50, j * (50 // contours_count))
            cv2.line(canvas, start_point, end_point, 255, 1)
        contours_list.append(canvas)
    
    return contours_list

def list_of_shapes(size=50):

    key_size = size

    typical_shapes = [
        draw_circle(key_size),
        draw_square(key_size),
        draw_triangle(key_size),
        draw_star(key_size),
        draw_hexagon(key_size),
        draw_cross(key_size),
        draw_turned_cross(key_size),
        draw_turned_square(key_size),
        draw_heart(key_size)
    ]
    
    return typical_shapes

# Display all shapes

#ts = list_of_shapes()
#for i, shape in enumerate(ts):
#    #print(shape.shape[0])
#    cv2.imshow(f'Shape {i+1}', shape)
#    cv2.waitKey(0)
#cv2.destroyAllWindows()

