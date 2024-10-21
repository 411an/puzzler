import cv2
import numpy as np
from sklearn.cluster import KMeans
from shapely.geometry import LineString, Polygon
import shapelist

# Looking for the line intersections
def find_line_intersection(start_point, end_point, contour):
    """
    The first intersection
    
    Parameters:
    - start_point: tuple (x, y) - Start of line
    - end_point: tuple (x, y) - End of line
    - contour: numpy array of shape (n, 1, 2) - Checking the intersections contour
    
    Returns:
    - tuple (x, y): the first point or None if no intersection
    """
    # Turn the segment to the line
    line = LineString([start_point, end_point])
    
    # Turn the contour in polygon , deleting one dimencion
    polygon = Polygon(contour.reshape(-1, 2))

    # Looking for the intersection
    intersection = line.intersection(polygon)

    # If not
    if intersection.is_empty:
        return None
    
    # If only one
    elif intersection.geom_type == 'Point':
        return (intersection.x, intersection.y)
    
    # If many return first
    elif intersection.geom_type == 'MultiPoint':
        return (intersection.geoms[0].x, intersection.geoms[0].y)
    
    # If line return the first dot
    elif intersection.geom_type == 'LineString':
        return tuple(intersection.coords[0])
    
    # Multiline
    elif intersection.geom_type == 'MultiLineString':
        return tuple(intersection.geoms[0].coords[0])
    
    return None

# The main function for detecting key shapes
def extract_shapes(image_path, side_size=50, canny_low=50, canny_high=150, border_threshold=0.04, kernel_size = 3):
    
    image = None

    if isinstance(image_path, str):
        # If path
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    elif isinstance(image_path, np.ndarray):
        # If np array
        print(len(image_path.shape))
        if len(image_path.shape) == 3:  # if colorful
            image = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported input type")


    # Step 1: Looking for edges
    edges = cv2.Canny(image, canny_low, canny_high)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    #cv2.imshow(f'edges', edges)
    #cv2.waitKey(0)

    # Step 2: Looking for contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f'Contours: {len(contours)}')
    
    # If the wrong key picture
    if len(contours) < 3:
        raise KeyError("Wrong numbers of contours")    

    lowest_level_contours = []
    height, width = image.shape[:2]

    # Filtering the low-level contours
    for i, h in enumerate(hierarchy[0]):
        if h[2] == -1:  # Weuse only low level contours
            contour = contours[i]
            x, y, w, h = cv2.boundingRect(contour)
            # We use only contours far from borders
            if (
                x > border_threshold * width and
                x + w < (1 - border_threshold) * width and
                y > border_threshold * height and
                y + h < (1 - border_threshold) * height
                and cv2.contourArea(contour) > 10
            ):
                lowest_level_contours.append(contour)

    print(f'Low level contours: {len(lowest_level_contours)}')

    # Step 3: Contours centers
    centers = []
    for contour in lowest_level_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cX = x + w // 2 
        cY = y + h // 2 
        centers.append((cX, cY))

    print(f'We found centers: {len(centers)}')

    if not centers:
        raise KeyError("Not enough keys")
        return []

    # Step 4: KMeans clustering of contours centers
    n_clusters = len(lowest_level_contours) 
    if len(centers) > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(centers)
        labels = kmeans.labels_

        # Filtering centers and contours to delete wrong ones
        filtered_centers = [centers[i] for i in range(len(centers))]
        filtered_contours = [lowest_level_contours[i] for i in range(len(lowest_level_contours))]
    else:
        filtered_centers = centers
        filtered_contours = lowest_level_contours

    contours_to_remove = set()

    for i, contour in enumerate(filtered_contours):
        # Here is a real center of contour (of bounding rectangle)
        x, y, w, h = cv2.boundingRect(contour)
        cX = x + w // 2 
        cY = y + h // 2 

        # Vertical line up
        for other_contour in filtered_contours.copy():
            if not np.array_equal(contour, other_contour):
                intersection_other = find_line_intersection((x + w // 2, y), (x + w // 2, 0), other_contour)
                if intersection_other is not None and intersection_other[1] < y:
                    contours_to_remove.add(tuple(other_contour.flatten()))
        # Vertical line down
        for other_contour in filtered_contours.copy():
            if not np.array_equal(contour, other_contour):
                intersection_other = find_line_intersection((x + w // 2, y + h), (x + w // 2, image.shape[0] - 1), other_contour)
                if intersection_other is not None and intersection_other[1] > y + h:
                    contours_to_remove.add(tuple(other_contour.flatten()))
    
    # We destory other contours that exist between the border and the center
    filtered_contours_final = []
    for contour in filtered_contours:
        if tuple(contour.flatten()) not in contours_to_remove:
            filtered_contours_final.append(contour)

    filtered_contours = filtered_contours_final

    print(f'Centers after clustering: {len(filtered_centers)}')
    print(f'Contours after filtering: {len(filtered_contours)}')

    if len(filtered_contours) < 3 or len(filtered_contours) > 6:
        raise KeyError("Wrong number of keys")

    # Step 5: Getting and frawing contours
    key_size = 0
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        canvas_width = int(w * 1.1)  # Add 10% for canvas
        canvas_height = int(h * 1.1)
        key_size = max(key_size, canvas_width, canvas_height)

    if key_size < side_size:
        key_size = 50
    
    extracted_shapes = []
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        canvas_width = canvas_height = key_size
        canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        
        offset_x = (canvas_width - w) // 2
        offset_y = (canvas_height - h) // 2
        cv2.drawContours(canvas, [contour - np.array([x, y]) + np.array([offset_x, offset_y])], -1, 255, thickness=1)

        extracted_shapes.append(canvas)

    print(f'Shapes: {len(extracted_shapes)}')

    return extracted_shapes, key_size
