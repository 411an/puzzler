import cv2
import numpy as np

# The main code for grid recognizing

# Reading the picture of grid
def load_and_preprocess_image(image_path):

    if isinstance(image_path, str):
        # If its path
        img = cv2.imread(image_path)
    elif isinstance(image_path, np.ndarray):
        # If its numpy array
        if len(image_path.shape) == 3:  # Checking if it the colorful
            img = image_path
    else:
        raise ValueError("Unsupported input type")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray)
    edges = cv2.Canny(equalized_image, 50, 150)
    return img, edges

# Detect lines
def detect_lines(edges, sigma):
    return cv2.HoughLines(edges, 1, np.pi / 180, sigma)

# Separating lines to horizontal and vertical according the angle
def sort_lines(lines, theta_threshold=np.pi / 180):
    
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        rho, theta = line[0]  # rho and theta from lines
        if abs(theta) < theta_threshold or abs(theta - np.pi) < theta_threshold:  # Horisontal
            horizontal_lines.append((rho, theta))
        elif abs(theta - np.pi / 2) < theta_threshold:  # Vertical
            vertical_lines.append((rho, theta))

    return horizontal_lines, vertical_lines

# Drawing lines
def draw_lines(img, lines):
    for line in lines:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Crom the grid inside the side lines
def crop_image_by_lines(img, clean_img, h_lines, v_lines):
    if not h_lines or not v_lines:
        return img, clean_img, [], []

    # Corner of the square
    min_y = min(v_lines)
    max_y = max(v_lines)
    min_x = min(h_lines)
    max_x = max(h_lines)

    # Cut the images
    cropped_img = img[min_y:max_y, min_x:max_x]
    cropped_img_no_lines = clean_img[min_y:max_y, min_x:max_x]

    # Update lines coordinates

    updated_v_lines = []
    for line in v_lines:
        # If the line inside the square
        if min_y < line < max_y:
            updated_v_lines.append(line)

    updated_h_lines = []
    for line in h_lines:
        # If the line inside the square
        if min_x < line < max_x:
            updated_h_lines.append(line)

    updated_h_lines = [x - min_x for x in updated_h_lines]
    updated_v_lines = [y - min_y for y in updated_v_lines]

    return cropped_img, cropped_img_no_lines, updated_h_lines, updated_v_lines

# The main matrix
def visualize_intersections_matrix(filtered_h_lines, filtered_v_lines, img_shape, epsilon=1e-6):

    height, width = img_shape[:2]
    matrix = np.empty((len(filtered_h_lines), len(filtered_v_lines), 2), dtype=object)

    for h_index, (rho2, theta2) in enumerate(filtered_h_lines):
        for v_index, (rho1, theta1) in enumerate(filtered_v_lines):
            # Checking the good lines
            if abs(theta2 - theta1) % np.pi < 1e-6 or \
               abs(abs(theta2 - theta1) - np.pi) < 1e-6:  # Parallels
                matrix[h_index, v_index] = None 
            elif abs(abs(theta2 - theta1) - np.pi/2) < 1e-6:  # Straight angle
                # Dot of intersection
                if abs(theta1) < 1e-6: # Horizontal line
                    x = rho1
                    y = rho2
                elif abs(abs(theta1) - np.pi / 2) < 1e-6: # Vertical line
                    x = rho2
                    y = rho1
                else:
                    x = (rho2 * np.sin(theta1) - rho1 * np.sin(theta2)) / (np.cos(theta2) * np.sin(theta1) - np.cos(theta1) * np.sin(theta2))
                    y = (rho1 * np.cos(theta2) - rho2 * np.cos(theta1)) / (np.cos(theta2) * np.sin(theta1) - np.cos(theta1) * np.sin(theta2))
            else:
                # Stright horizontal line
                if abs(theta2) < 1e-6: 
                    x = rho2
                    y = (rho1 - x * np.cos(theta1)) / np.sin(theta1)
                # Stright vertical line
                elif abs(abs(theta1) - np.pi / 2) < 1e-6:
                    y = rho1
                    x = (rho2 - y * np.sin(theta2)) / np.cos(theta2)
                else:
                    # Other lines, system of equal
                    A = np.array([[np.cos(theta2), np.sin(theta2)],
                                  [np.cos(theta1), np.sin(theta1)]])
                    b = np.array([rho1, rho2])
                    x_coord, y_coord = np.linalg.solve(A, b)
                    print(x_coord, y_coord,h_index, v_index)
                    
                    # The order of coordinaates
                    if abs(theta1) < abs(theta2):  # If the line 1's angle is less
                        x = x_coord
                        y = y_coord
                    else:
                        y = x_coord
                        x = y_coord

            # Check with epsilon error
            if 0 - epsilon <= x <= width + epsilon and 0 - epsilon <= y <= height + epsilon:
                x, y = int(x), int(y) 
                matrix[h_index, v_index] = (x, y)
            else:
                matrix[h_index, v_index] = None
    return matrix

# averaging lines to remove unnecessary дштуы
def average_lines(intersection_matrix):
    horizontal_lines = []
    vertical_lines = []
    num_rows = len(intersection_matrix)
    num_cols = len(intersection_matrix[0]) # Columns

    # Averaging horizontal lines
    for i in range(num_rows):
        horizontal_line = [intersection_matrix[i][j][0] for j in range(num_cols) if intersection_matrix[i][j] is not None]
        if horizontal_line:
            horizontal_lines.append(round(np.mean(horizontal_line)))

    # Averaging vertical lines
    for j in range(num_cols):
        vertical_line = [intersection_matrix[i][j][1] for i in range(num_rows) if intersection_matrix[i][j] is not None]
        if vertical_line:
            vertical_lines.append(round(np.mean(vertical_line)))
    
    # KEEP ONLY UNIQ LINES
    horizontal_lines = list(set(horizontal_lines))
    vertical_lines = list(set(vertical_lines))

    return horizontal_lines, vertical_lines

# Clusterization of lines
def cluster_lines(lines, size):
    print("Entering line")
    print(lines)
    print("Entering dimension")
    print(size)
    if len(lines) <= 1:
        return lines  # If we have less then 2 lines we return the same list

    lines.sort()
    distances = [lines[i+1] - lines[i] for i in range(len(lines)-1)]
    min_distance = min(distances)
    mean_distance = np.mean(distances)
    
    # Statistical border for clasters is 0.9
    if min_distance > 0 and min_distance / mean_distance < 0.9:
        clusters = []
        current_cluster = [lines[0]]
        for i in range(1, len(lines)):
            if lines[i] - lines[i-1] < mean_distance:
                current_cluster.append(lines[i])
            else:
                clusters.append(round(np.mean(current_cluster))) # Averaging the current claster
                current_cluster = [lines[i]]
        if current_cluster:
            clusters.append(round(np.mean(current_cluster))) # Averaging the last claster

        print(clusters) 
    
        # Filtering side lines
        filtered_clusters = []
        for cluster in clusters:
            # The distance to the left, to the right and to the previouse claster
            distance_to_left = cluster - 0 
            distance_to_right = size - cluster
            distance_to_previous = cluster - (clusters[clusters.index(cluster) - 1] if clusters.index(cluster) > 0 else 0) 

            # Minimal distance:
            min_distance = min(distance_to_left, distance_to_right, distance_to_previous)

            # With also 0.9 value
            if min_distance > mean_distance * 0.9:  
                filtered_clusters.append(cluster)
        
        print(filtered_clusters)

        return filtered_clusters

    else:
        return lines  # If we have no clusters we return the original list


# Main function for cut small squares from the grid
def detect_rectangles(image, horizontal_lines, vertical_lines):

    print("Entering lines")
    print(horizontal_lines, vertical_lines)

    rectangles = []

    # If we have not the borders of the image in lines we add them
    if 0 not in horizontal_lines:
        horizontal_lines = [0] + horizontal_lines
    if image.shape[0] not in horizontal_lines:
        horizontal_lines = horizontal_lines + [image.shape[0]]

    if 0 not in vertical_lines:
        vertical_lines = [0] + vertical_lines
    if image.shape[1] not in vertical_lines:
        vertical_lines = vertical_lines + [image.shape[1]]

    print("Result of the image and lines")
    print(image.shape, horizontal_lines, vertical_lines)

    # Not less 3x3!
    if len(vertical_lines) != len(horizontal_lines) or len(vertical_lines) < 4 or len(horizontal_lines) < 4:
               raise KeyError("Can not recognize or wrong grid!")

    # The copy of the image to draw lines
    image_with_lines = image.copy()
    
    # Draw horizontal
    for x in horizontal_lines:
        cv2.line(image_with_lines, (0, x), (image.shape[1], x), (0, 255, 0), 2)
    
    # Draw vertical
    for y in vertical_lines:
        cv2.line(image_with_lines, (y, 0), (y, image.shape[0]), (255, 0, 0), 2)
    
    # Showing lines
    #cv2.imshow('Image with Detected Lines', image_with_lines)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    for i in range(len(vertical_lines) - 1):
        row = []
        for j in range(len(horizontal_lines) - 1):
            top_left_x = vertical_lines[i]
            top_left_y = horizontal_lines[j]
            bottom_right_x = vertical_lines[i + 1]
            bottom_right_y = horizontal_lines[j + 1]
  
            # Cut the small square and update the matrix
            cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            row.append(cropped_image)

            #cv2.imshow(f"Cell {i},{j}", cropped_image)
            #cv2.waitKey(0)

        rectangles.append(row)
        
    return rectangles        


# Main function for grid recognizing
def grid_recognizing(image_path):
    
    # We prepare the image
    img, edges = load_and_preprocess_image(image_path)
    clean_img = img.copy()

    # We find lines with the same color not less the half of the height
    height, width = img.shape[:2]        
    
    print("Original height, width = img.shape[:2]")
    print(height, width)

    sigma =  int(height // 2.5)
    lines = detect_lines(edges, sigma)
    
    if lines is not None and len(lines) > 0:

        # Sorting lines to vertical and horizontal
        filtered_h_lines, filtered_v_lines = sort_lines(lines)
        
        # Build intersection matrix
        intersection_matrix = visualize_intersections_matrix(filtered_h_lines, filtered_v_lines, img.shape)

        # Strighting all lines
        horizontal_lines, vertical_lines = average_lines(intersection_matrix)

        draw_lines(img,filtered_h_lines)
        draw_lines(img,filtered_v_lines)

        # Show it
        #cv2.imshow('Question Mark', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # Cut the main square of the grid
        cropped_img, cropped_img_no_lines, updated_h_lines, updated_v_lines = crop_image_by_lines(img,  clean_img,  horizontal_lines,  vertical_lines)

        print("Main size for picture with lines after cut cropped_img.shape[:2]")
        print(cropped_img.shape[:2])

        print("Main size for clean picture after cut cropped_img_no_lines.shape[:2]")
        print(cropped_img_no_lines.shape[:2])

        height, width, channels = cropped_img.shape

        # Clustering the lines to avoid too close ones
        clustered_h_lines = cluster_lines(updated_h_lines, width)
        clustered_v_lines = cluster_lines(updated_v_lines, height)

        print(cropped_img.shape)
        print(cropped_img_no_lines.shape)

        # Cut small squares and build the matrix
        clean_img, h_l, v_l = cropped_img_no_lines, clustered_h_lines, clustered_v_lines
        pg = detect_rectangles(clean_img, h_l, v_l)

        return pg

    else:
        print("No lines found.")

