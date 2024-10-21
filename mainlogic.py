import cv2
import numpy as np
import clustergrid
import curves
from curves import graycheck
import compares
from PIL import Image, ImageDraw, ImageFont

# Comparing images in key and matrix
def compare_matrix(squares_matrix, key_matrix, qm):
    # Getting the matrix size
    rows = len(squares_matrix)
    cols = len(squares_matrix[0])
    
    # Build the empty result matrix
    result_matrix = [[None for _ in range(cols)] for _ in range(rows)]
    
    # Loop for keys
    for idx, key in enumerate(key_matrix):
        
        # Loop for matrix
        for i in range(rows):
            for j in range(cols):
                
                # Skip the recognized cell
                if result_matrix[i][j] is not None:
                    continue

                # Skip empty pictures
                if not np.any(squares_matrix[i][j]):
                    continue

                # Comparing with the key
                is_similar = are_images_similar(squares_matrix[i][j], key)
                
                if is_similar:
                    # Write the key index
                    result_matrix[i][j] = idx
                else:
                    # Check the question mark if the picture but uncnown
                    is_question = are_images_similar(squares_matrix[i][j], qm)
                    if is_question:
                        # We know its not a known shape so empty
                        result_matrix[i][j] = None
                    else:
                        # If we found nothing
                        result_matrix[i][j] = None
    return result_matrix

# The main comparing function
def are_images_similar(image1, image2):
    # Bunch of methods

    # Form-factor
    form_factor1 = curves.get_ff(image1)
    form_factor2 = curves.get_ff(image2)
    FF_point = compares.check_ff_difference(form_factor1, form_factor2)

    # Dynamics of pixels
    wpixels_dinamics1 = curves.width_pixel_dynamics(image1)
    wpixels_dinamics2 = curves.width_pixel_dynamics(image2)
    hpixels_dinamics1 = curves.height_pixel_dynamics(image1)
    hpixels_dinamics2 = curves.height_pixel_dynamics(image2)
    pixels_dinamics_sum1 = wpixels_dinamics1 + hpixels_dinamics1
    pixels_dinamics_sum2 = wpixels_dinamics2 + hpixels_dinamics2
    
    # Dynamics of pixels: quick square checking 
    squares = compares.squares_check(pixels_dinamics_sum1, pixels_dinamics_sum2)
    
    # Dynamics of pixels: Pearson
    pearson_pixels_coefficient = compares.pearson_similarity(pixels_dinamics_sum1, pixels_dinamics_sum2)

    # Dynamics of pixels: Cyclic correlation
    pyxel_cyclic_corr= compares.curve_similarity(pixels_dinamics_sum1, pixels_dinamics_sum2)

    # Curves
    img1 = graycheck(image1)
    img2 = graycheck(image2)

    curve1 = curves.get_normalized_curve(img1)
    curve2 = curves.get_normalized_curve(img2)
    
    # Curves: Cyclic correlation
    cyclic_corr = compares.curve_similarity(curve1,curve2)


    print("Start +++++")

    print("FF_point")
    print(FF_point)
    print("pearson_pixels_coefficient")
    print(pearson_pixels_coefficient)

    print("pyxel_cyclic_corr")
    print(pyxel_cyclic_corr)
    print("curve coefficient")
    print(cyclic_corr)
    
    print("Finish +++++")

    # The logic of similarity
    conditions = [
        pearson_pixels_coefficient > 0.8,
        pyxel_cyclic_corr > 0.8,
        cyclic_corr > 0.8,
        FF_point
    ]

    if sum(conditions) >= 3:
        return True
    
    if squares:
        print("Maybe there are two squares")
        return True

    return False # No similarity

# Checking the contour close to the center of the image
# (We need to  delete wrong contours which too close to borders)
def is_contour_outside_radius(contour, image_shape, percentage=25):

    # Empty mask
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    # Center and radius of  the circle on the mask
    image_center_x = image_shape[1] // 2
    image_center_y = image_shape[0] // 2
    radius = int(0.01 * percentage * min(image_shape[:2]))

    cv2.circle(mask, (image_center_x, image_center_y), radius, 1, -1)

    # Another mask for our contour
    contour_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 1, -1)

    # Intersection of masks
    intersection = cv2.bitwise_and(mask, contour_mask)
    intersection_area = cv2.countNonZero(intersection)

    return intersection_area == 0

# The main function for contours recognizing
def process_contours(image):

    gray = graycheck(image)
    
    edges = cv2.Canny(gray, 50, 200)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    height, width = image.shape[:2]
    border_threshold = 0.05

    canvas = np.zeros_like(image)
    lowest_level_contours = []

    if contours:
        for i, h in enumerate(hierarchy[0]):
            contour = contours[i]
            if not is_contour_outside_radius(contour, image.shape):
                leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
                rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
                topmost = tuple(contour[contour[:, :, 1].argmin()][0])
                bottommost = tuple(contour[contour[:, :, 1].argmax()][0])

                if (
                    leftmost[0] > border_threshold * width and
                    rightmost[0] < (1 - border_threshold) * width and
                    topmost[1] > border_threshold * height and
                    bottommost[1] < (1 - border_threshold) * height
                ):
                    lowest_level_contours.append(contour)
                else:
                    print(f"Contour {i} excluded due to border proximity")
            else:
                print(f"Contour {i} excluded due to distance from center")

        cv2.drawContours(canvas, lowest_level_contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    return canvas

# The question mark contour to looking for question mark
def create_question_mark_contour(height, width):

    # Font
    font_size = int(0.9 * height)

    # Empty image
    image = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", font_size)
    
    # Size of symbol
    text = "?"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Centering
    x = (width - text_width) // 2 - bbox[0]
    y = (height - text_height) // 2 - bbox[1]

    # Checking
    #print(f"Image size: {width}x{height}")
    #print(f"Text size: {text_width}x{text_height}")
    #print(f"Text position: ({x}, {y})")

    # Drawing and turning to np
    draw.text((x, y), text, fill="white", font=font)

    image_np = np.array(image)
    image_np = image_np.astype(np.uint8)
      
    #cv2.imshow('Question Mark', image_np)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return image_np

# Building the standartized square matrix of all contours from square matrix of small square pictures
def build_square_matrix(rectangles):
    # Looking the max side 
    max_side = max(max(image.shape[0], image.shape[1]) for row in rectangles for image in row) + 1

    # New contours matrix
    square_matrix = []

    for row in rectangles:
        square_row = []
        for image in row:
            # The main canvas - the biggest canvas, all squares will fit
            square_canvas = np.zeros((max_side, max_side, 3), dtype=np.uint8)

            # Original size
            height, width = image.shape[:2]

            # Centering
            top = (max_side - height) // 2
            left = (max_side - width) // 2
            square_canvas[top:top + height, left:left + width] = image

            # Looking for contours
            processed_canvas = process_contours(square_canvas)

            # Add to the new contours row in the matrix
            square_row.append(processed_canvas)

        # Add the new row to the matrix
        square_matrix.append(square_row)

    # Build the question mark of the same size
    question_mark = create_question_mark_contour(max_side, max_side)

    return square_matrix, question_mark, max_side

# The main function for recognizing matrix according the keys indexes
def main_grid(img, keylist):
    # Looking for a grid
    pg = clustergrid.grid_recognizing(img)
    # Turning into standatrtized contours
    resized_matrix, question_mark, side_size = build_square_matrix(pg)
    # Build the comparing matrix
    compared_matrix = compare_matrix(resized_matrix, keylist, question_mark)
    print(compared_matrix)
    # Transponed the matrix to make it similar with the original picture
    transposed_matrix = list(map(list, zip(*compared_matrix)))
    return transposed_matrix, side_size