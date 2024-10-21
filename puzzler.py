import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pyautogui
from mainlogic import main_grid
import keyread
import numpy as np
from solvepuzzle import puzzle_solution
import help
import cv2
import shapelist

def create_image_with_text(size, color, text):
    color_rgb = Image.new('RGB', (1, 1), color).convert("RGB").getpixel((0,0))
    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
    image = np.full((size, size, 3), color_bgr, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    font_scale = 1

    # Choose the font size
    while True:
        textsize = cv2.getTextSize(text, font, font_scale, thickness)[0]
        if textsize[0] < size * 0.9 and textsize[1] < size * 0.9:
            break
        font_scale -= 0.1
        if font_scale <= 0:
            font_scale = 0.1  
            break #  Break if text too big

    textX = (size - textsize[0]) // 2
    textY = (size + textsize[1]) // 2
    cv2.putText(image, text, (textX, textY), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return ImageTk.PhotoImage(Image.fromarray(image))

def create_main_window():
    def show_help():
        help.show_help1()  

    root = tk.Tk()
    root.title("Puzzle Solver")
    root.geometry("650x750")
    root.resizable(False, False)

    message_label = tk.Label(root, text="", fg="red")
    message_label.grid(row=0, column=0, columnspan=2, pady=(10, 10))

    def update_message(text=""):
        message_label.config(text=text)

    # --- Key Section ---
    load_key_button = tk.Button(root, text="Load Key", command=lambda: cut_image("key"))
    recognize_key_button = tk.Button(root, text="Recognize Key", state=tk.DISABLED, command=lambda: recognize_key())
    confirm_key_button = tk.Button(root, text="Confirm Key", state=tk.DISABLED, command=lambda: confirm_key())  
    key_image_label = tk.Label(root)  # Label for displaying the key image
    key_grid_frame = tk.Frame(root) # Frame for the interactive key grid

    load_key_button.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
    recognize_key_button.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
    key_image_label.grid(row=2, column=0, columnspan=2, pady=(0, 5))
    key_grid_frame.grid(row=3, column=0, columnspan=2, pady=(0, 5))
    confirm_key_button.grid(row=4, column=0, columnspan=2, pady=5, sticky="ew") 

    # --- Grid Section ---
    load_grid_button = tk.Button(root, text="Load Grid", state=tk.DISABLED, command=lambda: cut_image("grid"))
    recognize_grid_button = tk.Button(root, text="Recognize Grid", state=tk.DISABLED,
                                        command=lambda: recognize_grid()) 
    grid_image_label = tk.Label(root)
    grid_display_frame = tk.Frame(root)

    load_grid_button.grid(row=5, column=0, padx=10, pady=5, sticky="ew")
    recognize_grid_button.grid(row=5, column=1, padx=10, pady=5, sticky="ew")
    grid_image_label.grid(row=6, column=0, columnspan=2, pady=(0, 5))
    grid_display_frame.grid(row=7, column=0, columnspan=2, pady=(0, 5))

    # --- Action Buttons ---
    solve_button = tk.Button(root, text="Solve Puzzle", state=tk.DISABLED, command=lambda: solve_puzzle())
    new_puzzle_button = tk.Button(root, text="New Puzzle", command=lambda: new_puzzle())
    help_button = tk.Button(root, text="Help", command=show_help)

    solve_button.grid(row=8, column=0, columnspan=2, pady=(10, 5), sticky="ew") 
    new_puzzle_button.grid(row=9, column=0, columnspan=2, pady=5, sticky="ew")
    help_button.grid(row=10, column=0, columnspan=2, pady=5, sticky="ew")
    

    # --- Variables ---
    grid_image = None
    keylist = None
    extracted_keys = None
    transposed_matrix = None
    size_of_square = 50
    key_widgets = []
    grid_widgets = []

    # --- Functions ---
    def cut_image(mode):
        nonlocal grid_image, size_of_square

        screenshot = pyautogui.screenshot()
        screenshot_pil = Image.fromarray(np.array(screenshot))
        root.config(cursor="cross")


        def create_transparent_window(image):
            transparent_window = tk.Toplevel(root)
            transparent_window.focus_set()
            transparent_window.overrideredirect(True)
            transparent_window.attributes("-topmost", True)
            transparent_window.config(cursor="cross")

            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            transparent_window.geometry(f"{screen_width}x{screen_height}+0+0")


            canvas = tk.Canvas(transparent_window, width=image.width, height=image.height)
            canvas.pack(fill="both", expand=True)


            photo = ImageTk.PhotoImage(image)
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.image = photo

            # If it was just a click
            start_x, start_y = 0, 0
            selection_rect = None


            def on_mouse_press(event):
                nonlocal start_x, start_y, selection_rect
                start_x, start_y = event.x, event.y
                selection_rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red', width=2)

            def on_mouse_motion(event):
                nonlocal selection_rect
                if selection_rect:
                    canvas.coords(selection_rect, start_x, start_y, event.x, event.y)

            def on_mouse_release(event):
                nonlocal grid_image, start_x, start_y # Make start_x, start_y


                end_x, end_y = event.x, event.y
                x1, y1 = min(start_x, end_x), min(start_y, end_y)
                x2, y2 = max(start_x, end_x), max(start_y, end_y)


                if x1 == x2 or y1 == y2:
                    transparent_window.destroy()
                    return


                cropped_image = screenshot_pil.crop((x1, y1, x2, y2))
                width, height = cropped_image.size

                if mode == "grid":
                    if max(width, height) / min(width, height) > 1.1:
                        update_message("Cut only square part of puzzle")
                        transparent_window.destroy()
                        return
                elif mode == "key":
                    if width < height or max(width, height) / min(width, height) < 1.5:
                        update_message("Cut only long horizontal rectangle key part of picture")
                        transparent_window.destroy()
                        return

                max_size = 600
                if width > max_size:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                    cropped_image = cropped_image.resize((new_width, new_height))


                cropped_photo = ImageTk.PhotoImage(cropped_image)

                if mode == "grid":
                    grid_image_label.config(image=cropped_photo)
                    grid_image_label.image = cropped_photo
                    grid_image_label.original_image = cropped_image
                    recognize_grid_button.config(state=tk.NORMAL)
                elif mode == "key":
                    key_image_label.config(image=cropped_photo, anchor=tk.NW)
                    key_image_label.image = cropped_photo
                    key_image_label.original_image = cropped_image 
                    recognize_key_button.config(state=tk.NORMAL)


                transparent_window.destroy()


            canvas.bind("<ButtonPress-1>", on_mouse_press)
            canvas.bind("<B1-Motion>", on_mouse_motion)
            canvas.bind("<ButtonRelease-1>", on_mouse_release)
            transparent_window.bind("<Escape>", lambda event: transparent_window.destroy())

            transparent_window.mainloop()
            
        root.config(cursor='')
        create_transparent_window(screenshot_pil)
        


    def recognize_key():
        nonlocal extracted_keys, size_of_square
        if hasattr(key_image_label, 'original_image'):
            key_image = key_image_label.original_image 
            
            # Turn PIL to NumPy
            key_image_np = np.array(key_image)

            if len(key_image_np.shape) == 3:  # if RGB
                key_image_np = cv2.cvtColor(key_image_np, cv2.COLOR_RGB2BGR)  # to BGR

            try:
                extracted_keys, size_of_square = keyread.extract_shapes(key_image_np) # Recognizing keys
            except Exception as e:
                print(f"Error: {e}")
                update_message("You need to choose keys")
                extracted_keys, size_of_square = key_error_handler() 

        else:
            print("Error: original_image not found in key_image_label.")
        
        # Show keys
        display_key_grid(extracted_keys)

        key_image_label.grid_remove()

        # Enable Confirm button
        confirm_key_button.config(state=tk.NORMAL)

    def key_error_handler():
        def on_select(event=None):
            nonlocal chosen_size
            chosen_size = combo.get()
            if chosen_size:
                top.destroy()

        root = tk.Tk()
        root.withdraw()

        top = tk.Toplevel(root)
        top.title("Default grid:")
        
        label = tk.Label(top, text="The key was not recognized, choose how many keys you have")
        label.pack()

        chosen_size = None

        combo = ttk.Combobox(top, values=[3, 4, 5, 6])
        combo.current(0) # 3 is default
        combo.pack()
        combo.bind("<<ComboboxSelected>>", lambda event: combo.focus())

        ok_button = tk.Button(top, text="OK", command=on_select)
        ok_button.pack()

        top.protocol("WM_DELETE_WINDOW", lambda: on_select())  # Default if close
        
        top.update_idletasks()
        width = top.winfo_width()
        height = top.winfo_height()
        x = (top.winfo_screenwidth() // 2) - (width // 2)
        y = (top.winfo_screenheight() // 2) - (height // 2)
        top.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        top.deiconify()

        root.wait_window(top)

        if chosen_size is None:
            chosen_size = 3

        chosen_size = int(chosen_size)
        size_of_square = 50

        empty_shapes = []
        for i in range(chosen_size):
            empty_shapes.append(np.zeros((size_of_square, size_of_square)))

        return empty_shapes, size_of_square

    def display_key_grid(keys):
        nonlocal key_widgets
        for widget in key_widgets:
            widget.destroy()
        key_widgets.clear()

        selected_key_index = None

        def on_key_click(index):
            nonlocal selected_key_index
            selected_key_index = index

            def handle_choice(choice_index):
                nonlocal keys, selected_key_index
                if selected_key_index is not None:
                    if choice_index == len(shapes):  # Plus
                        # Add empty
                        keys.insert(selected_key_index + 1, np.zeros_like(shapes[0]))
                    elif choice_index == len(shapes) + 1:  # Minus
                        if len(keys) > 1:
                            del keys[selected_key_index]
                    elif choice_index is not None:  # Replace
                        # Change the key
                        keys[selected_key_index] = shapes[choice_index].copy()

                    # Обновляем отображение списка
                    display_key_grid(keys)

                if hasattr(choice_window, "destroy"): 
                    choice_window.destroy()
                selected_key_index = None

            shapes = shapelist.list_of_shapes(size_of_square)
            choice_window = tk.Toplevel(root)
            choice_window.focus_set()
            choice_window.overrideredirect(True)  # Убираем рамки окна
            choice_window.attributes('-topmost', True)
            
            x, y = root.winfo_pointerxy()
            choice_window.geometry(f"+{x}+{y}") 

            rows = 3
            cols = (len(shapes) + 2 + rows - 1) // rows

            for i, shape in enumerate(shapes):
                shape_photo = ImageTk.PhotoImage(Image.fromarray(shape))  # NumPy to Image for Tkinter
                choice_btn = tk.Button(choice_window, image=shape_photo, command=lambda idx=i: handle_choice(idx))
                choice_btn.image = shape_photo  # Keep the picture
                choice_btn.grid(row=i // cols, column=i % cols, padx=2, pady=2)

            # ADD
            plus_photo = create_image_with_text(50, 'lightgreen', "Add new")
            plus_button = tk.Button(choice_window, image=plus_photo, command=lambda: handle_choice(len(shapes)))
            plus_button.image = plus_photo
            plus_button.grid(row=len(shapes) // cols, column=len(shapes) % cols, padx=2, pady=2)

            # DELETE
            minus_photo = create_image_with_text(50, 'lightcoral', "Delete")
            minus_button = tk.Button(choice_window, image=minus_photo, command=lambda: handle_choice(len(shapes) + 1))
            minus_button.image = minus_photo
            minus_button.grid(row=(len(shapes) + 1) // cols, column=(len(shapes) + 1) % cols, padx=2, pady=2)

            # Close choose window
            choice_window.bind("<Key>", lambda event: handle_choice(None))
            choice_window.bind("<Button-1>", lambda e: choice_window.focus_set())
            root.bind("<Button-1>", lambda e: (handle_choice(None) if hasattr(choice_window, 'destroy') else None))

        for i, key_image in enumerate(keys):
            frame = tk.Frame(key_grid_frame, bg="white", relief="solid", highlightbackground="white", highlightthickness=2)
            frame.grid(row=0, column=i, padx=2, pady=2)
            key_widgets.append(frame)

            def update_border(event, frame=frame):
                frame.config(relief="sunken", bg="black", highlightbackground="black", highlightthickness=2)

            def reset_border(event, frame=frame):
                frame.config(relief="solid", bg="white", highlightbackground="white", highlightthickness=2)

            frame.bind("<Enter>", update_border)
            frame.bind("<Leave>", reset_border)

            key_photo = ImageTk.PhotoImage(Image.fromarray(key_image)) 
            key_label = tk.Label(frame, image=key_photo)
            key_label.image = key_photo
            key_label.bind("<Button-1>", lambda event, idx=i: on_key_click(idx))
            key_label.pack()


    def confirm_key():
        nonlocal keylist, extracted_keys
        for key_image in extracted_keys:
            if np.all(key_image == 0):  # Check if key image is empty
                update_message("You should not have empty key")
                return 

        keylist = extracted_keys[:]  # Save the keys

        # Enable Load Grid button, disable other
        load_key_button.config(state=tk.DISABLED)
        load_grid_button.config(state=tk.NORMAL)
        recognize_key_button.config(state=tk.DISABLED)
        confirm_key_button.config(state=tk.DISABLED)
        update_message("Key confirmed. Load grid now.")

        # Disable key interaction and frame changes
        disable_key_interaction()


    def disable_key_interaction():
        for widget in key_widgets:
            widget.winfo_children()[0].unbind("<Button-1>") # Disable clicks on key images

    def recognize_grid():

        nonlocal transposed_matrix

        if keylist is None:
            update_message("Confirm key first!")
            return
        
        grid_image_np = np.array(grid_image_label.original_image)

        try:
            transposed_matrix, _ = main_grid(grid_image_np, keylist)
        except Exception as e:
            print(f"Error: {e}")
            update_message("Cannot recognize, you need to choose grid size")
            default_grid_size = len(keylist)  # Use the number of keys as default
            transposed_matrix = grid_error_handler(default_grid_size) # Pass the default size

        display_grid(transposed_matrix, keylist) # Update grid display
        grid_image_label.grid_remove() # Remove the grid image label
        solve_button.config(state=tk.NORMAL) # Enable solve button
        recognize_grid_button.config(state=tk.DISABLED) # Disable recognize button
        load_grid_button.config(state=tk.DISABLED) # Disable load button

    def grid_error_handler(default_size):
        def on_select(event=None):
            nonlocal chosen_size
            chosen_size = combo.get()
            if chosen_size:
                top.destroy()

        root = tk.Tk()
        root.withdraw()

        top = tk.Toplevel(root)
        top.title("Default grid:")

        label = tk.Label(top, text="The grid was not recognized, choose the size of grid")
        label = tk.Label(top, text="(Nubers of keys by default)") 
        label.pack()

        chosen_size = None

        display_values = [f"{n} x {n}" for n in [3, 4, 5, 6]]
        combo = ttk.Combobox(top, values=display_values)

        try:
            default_index = [3, 4, 5, 6].index(default_size)
            combo.current(default_index)
        except ValueError:
            combo.current(0)

        combo.pack()
        combo.bind("<<ComboboxSelected>>", on_select)

        ok_button = tk.Button(top, text="OK", command=on_select)
        ok_button.pack()

        top.protocol("WM_DELETE_WINDOW", lambda: on_select())
        
        # Центрируем окно
        top.update_idletasks()
        width = top.winfo_width()
        height = top.winfo_height()
        x = (top.winfo_screenwidth() // 2) - (width // 2)
        y = (top.winfo_screenheight() // 2) - (height // 2)
        top.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        top.deiconify()

        root.wait_window(top)

        if chosen_size is None:
            chosen_size = default_size
        else:
            chosen_size = int(chosen_size.split(" ")[0])

        transposed_matrix = [[None for _ in range(chosen_size)] for _ in range(chosen_size)]

        return transposed_matrix


    def display_grid(matrix, keylist):
        nonlocal grid_widgets
        selected_cell = None

        # Clear old widgets
        for widget in grid_widgets:
            widget.destroy()
        grid_widgets.clear()

        # Add clear option to allowed shapes
        allowed_shapes = keylist[:] + [np.zeros_like(keylist[0])]  

        def on_cell_click(row, col):
            nonlocal selected_cell, matrix

            selected_cell = (row, col)

            def handle_choice(choice_index):
                nonlocal matrix, selected_cell
                if selected_cell:
                    if choice_index is not None:
                        if choice_index < len(allowed_shapes) - 1:
                            matrix[selected_cell[0]][selected_cell[1]] = choice_index
                        elif choice_index == len(allowed_shapes) - 1:  # Clear
                            matrix[selected_cell[0]][selected_cell[1]] = None
                        elif choice_index == len(allowed_shapes): #del line
                            del matrix[selected_cell[0]]
                            display_grid(matrix,keylist)
                        elif choice_index == len(allowed_shapes) + 1: #del col
                            for row in matrix:
                                del row[selected_cell[1]]
                            display_grid(matrix,keylist)
                        elif choice_index == len(allowed_shapes) + 2: #add line
                            matrix.insert(selected_cell[0] + 1, [None for _ in range(len(matrix[0]))])
                            display_grid(matrix,keylist)
                        elif choice_index == len(allowed_shapes) + 3: #add col
                            for row in matrix:
                                row.insert(selected_cell[1] + 1, None)
                            display_grid(matrix,keylist)
                        else:  # Clear option selected
                            matrix[selected_cell[0]][selected_cell[1]] = None
                        selected_cell=None
                        display_grid(matrix, keylist)

                # Close the choice window
                if hasattr(choice_window, "destroy"):
                    choice_window.destroy()

                root.unbind("<Button-1>")
                selected_cell = None

            # Display the choice window
            choice_window = tk.Toplevel(root)
            choice_window.focus_set()
            choice_window.overrideredirect(True)
            choice_window.attributes('-topmost', True)

            x, y = root.winfo_pointerxy()
            choice_window.geometry(f"+{x}+{y}") 

            # Frame for shape buttons
            shapes_frame = tk.Frame(choice_window)
            shapes_frame.grid(row=0, column=0, sticky="nsew") # or pack(side=tk.LEFT)

            # Frame for control buttons
            control_frame = tk.Frame(choice_window)
            control_frame.grid(row=1, column=0, sticky="nsew") # or pack(side=tk.RIGHT)

            num_control_buttons = 5  # Clear, Del Row, Del Col, Add Row, Add Col
            rows_choice = 3
            cols_choice = (len(allowed_shapes) - 1 + num_control_buttons + rows_choice - 1) // rows_choice
 
            # Create buttons for each shape in allowed_shapes
            for i, shape in enumerate(allowed_shapes[:-1]):
                shape_photo = ImageTk.PhotoImage(Image.fromarray(shape))
                choice_btn = tk.Button(choice_window, image=shape_photo, command=lambda idx=i: handle_choice(idx))
                choice_btn.image = shape_photo
                choice_btn.grid(row=i // cols_choice, column=i % cols_choice, padx=2, pady=2, in_=shapes_frame)

            # Create Clear, Add/Remove Row/Col buttons in the control_frame
            clear_photo = create_image_with_text(50, 'lightgray', "Clear")

            clear_button = tk.Button(control_frame, image=clear_photo,
                                    command=lambda: handle_choice(len(allowed_shapes) - 1))
            clear_button.image = clear_photo
            clear_button.grid(row=0, column=0, padx=2, pady=2)

            del_line_photo = create_image_with_text(50, 'lightcoral', "Del Row")
            del_line = tk.Button(control_frame, image=del_line_photo, command=lambda: handle_choice(len(allowed_shapes)))
            del_line.image = del_line_photo
            del_line.grid(row=0, column=1, padx=2, pady=2)

            del_col_photo = create_image_with_text(50, 'lightcoral', "Del Col")
            del_col = tk.Button(control_frame, image=del_col_photo, command=lambda: handle_choice(len(allowed_shapes)+1))
            del_col.image = del_col_photo
            del_col.grid(row=1, column=0, padx=2, pady=2)

            add_line_photo = create_image_with_text(50, 'lightgreen', "Add Row")
            add_line = tk.Button(control_frame, image=add_line_photo, command=lambda: handle_choice(len(allowed_shapes)+2))
            add_line.image = add_line_photo
            add_line.grid(row=1, column=1, padx=2, pady=2)

            add_col_photo = create_image_with_text(50, 'lightgreen', "Add Col")
            add_col = tk.Button(control_frame, image=add_col_photo, command=lambda: handle_choice(len(allowed_shapes)+3))
            add_col.image = add_col_photo
            add_col.grid(row=2, column=0, padx=2, pady=2)

            # Bind events for closing the choice window
            choice_window.bind("<Key>", lambda event: handle_choice(None))
            root.bind("<Button-1>", lambda e: (handle_choice(None) if hasattr(choice_window, 'destroy') else None))

            choice_window.bind("<Button-1>", lambda e: choice_window.focus_set())
            root.bind("<Button-1>", lambda e: (handle_choice(None) if hasattr(choice_window, 'destroy') else None))

        # Display the grid

        rows = len(matrix)
        cols = len(matrix[0])
        for i in range(rows):
            for j in range(cols):
                frame = tk.Frame(grid_display_frame, bg="white", relief="solid", highlightbackground="white", highlightthickness=2)
                frame.grid(row=i, column=j, padx=2, pady=2)
                grid_widgets.append(frame)
                value = matrix[i][j]

                def update_border(event, val=value, frame=frame):
                    update_message(f"Key index: {val}")
                    frame.config(relief="sunken", bg="black", highlightbackground="black", highlightthickness=2)

                def reset_border(event, frame=frame):
                    frame.config(relief="sunken", bg="white", highlightbackground="white", highlightthickness=2)

                # Send frame and value for message
                frame.bind("<Enter>", lambda event, val=value, fr=frame: update_border(event, val, fr))
                frame.bind("<Leave>", lambda event, fr=frame: reset_border(event, fr))


                if value is not None and 0 <= value < len(keylist):
                    img_pil = Image.fromarray(cv2.cvtColor(keylist[value], cv2.COLOR_BGR2RGBA))
                    img_tk = ImageTk.PhotoImage(img_pil)
                    image_label = tk.Label(frame, image=img_tk)
                    image_label.image = img_tk
                    image_label.bind("<Button-1>", lambda event, r=i, c=j: on_cell_click(r, c))
                    image_label.pack()
                else:  # Display empty image for None values
                    empty_image = ImageTk.PhotoImage(Image.fromarray(np.zeros_like(keylist[0]))) # Create empty image
                    empty_label = tk.Label(frame, image=empty_image)  # Use image, not text
                    empty_label.image = empty_image # Keep a reference
                    empty_label.bind("<Button-1>", lambda event, r=i, c=j: on_cell_click(r, c))
                    empty_label.pack()


    def solve_puzzle():
        nonlocal transposed_matrix, keylist
        print("transposed_matrix")
        print(transposed_matrix)

        if transposed_matrix is None or keylist is None:
            update_message("Recognize grid and confirm key first!")
            return

        key_indices = list(range(len(keylist)))
        new_matrix = puzzle_solution(transposed_matrix, key_indices)

        if new_matrix:
            display_grid(new_matrix, keylist)
            update_message("Puzzle solved!")


    def new_puzzle():
        nonlocal grid_image, key_grid_frame, grid_display_frame, key_image_label, grid_image_label, keylist, extracted_keys, transposed_matrix, key_widgets, grid_widgets
        # Reset variables and clear displays
        grid_image = None
        keylist = None
        extracted_keys = None
        transposed_matrix = None

        # Clear key and grid images from labels before destroying the widgets!
        key_image_label.config(image="")
        key_image_label.image = None  # Release the image reference
        grid_image_label.config(image="")
        grid_image_label.image = None  # Release the image reference
        
        # Clear display widgets using winfo_children() to get ALL children
        for widget in key_grid_frame.winfo_children():
            widget.destroy()

        for widget in grid_display_frame.winfo_children():
            widget.destroy()

        for widget in key_widgets:
            widget.destroy()
        key_widgets.clear()      

        for widget in grid_widgets:
            widget.destroy()
        grid_widgets.clear()
  
        # Clear and recreate frames
        key_grid_frame.destroy()
        key_grid_frame = tk.Frame(root)
        key_grid_frame.grid(row=2, column=0, columnspan=2, pady=(0, 5))  # Place frame for key grid

        grid_display_frame.destroy() 
        grid_display_frame = tk.Frame(root) 
        grid_display_frame.grid(row=7, column=0, columnspan=2, pady=(0, 5)) 

        # Reset button states
        load_key_button.config(state=tk.NORMAL)
        recognize_key_button.config(state=tk.DISABLED)
        confirm_key_button.config(state=tk.DISABLED)
        load_grid_button.config(state=tk.DISABLED)
        recognize_grid_button.config(state=tk.DISABLED)
        solve_button.config(state=tk.DISABLED)

        # Clear message label
        update_message("")

        # Restore key_image_label and grid_image_label (if they were removed)
        key_image_label.grid()
        grid_image_label.grid()

        root.update_idletasks()


    # Grid layout configuration (same as before)

    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)

    root.mainloop()


create_main_window()