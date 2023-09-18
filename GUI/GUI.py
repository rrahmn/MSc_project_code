import tkinter as tk
from tkinter import filedialog
import cv2 as cv
import numpy as np
from PIL import Image, ImageTk
import os
import shutil
import datetime
import glob

class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Selection")

        self.image_paths = []
        self.selected_image = None

        self.create_widgets()

    def create_widgets(self):

        #creating directory to hold images used in calibration
        self.final_images_dir = os.getcwd() + '\Calibration_Images'
        if os.path.exists(self.final_images_dir):
            shutil.rmtree(self.final_images_dir)
        os.makedirs(self.final_images_dir)

        # Frame to hold buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        # Button to upload images
        upload_button = tk.Button(button_frame, text="Upload Images", command=self.upload_images)
        upload_button.pack(side="left", padx=10)

        # Button to open webcam
        webcam_button = tk.Button(button_frame, text="Open Webcam", command=self.open_webcam)
        webcam_button.pack(side="left", padx=10)


        # Create the entry fields for the numbers
        # Create the labels for the entry fields
        label1 = tk.Label(button_frame, text="Chessboard width:")
        label1.pack(side=tk.LEFT, padx=5, pady=5)
        self.chessboard_width_field = tk.Entry(button_frame)
        self.chessboard_width_field.pack(side=tk.LEFT, padx=5, pady=5)


        label2 = tk.Label(button_frame, text="Chessboard height:")
        label2.pack(side=tk.LEFT, padx=5, pady=5)
        self.chessboard_height_field = tk.Entry(button_frame)
        self.chessboard_height_field.pack(side=tk.LEFT, padx=5, pady=5)

        # Button to go to calibration screen
        calibrate_screen = tk.Button(button_frame, text="Calibration window", command=self.calibration_screen)
        calibrate_screen.pack(side="left", padx=10)

        # Frame to hold scrollable image viewer
        image_frame = tk.Frame(self.root, width=100, height=30)
        image_frame.pack(pady=10, fill=None, expand=False)
        # Scrollbar for image frame
        scrollbar = tk.Scrollbar(image_frame)
        scrollbar.pack(side="right", fill="y")

        # Listbox to display images
        self.image_listbox = tk.Listbox(
            image_frame, selectmode="extended", yscrollcommand=scrollbar.set, width=70, height=30
        )
        self.image_listbox.pack(side="left", fill=tk.BOTH, expand=True)

        scrollbar.config(command=self.image_listbox.yview)

        # Bind selection event
        self.image_listbox.bind("<<ListboxSelect>>", self.select_image)



        # Canvas to display selected image
        self.image_canvas = tk.Canvas(image_frame, width=400, height=400)
        self.image_canvas.pack(side="left")
        

        # Button to delete selected image
        delete_button = tk.Button(self.root, text="Delete Image", command=self.delete_image)
        delete_button.pack(pady=10)

    def upload_images(self):
        filetypes = (("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
        paths = filedialog.askopenfilenames(title="Select Images", filetypes=filetypes)

        for path in paths:
            


            # Construct the full paths of the source and destination images
            source_path = path
            destination_path = self.final_images_dir + '\\' + os.path.basename(path)
             # Use shutil to copy the image from the source to destination directory
            shutil.copy(source_path, destination_path)
            self.image_paths.append(destination_path)
            self.image_listbox.insert(tk.END, destination_path)
        

    def open_webcam(self):
        #flag to check whether we have an image snapped
        self.holding_snapped_image=False
        self.snapped_image_unsaved = True

        self.webcam_window = tk.Toplevel(self.root)
        self.webcam_window.title("Webcam Opened")
        self.webcam_window.geometry("1000x600")


        


        # Frame to hold buttons
        webcam_button_frame = tk.Frame(self.webcam_window)
        webcam_button_frame.pack( pady=10)

        # Button to snap images
        snapshot_button = tk.Button(webcam_button_frame, text="Snap Image", command=self.snap_image)
        snapshot_button.pack(side="left", padx=10)

        # Button to keep image
        keep_button = tk.Button(webcam_button_frame, text="Keep Image", command=self.keep_image)
        keep_button.pack(side="left", padx=10)

        # Button to discard image
        discard_button = tk.Button(webcam_button_frame, text="Discard Image", command=self.discard_image)
        discard_button.pack(side="left", padx=10)


        # label to display webcam frame
        self.webcam_label = tk.Label(self.webcam_window, width=400, height=400)
        self.webcam_label.pack(side="left")

        # label to display snapped frame
        self.snapped_image_label = tk.Label(self.webcam_window, width=400, height=400)
        self.snapped_image_label.pack(side="right")

        # Initialize video capture
        self.cap = cv.VideoCapture(0)

        if self.cap.isOpened():
            self.show_frame()

    def show_frame(self):

        if not self.webcam_window.winfo_exists():
            # Window has been closed, do any necessary cleanup here
            self.cap.release()  # Release the video capture
            return


        ret, frame = self.cap.read()

        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame.thumbnail((400, 400))

            
            try:
                self.frame = frame
                self.webcam_label.image = ImageTk.PhotoImage(self.frame)
                self.webcam_label.configure(image=self.webcam_label.image)
            except:
                pass

        self.root.after(1, self.show_frame)

    def snap_image(self):
        
        if self.cap.isOpened():
            ret, frame = self.cap.read()

            if ret:
                self.holding_snapped_image = True
                self.snapped_image_unsaved = True
                snapped_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                snapped_image = Image.fromarray(snapped_image)
                snapped_image.thumbnail((400, 400))

                self.snapped_image = snapped_image
                # Generate a timestamp using the current time
                self.timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                self.snapped_image_path = self.final_images_dir + '\\' + f'{self.timestamp}.png'
                self.snapped_image_label.image = ImageTk.PhotoImage(self.snapped_image)
                self.snapped_image_label.configure(image=self.snapped_image_label.image)


    def keep_image(self):
        if(self.holding_snapped_image and self.snapped_image_unsaved):
            self.snapped_image_unsaved = False
            #save snapped image to final calibration directory
            (self.snapped_image).save(self.snapped_image_path)

            #add to list and dropdown
            self.image_paths.append(self.snapped_image_path)
            self.image_listbox.insert(tk.END, self.snapped_image_path)

    def discard_image(self):
        if(self.holding_snapped_image):
            try:
                self.snapped_image_unsaved = True

                #remove from final calibration directory
                os.remove(self.snapped_image_path)
                #empty snapped image and clear label
                self.holding_snapped_image = False
                self.snapped_image = None
                self.snapped_image_label.image = ImageTk.PhotoImage(Image.new("RGB", (1, 1)))  # Create an empty PhotoImage
                self.snapped_image_label.configure(image=self.snapped_image_label.image)

                #remove from list and dropdown
                if self.snapped_image_path in self.image_paths:
                    index = self.image_paths.index(self.snapped_image_path)
                    self.image_listbox.delete(index)
                    image_path = self.image_paths[index]

                self.image_paths.remove(image_path)
            except:
                pass
    def calibration_screen(self):
        self.calibration_window = tk.Toplevel(self.root)
        self.calibration_window.title("Calibration Window")
        self.calibration_window.geometry("1200x800")
        # Get the values from the entry fields
        self.chessboard_width = int(self.chessboard_width_field.get())
        self.chessboard_height = int(self.chessboard_height_field.get())

        
        # Frame to hold scrollable image viewer
        image_frame = tk.Frame(self.calibration_window, width=1000, height=600)
        image_frame.pack(pady=10, fill=None, expand=False)
        # Scrollbar for image frame
        scrollbar = tk.Scrollbar(image_frame)
        scrollbar.pack(side="right", fill="y")

        # Listbox to display images
        self.image_listbox2 = tk.Listbox(
            image_frame, selectmode="extended", yscrollcommand=scrollbar.set, width=30, height=30
        )
        self.image_listbox2.pack(side="left", fill=tk.BOTH, expand=True)

        scrollbar.config(command=self.image_listbox2.yview)

        # Bind selection event
        self.image_listbox2.bind("<<ListboxSelect>>", self.select_image2)



        # Canvas to display selected image
        self.image_canvas2 = tk.Canvas(image_frame, width=1000, height=600)
        self.image_canvas2.pack(fill=None, expand=False)
        

        for path in self.image_paths:
            self.image_listbox2.insert(tk.END, path)

        # Button to delete selected image
        delete_button2 = tk.Button(self.calibration_window, text="Delete Image", command=self.delete_image2)
        delete_button2.pack(pady=10)

        #buttom to start calibration
        calibrate_button = tk.Button(self.calibration_window, text="Calibrate", command=self.calibration_func)
        calibrate_button.pack(side="right", padx=10)

        


        #drawing on top of image
        # termination criteria
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((self.chessboard_width*self.chessboard_height,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.chessboard_width,0:self.chessboard_height].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.

        
        indices_to_remove = []
        index=-1
        for path in self.image_paths:
            index+=1
            img = cv.imread(path)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, _ = cv.findChessboardCorners(gray, (self.chessboard_width,self.chessboard_height), None)
            if(ret == False):
                indices_to_remove.append(index)
        

        for index in reversed(indices_to_remove):
            self.image_listbox.delete(index)
            self.image_listbox2.delete(index)
            image_path = self.image_paths[index]
            # Remove from final calibration directory
            os.remove(image_path)
            self.image_paths.remove(image_path)




    def select_image(self, event):
        selections = self.image_listbox.curselection()

        if selections:
            index = selections[-1]
            image_path = self.image_paths[index]

            self.selected_image = Image.open(image_path)
            self.selected_image.thumbnail((400, 400))

            self.image_canvas.delete("all")
            self.image_canvas.image = ImageTk.PhotoImage(self.selected_image)
            self.image_canvas.create_image(0, 0, anchor="nw", image=self.image_canvas.image)

    def select_image2(self, event):
        selections = self.image_listbox2.curselection()

        if selections:
            index = selections[-1]
            image_path = self.image_paths[index]


            self.selected_image = Image.open(image_path)
            img = cv.imread(image_path)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (self.chessboard_width,self.chessboard_height), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), self.criteria)
                # Draw and display the corners
                cv.drawChessboardCorners(img, (self.chessboard_width,self.chessboard_height), corners2, ret)
                self.selected_image = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
                self.selected_image.thumbnail((1000, 600))

                self.image_canvas2.delete("all")
                self.image_canvas2.image = ImageTk.PhotoImage(self.selected_image)
                self.image_canvas2.create_image(0, 0, anchor="nw", image=self.image_canvas2.image)

            

    def delete_image(self):
        selections = self.image_listbox.curselection()

        for index in reversed(selections):
            self.image_listbox.delete(index)
            image_path = self.image_paths[index]

            #remove from final calibration directory
            os.remove(self.final_images_dir + '\\' + os.path.basename(image_path))

            self.image_paths.remove(image_path)

            if self.selected_image and image_path == self.selected_image.filename:
                self.selected_image = None
                self.image_canvas.delete("all")

    def delete_image2(self):
        selections = self.image_listbox2.curselection()

        for index in reversed(selections):
            self.image_listbox.delete(index)
            self.image_listbox2.delete(index)
            image_path = self.image_paths[index]

            #remove from final calibration directory
            os.remove(self.final_images_dir + '\\' + os.path.basename(image_path))

            self.image_paths.remove(image_path)

            if self.selected_image:
                self.selected_image = None
                self.image_canvas2.delete("all")

    def calibration_func(self):
        for path in self.image_paths:
            img = cv.imread(path)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (self.chessboard_width,self.chessboard_height), None)
            if ret == True:
                self.objpoints.append(self.objp)
                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), self.criteria)
                self.imgpoints.append(corners2)
        
        #calibration
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)

        #button to save calibration results
        save_calibrate_button = tk.Button(self.calibration_window, text="Save calibration ", command=self.save_calib)
        save_calibrate_button.pack(side="right", padx=10)


        mean_error = 0
        for i in range(len(self.objpoints)):
            self.imgpoints2, _ = cv.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
            error = cv.norm(self.imgpoints[i], self.imgpoints2, cv.NORM_L2)/len(self.imgpoints2)
            mean_error += error
        print( "total error: {}".format(mean_error/len(self.objpoints)) )
        print(mean_error)
        print(self.ret)
        print("Intrinsic: ")
        print(self.mtx)
        print("Distortion: ")
        print(self.dist)
        print("Extrinsic: ")
        print(self.rvecs)
        print(self.tvecs)
    
    def save_calib(self):
        # Open a file dialog to select the save location and filename
        filetypes = (("XML files", "*.xml"), ("All files", "*.*"))
        save_path = filedialog.asksaveasfilename(title="Save Calibration", filetypes=filetypes)
        
        # Create a FileStorage object
        fs = cv.FileStorage(save_path, cv.FILE_STORAGE_WRITE)

        # Write camera matrix and distortion coefficients
        fs.write('cameraMatrix', np.array(self.mtx))
        fs.write('distCoeffs', np.array(self.dist))

        # Write rotation and translation vectors
        fs.write('rvecs', np.array(self.rvecs))
        fs.write('tvecs', np.array(self.tvecs))

        # Release the FileStorage object
        fs.release()


#creation
root = tk.Tk()
root.title("Calibration Interface")
root.geometry("1000x600")

app = ImageViewer(root)
root.mainloop()


