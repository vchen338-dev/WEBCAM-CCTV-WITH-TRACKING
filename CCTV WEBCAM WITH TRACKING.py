import cv2
import tkinter as tk
from tkinter import simpledialog
from datetime import datetime
import threading
import numpy as np

# Global variables
camera_feeds = []
feed_threads = []
running = True
motion_notifications = []  # List to store motion detection notifications

def add_camera():
    """Add a new camera."""
    global camera_feeds
    camera_id = simpledialog.askinteger("Input", "Enter Camera ID (0 for default webcam):")
    if camera_id is not None:
        camera_feeds.append(camera_id)
        update_feed_status()

def update_feed_status():
    """Update the camera feed status in the GUI."""
    feed_status.config(text=f"Active Cameras: {len(camera_feeds)}")

def capture_feed(camera_id, output_dict, index):
    """Capture feed from a camera, perform motion detection, and store it in the output dictionary."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}.")
        return

    prev_frame = None
    persistent_boxes = []  # Track detection frames over time

    while running:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Camera {camera_id} stopped working.")
            break

        # Resize frame
        frame = cv2.resize(frame, (640, 480))

        # Convert to grayscale and blur for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Motion detection
        if prev_frame is None:
            prev_frame = gray
            continue

        # Frame differencing
        delta_frame = cv2.absdiff(prev_frame, gray)
        thresh = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Contour detection
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        new_boxes = []

        motion_detected = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Ignore small movements
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            new_boxes.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
            motion_detected = True

        # Persist detections for smoother tracking
        persistent_boxes = [
            box for box in persistent_boxes if any(
                abs(box[0] - new_box[0]) < 50 and abs(box[1] - new_box[1]) < 50 for new_box in new_boxes
            )
        ]
        persistent_boxes.extend(new_boxes)

        for (x, y, w, h) in persistent_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Update the output dictionary with the frame
        output_dict[index] = frame

        # Add motion detection notification
        if motion_detected:
            motion_notifications.append(f"Camera {camera_id} motion detected at {timestamp}")
            if len(motion_notifications) > 10:  # Limit notifications to the last 10
                motion_notifications.pop(0)

        # Update previous frame
        prev_frame = gray

    cap.release()

def start_feeds():
    """Start capturing feeds from all cameras."""
    global feed_threads, running
    running = True
    output_dict = {}

    # Start a thread for each camera
    for i, camera_id in enumerate(camera_feeds):
        output_dict[i] = np.zeros((480, 640, 3), dtype=np.uint8)  # Placeholder frame
        thread = threading.Thread(target=capture_feed, args=(camera_id, output_dict, i), daemon=True)
        feed_threads.append(thread)
        thread.start()

    # Display feeds in a single window
    display_feeds(output_dict)

def display_feeds(output_dict):
    """Display all camera feeds in a single fullscreen window."""
    global running

    cv2.namedWindow("CCTV System", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("CCTV System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while running:
        # Arrange feeds in a grid
        feeds = list(output_dict.values())
        if feeds:
            # Calculate grid size
            num_feeds = len(feeds)
            grid_cols = int(np.ceil(np.sqrt(num_feeds)))
            grid_rows = int(np.ceil(num_feeds / grid_cols))

            # Resize frames and arrange in grid
            height, width = feeds[0].shape[:2]
            grid_frame = np.zeros((grid_rows * height, grid_cols * width + 300, 3), dtype=np.uint8)

            for idx, frame in enumerate(feeds):
                row, col = divmod(idx, grid_cols)
                y_start, y_end = row * height, (row + 1) * height
                x_start, x_end = col * width, (col + 1) * width
                grid_frame[y_start:y_end, x_start:x_end] = cv2.resize(frame, (width, height))

            # Add notification panel on the right
            cv2.rectangle(grid_frame, (grid_cols * width, 0), (grid_cols * width + 300, grid_rows * height), (50, 50, 50), -1)
            y_offset = 30
            for notification in motion_notifications:
                cv2.putText(grid_frame, notification, (grid_cols * width + 10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

            # Show the grid
            cv2.imshow("CCTV System", grid_frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cv2.destroyAllWindows()

# GUI
root = tk.Tk()
root.title("CCTV System")

# Buttons and Labels
tk.Label(root, text="CCTV Camera System").pack(pady=10)
tk.Button(root, text="Add Camera", command=add_camera).pack(pady=5)
tk.Button(root, text="Start Feeds", command=start_feeds).pack(pady=5)
feed_status = tk.Label(root, text="Active Cameras: 0")
feed_status.pack(pady=5)

# Start the GUI
root.mainloop()

# Stop threads when the GUI is closed
running = False
