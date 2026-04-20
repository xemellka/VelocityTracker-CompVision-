import sim
import time
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

sim.simxFinish(-1)
client_id = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
if client_id == -1:
    raise Exception("No client connected")
print("Connected")

_, cam_handle = sim.simxGetObjectHandle(client_id, 'visionSensor', sim.simx_opmode_blocking)
_, resolution, image = sim.simxGetVisionSensorImage(client_id, cam_handle, 0, sim.simx_opmode_blocking)

img1 = np.array(image, dtype=np.int16)
img1 = (img1 + 128).astype(np.uint8)
img1 = img1.reshape([resolution[1], resolution[0], 3])
img1 = cv2.flip(img1, 0)
img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
print("First photo captured")

time_delay = 2.0
time.sleep(time_delay)

_, resolution, image = sim.simxGetVisionSensorImage(client_id, cam_handle, 0, sim.simx_opmode_blocking)

img2 = np.array(image, dtype=np.int16)
img2 = (img2 + 128).astype(np.uint8)
img2 = img2.reshape([resolution[1], resolution[0], 3])
img2 = cv2.flip(img2, 0)
img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
print("Second photo captured")

def detect_robot_and_draw(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_color = np.array([0, 0, 180])
    upper_color = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        moments = cv2.moments(largest_contour)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

            return (cx, cy)
    return None

pos1 = detect_robot_and_draw(img1)
pos2 = detect_robot_and_draw(img2)

pts_src = np.array([
    [471, 111],
    [43, 105],
    [66, 363],
    [473, 362]
], dtype='float32')

pts_dst = np.array([
    [-1.95, 1.4],
    [-2.075, -2.125],
    [0.925, -1.175],
    [0.925, 0.6]
], dtype='float32')

h_matrix, _ = cv2.findHomography(pts_src, pts_dst)


def get_real_coords(px_point, matrix):
    point = np.array([[[px_point[0], px_point[1]]]], dtype='float32')
    transformed = cv2.perspectiveTransform(point, matrix)
    return transformed[0][0]

if pos1 and pos2:
    real_pos1 = get_real_coords(pos1, h_matrix)
    real_pos2 = get_real_coords(pos2, h_matrix)

    dist_m = math.hypot(real_pos2[0] - real_pos1[0], real_pos2[1] - real_pos1[1])

    time_delay = 2.0
    velocity = dist_m / time_delay

    print("-" * 30)
    print(f"pos1: X={real_pos1[0]:.3f}, Y={real_pos1[1]:.3f}")
    print(f"pos2: X={real_pos2[0]:.3f}, Y={real_pos2[1]:.3f}")
    print(f"S: {dist_m:.3f} м")
    print(f"Speed: {velocity:.3f} м/с")
    print("-" * 30)

    cv2.putText(img2, f"V = {velocity:.3f} m/s", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
else:
    print("No robot")

cv2.imwrite("frame1_detected.png", img1)
cv2.imwrite("frame2_detected.png", img2)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title("Frame 1")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title("Frame 2")
plt.axis('off')
plt.show()

sim.simxFinish(client_id)