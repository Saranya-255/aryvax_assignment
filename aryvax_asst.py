#!/usr/bin/env python
# coding: utf-8

# In[2]:


http://localhost:8888/?token=cfab9918845b8db544e210df2e7e4f2636462c04e07b2dff!pip install opencv-python
get_ipython().system('pip install opencv-contrib-python')


# In[ ]:


import cv2
import numpy as np
import time

FRAME_W = 640
FRAME_H = 480
MIN_AREA = 2000

OBJ_CENTER = (int(FRAME_W * 0.75), int(FRAME_H * 0.45))
OBJ_RADIUS = 70

SAFE_DIST = 160
WARNING_DIST = 60

SMOOTH_ALPHA = 0.4  

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

# ------------------ Helpers ------------------
def nothing(x): pass

def read_hsv():
    hl = cv2.getTrackbarPos("H_low","HSV Tune")
    hh = cv2.getTrackbarPos("H_high","HSV Tune")
    sl = cv2.getTrackbarPos("S_low","HSV Tune")
    sh = cv2.getTrackbarPos("S_high","HSV Tune")
    vl = cv2.getTrackbarPos("V_low","HSV Tune")
    vh = cv2.getTrackbarPos("V_high","HSV Tune")
    return np.array([hl,sl,vl], dtype=np.uint8), np.array([hh,sh,vh], dtype=np.uint8)

def smooth_point(old, new, alpha=SMOOTH_ALPHA):
    if old is None:
        return new
    return (int(alpha*new[0] + (1-alpha)*old[0]), int(alpha*new[1] + (1-alpha)*old[1]))

def fingertip_from_defects(cnt, hull):
    # use convexity defects to detect fingers
    if len(hull) < 3:
        return None
    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return None
    defects = cv2.convexityDefects(cnt, hull_idx)
    if defects is None:
        return None
    pts = []
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        far = tuple(cnt[f][0])
        # filter defects by depth (d) to remove small ones
        if d > 3000:  # tune threshold by experimenting
            pts.append(tuple(cnt[s][0]))
            pts.append(tuple(cnt[e][0]))
    if not pts:
        return None
    # choose the top-most among candidate fingertip points (smallest y in image coords)
    pts = sorted(pts, key=lambda p: p[1])
    return pts[0]  # top-most point

def farthest_point_from_centroid(cnt):
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return None, None
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    centroid = (cx, cy)
    pts = cnt.reshape(-1, 2)
    dists = np.linalg.norm(pts - np.array(centroid), axis=1)
    idx = np.argmax(dists)
    return tuple(pts[idx]), centroid

def classify_status(fingertip, center, radius):
    dcenter = np.linalg.norm(np.array(fingertip) - np.array(center))
    d_to_boundary = max(0.0, dcenter - radius)
    if d_to_boundary <= 0:
        return "DANGER", d_to_boundary
    elif d_to_boundary <= WARNING_DIST:
        return "WARNING", d_to_boundary
    elif d_to_boundary <= SAFE_DIST:
        return "APPROACHING", d_to_boundary
    else:
        return "SAFE", d_to_boundary

# ------------------ Main ------------------

def create_trackbars():
    cv2.namedWindow("HSV Tune", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("H_low","HSV Tune",0,179,nothing)
    cv2.createTrackbar("H_high","HSV Tune",20,179,nothing)
    cv2.createTrackbar("S_low","HSV Tune",48,255,nothing)
    cv2.createTrackbar("S_high","HSV Tune",255,255,nothing)
    cv2.createTrackbar("V_low","HSV Tune",50,255,nothing)
    cv2.createTrackbar("V_high","HSV Tune",255,255,nothing)
    
    
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    create_trackbars()
    smoothed_tip = None
    prev = time.time()
    fps = 0

    flash_state = False
    flash_t = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower, upper = read_hsv()
        mask = cv2.inRange(hsv, lower, upper)
        # morphological cleanup
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
        mask = cv2.GaussianBlur(mask, (7,7), 0)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fingertip = None
        centroid = None
        if cnts:
            maxc = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(maxc)
            if area > MIN_AREA:
                cv2.drawContours(display, [maxc], -1, (0,255,0), 2)
                hull = cv2.convexHull(maxc)
                cv2.drawContours(display, [hull], -1, (0,200,200), 1)
                # try convexity defects method
                f_def = fingertip_from_defects(maxc, hull)
                if f_def is not None:
                    fingertip = f_def
                else:
                    # fallback: farthest point from centroid
                    f_far, centroid = farthest_point_from_centroid(maxc)
                    fingertip = f_far
                if fingertip is not None:
                    smoothed_tip = smooth_point(smoothed_tip, fingertip)
                    cv2.circle(display, smoothed_tip, 8, (0,0,255), -1)
                    if centroid:
                        cv2.circle(display, centroid, 4, (255,0,0), -1)

        # draw virtual object
        cv2.circle(display, OBJ_CENTER, OBJ_RADIUS, (200,200,200), 2)

        if smoothed_tip is not None:
            status, d_b = classify_status(smoothed_tip, OBJ_CENTER, OBJ_RADIUS)
            if status == "SAFE":
                color = (0,200,0)
                label = "SAFE"
            elif status == "APPROACHING" or status == "WARNING":
                color = (0,220,220)
                label = "WARNING"
            else:
                color = (0,0,255)
                label = "DANGER"

            # overlay status box
            cv2.rectangle(display, (5,5), (300,72), (0,0,0), -1)
            cv2.putText(display, f"State: {label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(display, f"Dist: {d_b:.1f}px", (10,58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)

            # danger flashing
            if label == "DANGER":
                if time.time() - flash_t > 0.5:
                    flash_state = not flash_state
                    flash_t = time.time()
                if flash_state:
                    overlay = display.copy()
                    cv2.rectangle(overlay, (0,0), (FRAME_W, FRAME_H), (0,0,255), -1)
                    display = cv2.addWeighted(overlay, 0.12, display, 0.88, 0)
                cv2.putText(display, "DANGER DANGER", (40, FRAME_H - 40), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0,0,255), 3)
        else:
            cv2.putText(display, "State: NO HAND", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180,180,180), 2)

        # FPS
        now = time.time()
        fps = 0.9*fps + 0.1*(1.0/(now-prev+1e-6))
        prev = now
        cv2.putText(display, f"FPS: {fps:.1f}", (FRAME_W-140,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),1)

        cv2.imshow("POC Hand Boundary", display)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




