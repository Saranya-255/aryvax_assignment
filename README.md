This project is a Proof of Concept (POC) demonstrating real-time hand / fingertip tracking using only classical computer-vision techniques (OpenCV + NumPy).
A virtual object (circle) is drawn on the screen, and the system classifies the hand's proximity to this object into:

🟢 SAFE – hand far from virtual boundary

🟡 WARNING – hand approaching boundary

🔴 DANGER – hand touching or extremely close

When in danger mode, the system displays a flashing “DANGER DANGER” warning on the camera feed.

The goal of the assignment is to:

Track the user’s hand and estimate fingertip position in real time

Draw a virtual boundary on the screen

Detect when the fingertip approaches or touches the boundary

Visually classify the interaction into SAFE / WARNING / DANGER

Provide a smooth, real-time visual overlay
