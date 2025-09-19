import cv2

image = cv2.imread("./images/front/front_20250919_152433_464920.jpg")

# Click event
if image is None:
    raise SystemExit("Failed to load image: front_20250919_152433_464920.jpg")

h, w = image.shape[:2]

window_name = "image"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
cv2.imshow(window_name, image)


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        norm_x = x / w
        norm_y = y / h
        print(f"x={norm_x:.6f}, y={norm_y:.6f}    (px=({x},{y}), size=({w},{h}))")

        vis = image.copy()
        cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow(window_name, vis)


cv2.setMouseCallback(window_name, on_mouse)

# Press 'q' or ESC to quit
while True:
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q'), ord('Q')):
        break

cv2.destroyAllWindows()