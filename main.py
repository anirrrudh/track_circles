from utils import *

# Lower and upper thresholds for color masking
# Circles are of 5 colors:
#   white (1 count), gray (6 counts), green (1), orange (1), and blue (1)
lower_white = (251, 251, 251)
upper_white = (255, 255, 255)

lower_gray = (227, 227, 227)
upper_gray = (233, 233, 233)

lower_green = (139, 205, 165)
upper_green = (147, 211, 171)

lower_orange = (170, 200, 243)
upper_orange = (175, 205, 248)

lower_blue = (228, 196, 176)
upper_blue = (232, 200, 180)

# Optimal HoughCircle function parameters:
#   (min_radius, max_radius, min_dist, param1, param2)
size_1_options = (97, 110, 30, 30, 10)
size_2_options = (55, 72, 15, 30, 20)
size_3_options = (37, 52, 15, 30, 20)
size_4_options = (22, 34, 15, 30, 15)

# Previous known circles
prev4 = None
prev5, prev6 = None, None
prev7 = None
prev8 = None
prev9, prev10 = None, None

# Initialize video capture
cap = cv2.VideoCapture('Track_Circles.mp4')

# Initialize video writer
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*'avc1')
output_file = 'Track_Circles_Output.mp4'
out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

frame_no = 0

# Save bubble locations for writing to text file
history = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_no += 1
    output = frame

    frame = cv2.GaussianBlur(frame, (5, 5), 1.5)

    # Tracking the Blue, Green, and Orange circles

    mask_orange = cv2.inRange(frame, lower_orange, upper_orange)
    mask_blue = cv2.inRange(frame, lower_blue, upper_blue)
    mask_green = cv2.inRange(frame, lower_green, upper_green)

    for i, mask in enumerate([mask_orange, mask_blue, mask_green]):
        circle = find_circle_enclosing_contours(mask)
        draw_circles(output, circle, color=(255, 0, 255), text=str(i + 1))
        add_to_history(frame_no, i + 1, circle, history)

    # Tracking the white circle

    mask_white = cv2.inRange(frame, lower_white, upper_white)

    _, white_circles = find_circles(mask_white, size_2_options)
    _, white_circle = find_closest(white_circles, prev4, frame_no=frame_no)
    draw_circles(output, white_circle, text=str(4))
    prev4 = prev4 if white_circle is None else white_circle
    add_to_history(frame_no, 4, white_circle, history)

    # Tracking gray circles - size 1

    mask_gray = cv2.inRange(frame, lower_gray, upper_gray)

    _, gray_1_circles = find_circles(mask_gray, size_1_options)
    curr5, curr6 = match_circles(frame_no, gray_1_circles, prev5, prev6)

    draw_circles(output, curr5, text=str(5))
    draw_circles(output, curr6, text=str(6))

    prev5 = update_speed(curr5, prev5)
    prev6 = update_speed(curr6, prev6)

    add_to_history(frame_no, 5, curr5, history)
    add_to_history(frame_no, 6, curr6, history)

    # Tracking gray circles - sizes 2 & 3

    _, gray_2_circles = find_circles(mask_gray, size_2_options)
    _, gray_2_circle = find_closest(gray_2_circles, prev7, frame_no=frame_no)
    draw_circles(output, gray_2_circle, text=str(7))
    prev7 = prev7 if gray_2_circle is None else gray_2_circle
    add_to_history(frame_no, 7, gray_2_circle, history)

    _, gray_3_circles = find_circles(mask_gray, size_3_options)
    _, gray_3_circle = find_closest(gray_3_circles, prev8, frame_no=frame_no)
    draw_circles(output, gray_3_circle, text=str(8))
    prev8 = prev8 if gray_3_circle is None else gray_3_circle
    add_to_history(frame_no, 8, gray_2_circle, history)

    # Tracking gray circles - size 4

    _, gray_4_circles = find_circles(mask_gray, size_4_options)
    curr9, curr10 = match_circles(frame_no, gray_4_circles, prev9, prev10)

    draw_circles(output, curr9, text=str(9))
    draw_circles(output, curr10, text=str(10))

    prev9 = update_speed(curr9, prev9)
    prev10 = update_speed(curr10, prev10)

    add_to_history(frame_no, 9, curr9, history)
    add_to_history(frame_no, 10, curr10, history)

    # Save video

    out.write(output)

    # cv2.imshow('Circle Tracking', output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()

cv2.destroyAllWindows()

save_history(history)
