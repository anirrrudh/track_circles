import csv
import math

import cv2
import numpy as np


class Circle:
    def __init__(self, x, y, r=None, frame_no=None, speed=None):
        self.x = int(x)
        self.y = int(y)
        self.r = int(r) if r else None

        self.frame_no = frame_no
        self.speed = speed

    def center(self):
        return self.x, self.y


def find_circles(binary_frame, options=(10, 200, 15, 30, 30)):
    """
    Find circles in frame using opencv's Hough Circle Transform method.
    """
    colored = cv2.cvtColor(binary_frame, cv2.COLOR_GRAY2BGR)
    grayed = cv2.cvtColor(colored, cv2.COLOR_BGR2GRAY)
    grayed = cv2.GaussianBlur(grayed, (5, 5), 3)

    found_circles = cv2.HoughCircles(
        grayed,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=options[2],
        param1=options[3],
        param2=options[4],
        minRadius=options[0],
        maxRadius=options[1]
    )

    if found_circles is not None:
        found_circles = list(map(lambda c: Circle(c[0], c[1], r=c[2]), found_circles[0]))

    return colored, found_circles


def draw_circles(frame, circles, color=(0, 255, 255), text=""):
    if circles is not None:
        if type(circles) is not list:
            circles = [circles]

        for c in circles:
            # Draw the circle
            cv2.circle(frame, c.center(), c.r, color, 3)

            # Draw the center of the circle
            # cv2.circle(_frame, (i[0], i[1]), 2, (255, 0, 255), 3)

            # Display text
            cv2.putText(frame, text, (c.x - 7, c.y + 7), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Display the radius
            # cv2.putText(_frame, str(_i.r), _i.center(), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    return frame


def find_circle_enclosing_contours(binary_frame):
    """
    In cases where circles were sufficiently isolated by color masking, detecting their location by
    locating contours in the masked videos was more effective than using the HoughGradients method.
    """
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    all_contours = np.concatenate(contours)
    center, radius = cv2.minEnclosingCircle(all_contours)

    if not center or not radius:
        return None
    else:
        return Circle(center[0], center[1], r=radius)


def dilate_and_erode(_frame, size=3, iterations=1):
    kernel_size = size
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    cv2.dilate(_frame, kernel, _frame, iterations=iterations)
    cv2.erode(_frame, kernel, _frame, iterations=iterations)


def find_closest(circles, prev, frame_no=None):
    """
    Find in 'circles' the closest one to 'prev' by using 'diff' method
    """
    if circles is None:
        return None, None

    if len(circles) == 1:
        if frame_no is not None:
            circles[0].frame_no = frame_no
        return 0, circles[0]

    if prev is None:
        return None, None

    if frame_no is not None:
        for circle in circles:
            circle.frame_no = frame_no

    match = 0
    max_dist = 10_000
    for i, circle in enumerate(circles):
        d = diff(circle, prev)
        if d < max_dist:
            max_dist = d
            match = i
    return match, circles[match]


def dist(curr: Circle, prev: Circle):
    """
    Returns Euclidean distance between the centers
    """
    return math.sqrt((curr.x - prev.x) ** 2 + (curr.y - prev.y) ** 2)


def speed(curr: Circle, prev: Circle):
    """
    Distance divided by number of frames elapsed
    """
    location_diff = dist(curr, prev)
    # Speed depends on how many frames (how much time) have passed since last location
    frame_diff = curr.frame_no - prev.frame_no
    return location_diff / frame_diff


def update_speed(curr: Circle, prev: Circle):
    if curr is None:
        return prev

    if prev is None:
        return curr

    curr.speed = speed(curr, prev)
    return curr


def diff(curr: Circle, prev: Circle):
    """
    A measure of difference in the states of two circles. The lower the difference, higher
    the likelihood of 'curr' being the same circle as 'prev'. The state comprises the location
    and speed. I considered speed over velocity because I noticed that speed of movement changed
    less erratically than the direction of movement.
    This measure is a combination of differences in both state variables, balanced by a
    coefficient currently set to 4.3 (after trial and error) for best tracking.
    """
    location_diff = dist(curr, prev)

    new_speed = speed(curr, prev)
    speed_diff = 0 if prev.speed is None else math.fabs(new_speed - prev.speed)

    return location_diff + 4.3 * speed_diff


def find_likely_pair(circles: list[Circle], prev1: Circle, prev2: Circle):
    """
    When more than 2 circles are identified by the HoughGradient algorith,
    this method can be used to separate out the erroneous ones.
    """
    idx1, circ1 = find_closest(circles, prev1)
    circles.pop(idx1)

    idx2, circ2 = find_closest(circles, prev2)

    return [circ1, circ2]


def match_circles(frame_no: int, circles: list[Circle], prev1: Circle, prev2: Circle):
    """
    When circles are not uniquely identifiable by their features (like color or radius),
    this method can be used to match/associate newly identified circles to previously identified ones.
    Supports tracking movements of only two identical circles.

    :param frame_no: Frame number of the new circles
    :param circles: List of new circles detected
    :param prev1: First of the two previously known circles
    :param prev2: Second of the two previously known circles
    :return: Tuple of two circles that are likely matches for prev1 and prev2 respectively
    """
    # Case 1: No circles detected
    if circles is None:
        return None, None

    # Case 2: One circle is detected
    if len(circles) == 1:
        circ = circles[0]
        circ.frame_no = frame_no

        # Case 2.1: No previous circles known
        if prev1 is None and prev2 is None:
            # Return in any order
            return circ, None

        # Case 2.2: One previous circle known
        if prev1 is None:
            if diff(circ, prev2) > 150:
                # If the new circle is too far from the one
                # previously known, it's probably the other one
                return circ, None
            else:
                return None, circ

        # Mirror case
        if prev2 is None:
            if diff(circ, prev1) > 150:
                return None, circ
            else:
                return circ, None

        # Case 2.3: Both previous circles known
        if diff(circ, prev1) < diff(circ, prev2):
            return circ, None
        else:
            return None, circ

    # Case 3: Two circles detected
    if len(circles) == 2:
        circ1 = circles[0]
        circ2 = circles[1]

        circ1.frame_no = frame_no
        circ2.frame_no = frame_no

        # Case 3.1: No previous circles known
        if prev1 is None and prev2 is None:
            # Return in any order
            return circ1, circ2

        # Case 3.2: One previous circle known
        if prev1 is None:
            if diff(circ1, prev2) < diff(circ2, prev2):
                return circ2, circ1
            else:
                return circ1, circ2

        # Mirror case
        if prev2 is None:
            if diff(circ1, prev1) < diff(circ2, prev1):
                return circ1, circ2
            else:
                return circ2, circ1

        # Case 3.3: Both previous circles known

        # Case 3.3.1: New circles are close to different previous circles
        if diff(circ1, prev1) < diff(circ1, prev2) and diff(circ2, prev2) < diff(circ2, prev1):
            return circ1, circ2

        if diff(circ1, prev2) < diff(circ1, prev1) and diff(circ2, prev1) < diff(circ2, prev2):
            return circ2, circ1

        # Case 3.3.2: New circles are close to the same previous circle
        if diff(circ1, prev1) < diff(circ1, prev2) and diff(circ2, prev1) < diff(circ2, prev2):
            # Choose the one that's closer among the two
            if diff(circ1, prev1) < diff(circ2, prev1):
                return circ1, circ2
            else:
                return circ2, circ1
        else:
            # Mirror case
            if diff(circ1, prev2) < diff(circ2, prev2):
                return circ2, circ1
            else:
                return circ1, circ2

    # Case 4: More than 2 circles detected
    if len(circles) > 2:

        # Case 4.1: One or more previous circles unknown
        if prev1 is None or prev2 is None:
            # It is not possible to determine which circles are real
            return None, None

        # Case 4.2: Both previous circles known
        for circ in circles:
            circ.frame_no = frame_no

        # Find 2 closer ones and try again
        circles = find_likely_pair(circles, prev1, prev2)
        return match_circles(frame_no, circles, prev1, prev2)


def add_to_history(frame_no, circle_no, circle, history: list[dict]):
    if len(history) < frame_no:
        history.append({'Frame': frame_no})

    history[frame_no - 1][f"Bubble_{circle_no}"] = "( - , - )" if circle is None else f"({circle.x}, {circle.y})"


def save_history(history: list[dict]):
    with open('bubble_locations.txt', 'w') as f:
        writer = csv.DictWriter(f, history[0], delimiter="\t")
        writer.writeheader()
        for h in history:
            writer.writerow(h)
