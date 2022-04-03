import sys

import cv2
import numpy as np

from video_perspective_cropping.common import deserialize_state, serialize_state

WINDOW_NAME = "video-perspective-cropping"

KEY_ARROW_ESCAPE = 27
KEY_ARROW_RIGHT = 2555904
KEY_ARROW_LEFT = 2424832
KEY_ARROW_UP = 2490368
KEY_ARROW_DOWN = 2621440
KEY_ARROW_PLUS = 43
KEY_ARROW_MINUS = 45
KEY_DELETE = 3014656
KEY_ENTER = 13
KEY_ARROW_ARROW_UP = 2162688
KEY_ARROW_ARROW_DOWN = 2228224

STEP_PLUS = 10


is_mouse_down = False
slider_frame_value = 0

keypoints = []
selected_knob = None

last_frame_index = None
last_frame_data = None


def format_duration(duration_seconds):
    return "%.2f" % duration_seconds


def main():
    global slider_frame_value
    global selected_knob
    global last_frame_index
    global last_frame_data
    assert len(sys.argv) == 1 + 1, "Usage: ./gui <input_video.mp4>"
    video_path = sys.argv[1]

    keypoints = deserialize_state(video_path)

    width = 1000
    height_control = 100
    slider_horizontal_margin = 25
    slider_top_margin = 40
    keypoints_top_margin = 70
    keypoints_radius = 3
    slider_knob_radius = 10
    assert width > slider_horizontal_margin * 2
    assert height_control > slider_top_margin

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise Exception("Cannot open video file")

    video_fps = capture.get(cv2.CAP_PROP_FPS)
    video_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_seconds = video_frame_count / video_fps

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    _, initial_frame = capture.read()
    video_width = initial_frame.shape[1]
    video_height = initial_frame.shape[0]
    height = width * video_height // video_width

    last_frame_index = 0
    last_frame_data = initial_frame

    def redraw():
        global last_frame_index
        global last_frame_data

        if slider_frame_value == last_frame_index:
            frame = np.copy(last_frame_data)
        else:
            capture.set(cv2.CAP_PROP_POS_FRAMES, min(slider_frame_value, video_frame_count - 2))
            ret, frame = capture.read()
            last_frame_index = slider_frame_value
            last_frame_data = np.copy(frame)

        frame = cv2.resize(frame, (width, height))

        control_image = np.zeros((height_control, width, frame.shape[2]), np.uint8)
        control_image[...] = (255, 255, 255)
        cv2.putText(control_image,
                    "%ss / %ss -- (%s / %s)" % (
                        format_duration(video_duration_seconds * slider_frame_value / video_frame_count), format_duration(video_duration_seconds),
                        slider_frame_value, video_frame_count
                    ),
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.line(control_image,
                 (slider_horizontal_margin, slider_top_margin),
                 (width - slider_horizontal_margin, slider_top_margin),
                 (0, 0, 0), 2)
        cv2.ellipse(control_image,
                    (int(slider_horizontal_margin +
                     (width - 2 * slider_horizontal_margin) * slider_frame_value / video_frame_count), slider_top_margin),
                    (slider_knob_radius, slider_knob_radius), 0, 0, 360, (0, 0, 255), -1)
        def convert_point(p):
            px, py = p
            return int(width * px / video_width), int(height * py / video_height)
        def points_to_polygon(points):
            return [points[0], points[1], points[3], points[2], points[0]]

        index_before = None
        index_after = None
        exists_same = False
        for index, (keypoint, _) in enumerate(keypoints):
            if keypoint == slider_frame_value:
                exists_same = True
            if keypoint < slider_frame_value and (index_before is None or keypoint > keypoints[index_before][0]):
                index_before = index
            if keypoint > slider_frame_value and (index_after is None or keypoint < keypoints[index_after][0]):
                index_after = index

        if not exists_same and (index_before is not None or index_after is not None):
            if index_before is None:
                interpolated = keypoints[index_after][1]
            elif index_after is None:
                interpolated = keypoints[index_before][1]
            else:
                keypoint_before, _ = keypoints[index_before]
                keypoint_after, _ = keypoints[index_after]
                v_interpolation = (slider_frame_value - keypoint_before) / (keypoint_after - keypoint_before)
                interpolated = []
                for i in range(len(keypoints[index_before][1])):
                    x_before, y_before = keypoints[index_before][1][i]
                    x_after, y_after = keypoints[index_after][1][i]
                    def interpolate(v_before, v_after):
                        return v_before + (v_after - v_before) * v_interpolation
                    interpolated.append((interpolate(x_before, x_after), interpolate(y_before, y_after)))

            interpolated_polygon = points_to_polygon(interpolated)
            for i in range(len(interpolated_polygon) - 1):
                p1 = interpolated_polygon[i]
                p2 = interpolated_polygon[i + 1]
                cv2.line(frame, convert_point(p1), convert_point(p2), (255, 0, 0), 1)

        for keypoint_frame, points in keypoints:
            is_current_frame = keypoint_frame == slider_frame_value
            color, radius = ((0, 0, 255), int(keypoints_radius * 2)) if is_current_frame else ((255, 0, 255), keypoints_radius)
            cv2.ellipse(control_image,
                        (int(slider_horizontal_margin +
                             (width - 2 * slider_horizontal_margin) * keypoint_frame / video_frame_count), keypoints_top_margin),
                        (radius, radius), 0, 0, 360, color, -1)
            if is_current_frame:
                for i, (point_x, point_y) in enumerate(points):
                    x = int(width * point_x / video_width)
                    y = int(height * point_y / video_height)
                    rectangle_size = 6
                    cv2.rectangle(frame, (x - rectangle_size, y - rectangle_size),
                                  (x + rectangle_size, y + rectangle_size), (0, 0, 255), 1)
                    cv2.line(frame, (x - rectangle_size, y - rectangle_size),
                             (x + rectangle_size, y + rectangle_size), (0, 0, 255), 1)
                    cv2.line(frame, (x - rectangle_size, y + rectangle_size),
                             (x + rectangle_size, y - rectangle_size), (0, 0, 255), 1)
                    sign_x = -rectangle_size if i % 2 == 0 else rectangle_size
                    sign_y = -rectangle_size if i < 2 else rectangle_size
                    cv2.rectangle(frame, (x + sign_x - 1, y + sign_y - 1),
                                  (x + sign_x + 1, y + sign_y + 1), (0, 255, 0), 2)
                points_polygon = points_to_polygon(points)
                for i in range(len(points_polygon) - 1):
                    p1 = points_polygon[i]
                    p2 = points_polygon[i + 1]
                    cv2.line(frame, convert_point(p1), convert_point(p2), (0, 0, 255), 1)

        canvas = cv2.vconcat([frame, control_image])

        def click_callback(event, x, y, flags, param):
            global slider_frame_value
            global is_mouse_down
            if event == cv2.EVENT_LBUTTONDOWN:
                is_mouse_down = True
            elif event == cv2.EVENT_LBUTTONUP:
                is_mouse_down = False

            if y >= height:  # Control
                if is_mouse_down:
                    value = max(min(round(video_frame_count * (x - slider_horizontal_margin) / (width - 2 * slider_horizontal_margin)), video_frame_count), 0)
                    slider_frame_value = value
                else:
                    return
            else:  # Image
                image_x = video_width * x / width
                image_y = video_height * y / height
                points = None
                for keypoint_index, ps in keypoints:
                    if keypoint_index == slider_frame_value:
                        points = ps
                if points is not None and is_mouse_down:
                    closest_index = None
                    closest_distance = None
                    for i in range(len(points)):
                        p_x, p_y = points[i]
                        def sq(v):
                            return v * v
                        distance_sq = sq(p_x - image_x) + sq(p_y - image_y)
                        if closest_distance is None or distance_sq < closest_distance:
                            closest_index = i
                            closest_distance = distance_sq
                    points[closest_index] = (image_x, image_y)
                else:
                    return
            redraw()

        cv2.setMouseCallback(WINDOW_NAME, click_callback)

        cv2.imshow(WINDOW_NAME, canvas)

    redraw()

    while True:
        key = cv2.waitKeyEx(0)
        if key == -1 or key == KEY_ARROW_ESCAPE:
            break
        elif key == KEY_ARROW_RIGHT:
            index = None
            for keypoint_index, _ in keypoints:
                if keypoint_index > slider_frame_value and (index is None or keypoint_index < index):
                    index = keypoint_index
            if index is not None:
                slider_frame_value = index
        elif key == KEY_ARROW_LEFT:
            index = None
            for keypoint_index, _ in keypoints:
                if keypoint_index < slider_frame_value and (index is None or keypoint_index > index):
                    index = keypoint_index
            if index is not None:
                slider_frame_value = index
        elif key == KEY_ARROW_ARROW_DOWN:
            index = None
            for keypoint_index, _ in keypoints:
                if index is None or keypoint_index > index:
                    index = keypoint_index
            if index is not None:
                slider_frame_value = index
        elif key == KEY_ARROW_ARROW_UP:
            index = None
            for keypoint_index, _ in keypoints:
                if index is None or keypoint_index < index:
                    index = keypoint_index
            if index is not None:
                slider_frame_value = index
        elif key == KEY_ARROW_PLUS:
            slider_frame_value = min(slider_frame_value + STEP_PLUS, video_frame_count)
        elif key == KEY_ARROW_MINUS:
            slider_frame_value = max(slider_frame_value - STEP_PLUS, 0)
        elif key == KEY_ARROW_UP:
            slider_frame_value = min(slider_frame_value + 1, video_frame_count)
        elif key == KEY_ARROW_DOWN:
            slider_frame_value = max(slider_frame_value - 1, 0)
        elif key == KEY_DELETE:
            index = None
            for i in range(len(keypoints)):
                keypoint, _ = keypoints[i]
                if keypoint == slider_frame_value:
                    index = i
            if index is not None:
                del keypoints[index]
        elif key == KEY_ENTER:
            closest_keypoint = None
            closest_points = None
            not_defined = True
            for i in range(len(keypoints)):
                keypoint, points = keypoints[i]
                if keypoint == slider_frame_value:
                    not_defined = False
                if keypoint < slider_frame_value and (closest_keypoint is None or closest_keypoint < keypoint):
                    closest_keypoint = keypoint
                    closest_points = points
            if not_defined:
                offset = 50
                if len(keypoints) == 0 or closest_points is None:
                    keypoints.append((slider_frame_value, [(offset, offset), (video_width - offset, offset),
                                                           (offset, video_height - offset),
                                                           (video_width - offset, video_height - offset)]))
                else:
                    new_points = []
                    for p in closest_points:
                        new_points.append(p)
                    keypoints.append((slider_frame_value, new_points))

        redraw()
        #print(key)
    cv2.destroyAllWindows()
    capture.release()

    serialize_state(video_path, keypoints)


if __name__ == "__main__":
    main()
