import sys

import cv2
import numpy as np

from video_perspective_cropping.common import deserialize_state


def main():
    assert len(sys.argv) == 4 + 1, \
        "Usage: ./restore <input_video.mp4> <output_video.avi> <target_width> <target_height>"
    input_video_path = sys.argv[1]
    output_video_path = sys.argv[2]
    target_width = int(sys.argv[3])
    target_height = int(sys.argv[4])

    keypoints = deserialize_state(input_video_path)

    assert len(keypoints) > 0, "No keypoints are defined"

    capture = cv2.VideoCapture(input_video_path)
    if not capture.isOpened():
        raise Exception("Cannot open video file")

    video_fps = capture.get(cv2.CAP_PROP_FPS)
    video_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_seconds = video_frame_count / video_fps

    _, initial_frame = capture.read()
    video_width = initial_frame.shape[1]
    video_height = initial_frame.shape[0]

    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('M','J','P','G'), video_fps, (target_width, target_height))

    frame_id = 0
    while True:
        ret, frame = capture.read()
        if ret:
            index_before = None
            index_after = None
            exists_same = False
            for index, (keypoint, _) in enumerate(keypoints):
                if keypoint <= frame_id and (index_before is None or keypoint > keypoints[index_before][0]):
                    index_before = index
                if keypoint >= frame_id and (index_after is None or keypoint < keypoints[index_after][0]):
                    index_after = index
            if index_before is None:
                interpolated = keypoints[index_after][1]
            elif index_after is None:
                interpolated = keypoints[index_before][1]
            else:
                keypoint_before, _ = keypoints[index_before]
                keypoint_after, _ = keypoints[index_after]
                v_interpolation = (frame_id - keypoint_before) / max(keypoint_after - keypoint_before, 1)
                interpolated = []
                for i in range(len(keypoints[index_before][1])):
                    x_before, y_before = keypoints[index_before][1][i]
                    x_after, y_after = keypoints[index_after][1][i]
                    def interpolate(v_before, v_after):
                        return v_before + (v_after - v_before) * v_interpolation
                    interpolated.append((interpolate(x_before, x_after), interpolate(y_before, y_after)))

            desired_output_points = [(0, 0), (target_width, 0), (0, target_height), (target_width, target_height)]
            perspective_transform = cv2.getPerspectiveTransform(np.float32(interpolated), np.float32(desired_output_points))
            warped_frame = cv2.warpPerspective(frame, perspective_transform, (target_width, target_height))
            #cv2.imshow("a", warped)
            #cv2.waitKey(0)
            video_writer.write(warped_frame)
        else:
            break

        frame_id += 1
        #if frame_id > 790:
        #    break

        if frame_id % 25 == 0:
            print("%.2f%% (%s)" % (100 * frame_id / video_frame_count, frame_id))

    capture.release()
    video_writer.release()


if __name__ == "__main__":
    main()
