import hashlib
import json


def get_serialize_path(original_file):
    hash_object = hashlib.sha256(original_file.encode())
    digest = hash_object.hexdigest()[:8]
    return "video_perspective_cropping_%s.save.json" % digest


def serialize_state(original_file, keypoints):
    print("Saving file...")
    with open(get_serialize_path(original_file), "w") as outfile:
        json.dump(keypoints, outfile)
        print("Saved %s keypoint(s)" % len(keypoints))


def deserialize_state(original_file):
    print("Loading file...")
    try:
        with open(get_serialize_path(original_file), "r") as infile:
            keypoints = json.load(infile)
            print("Loaded %s keypoint(s)" % len(keypoints))
            return keypoints
    except FileNotFoundError:
        print("No file found, initializing a new state")
        return []
