"""Save one frame per camera for visual inspection."""
import cv2, os

BASE = r"data\cctv\CCTV Footage"
OUT  = r"data\cctv\frames"
os.makedirs(OUT, exist_ok=True)

cams = {
    "CAM_01": os.path.join(BASE, "CAM 1.mp4"),
    "CAM_02": os.path.join(BASE, "CAM 2.mp4"),
    "CAM_03": os.path.join(BASE, "CAM 3.mp4"),
    "CAM_04": os.path.join(BASE, "CAM 4.mp4"),
    "CAM_05": os.path.join(BASE, "CAM 5.mp4"),
}

for cam_id, path in cams.items():
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # grab frame at ~30% through the clip (more activity than frame 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.3))
    ret, frame = cap.read()
    cap.release()
    if ret:
        out_path = os.path.join(OUT, f"{cam_id}.jpg")
        cv2.imwrite(out_path, frame)
        print(f"Saved {out_path}")
