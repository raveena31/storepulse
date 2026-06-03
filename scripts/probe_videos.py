import cv2, os, sys

BASE = r"data\cctv\CCTV Footage"
videos = {
    "CAM_01": os.path.join(BASE, "CAM 1.mp4"),
    "CAM_02": os.path.join(BASE, "CAM 2.mp4"),
    "CAM_03": os.path.join(BASE, "CAM 3.mp4"),
    "CAM_04": os.path.join(BASE, "CAM 4.mp4"),
    "CAM_05": os.path.join(BASE, "CAM 5.mp4"),
}

for name, path in videos.items():
    cap = cv2.VideoCapture(path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur_s = total / fps if fps > 0 else 0
    ret, frame = cap.read()
    cap.release()
    size_mb = os.path.getsize(path) / 1e6
    print(f"{name}: {w}x{h}  fps={fps:.1f}  frames={total}  dur={dur_s/60:.1f}min  size={size_mb:.0f}MB  readable={ret}")
