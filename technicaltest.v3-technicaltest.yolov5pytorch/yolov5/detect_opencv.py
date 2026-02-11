import cv2
import torch
import numpy as np
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# =====================
# LOAD MODEL
# =====================
weights = "runs/train/exp2/weights/last.pt"
device = select_device("cpu")

model = DetectMultiBackend(weights, device=device)
stride, names = model.stride, model.names
model.eval()

# Warmup
model.warmup(imgsz=(1, 3, 640, 640))

print("Class names:", names)

# =====================
# OPEN CAMERA
# =====================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # =====================
    # PREPROCESS
    # =====================
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    img = img.unsqueeze(0)

    # =====================
    # INFERENCE
    # =====================
    with torch.no_grad():
        pred = model(img, augment=False, visualize=False)

    # =====================
    # NMS
    # =====================
    pred = non_max_suppression(
        pred,
        conf_thres=0.25,  
        iou_thres=0.45
    )

    det = pred[0]

    # =====================
    # DRAW BOX
    # =====================
    if det is not None and len(det):
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

        for *xyxy, conf, cls in det:
            cls = int(cls)
            label = f"{names[cls]} {conf:.2f}"

            cv2.rectangle(frame,
                          (int(xyxy[0]), int(xyxy[1])),
                          (int(xyxy[2]), int(xyxy[3])),
                          (0, 255, 0), 2)

            cv2.putText(frame, label,
                        (int(xyxy[0]), int(xyxy[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    cv2.imshow("YOLOv5 Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
