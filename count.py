from collections import Counter
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # nano weights, auto-downloads on first run

counts = []
results = model.predict(source="IMG_9858.MOV", stream=True, verbose=False, save=True)

for i, r in enumerate(results):
    labels = [model.names[int(c)] for c in r.boxes.cls]
    counts.append(len(labels))
    print(f"Frame {i}: {len(labels)} objects  {dict(Counter(labels))}")

print(f"\nFrames processed: {len(counts)}")
print(f"Max objects in a frame: {max(counts)}")
print(f"Average per frame: {sum(counts) / len(counts):.1f}")
