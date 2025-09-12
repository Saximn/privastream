from ultralytics import YOLO

#model = YOLO("best.pt")

#model.export(format="engine", dynamic=True)

# Load the exported TensorRT model
tensorrt_model = YOLO("best.engine")
tensort2 = YOLO("old_best.engine")

# Run inference
results = tensorrt_model("https://ultralytics.com/images/bus.jpg")
results2 = tensort2("https://ultralytics.com/images/bus.jpg")
