from ultralytics import YOLO

# train new model
model = YOLO("yolov8n.yaml")

results = model.train(data="config.yaml", epochs=100, batch=-1)

# continue training
# model = YOLO("E:\PyCharmProjects\yolo_test\\runs\detect\\train4\weights\last.pt")
#
# results = model.train(resume=True)