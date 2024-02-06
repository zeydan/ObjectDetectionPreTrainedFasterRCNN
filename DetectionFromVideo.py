import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision import transforms

cap = cv2.VideoCapture(0)

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights)
model.eval()

preprocess = weights.transforms()
transform = transforms.ToTensor()

while True:
    _, cv_image = cap.read()
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    torch_image = transform(rgb_image)
    batch = preprocess(torch_image).unsqueeze(0)

    prediction = model(batch)[0]

    labels = [weights.meta["categories"][i] for i in prediction["labels"]]

    for i in range(len(labels)):
        score = prediction['scores'][i].item()
        if score > 0.5:
            x1, y1, x2, y2 = prediction['boxes'][i] 
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = labels[i]
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(cv_image, f'{label}: {score:.4f}', (x1,y1+25), 1, 2, (255, 0 , 255), 2)

    cv2.imshow('Result', cv_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()