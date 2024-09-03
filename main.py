from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
import numpy as np
import cv2
from ultralytics import YOLO
import base64

app = FastAPI()

model = YOLO('yolo.pt')

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def main():
    return templates.TemplateResponse("index.html", {"request": {}})

@app.get("/about", response_class=HTMLResponse)
async def about():
    return templates.TemplateResponse("about.html", {"request": {}})

def img_pred_yolo(image):
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    results = model.predict(opencv_image)
    
    labels = results[0].names
    coordinates = results[0].boxes.xyxy.numpy()
    confidences = results[0].boxes.conf.numpy()
    class_ids = results[0].boxes.cls.numpy()

    detected_class = None
    if len(class_ids) > 0:
        detected_class_id = int(class_ids[0])  
        detected_class = labels[detected_class_id]

    annotated_image = np.array(image).copy()

    if detected_class == 'no_tumor':
        return np.array(image), detected_class
    else:
        for i in range(len(class_ids)):
            label = int(class_ids[i])
            x1, y1, x2, y2 = coordinates[i]
            confidence = confidences[i]
            cv2.rectangle(annotated_image, (int(x1), int(y1)), 
                          (int(x2), int(y2)), (255, 0, 0), 2)

    return annotated_image, detected_class

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    annotated_image, detected_class = img_pred_yolo(image)

    img_byte_arr = io.BytesIO()
    Image.fromarray(annotated_image).save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    result_sentence = f"The detected tumor is {detected_class.capitalize()}" if detected_class and detected_class != 'no_tumor' else "No tumor detected"

    return templates.TemplateResponse("result.html", {
        "request": {},
        "image_data": base64.b64encode(img_byte_arr).decode('utf-8'),
        "result_sentence": result_sentence
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
