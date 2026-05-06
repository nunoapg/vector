import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import vtracer
import cv2
import numpy as np
import tempfile
import os

app = FastAPI()

@app.post("/vectorize")
async def vectorize_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
        cv2.imwrite(tmp_in.name, img)
        tmp_out = tmp_in.name.replace(".png", ".svg")
        
        vtracer.convert_image_to_svg(
            tmp_in.name,
            tmp_out,
            mode='spline',
            precision=2,
            filter_speckle=4,
            color_precision=6,
            hierarchical='stacked'
        )
        
        with open(tmp_out, "r") as f:
            svg_data = f.read()
            
        os.unlink(tmp_in.name)
        os.unlink(tmp_out)
        
    return Response(content=svg_data, media_type="image/svg+xml")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))