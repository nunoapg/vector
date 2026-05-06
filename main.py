import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, HTMLResponse
import vtracer
import cv2
import numpy as np
import os
import traceback

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def read_item():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vectorize AI Clone</title>
        <style>
            body { font-family: sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; margin: 0; background-color: #121212; color: white; }
            .container { background: #1e1e1e; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); text-align: center; width: 80%; max-width: 600px; }
            input { margin: 1rem 0; color: white; }
            button { background: #3f51b5; color: white; border: none; padding: 12px 24px; border-radius: 4px; cursor: pointer; font-weight: bold; }
            button:disabled { background: #555; cursor: not-allowed; }
            #result { margin-top: 2rem; width: 100%; display: flex; flex-direction: column; align-items: center; }
            svg { background: white; max-width: 100%; height: auto; border-radius: 4px; border: 5px solid #333; }
            #download { display: none; margin-top: 1rem; color: #4caf50; text-decoration: none; font-weight: bold; border: 2px solid #4caf50; padding: 8px 16px; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Vectorize AI Clone</h1>
            <p>Processamento Avançado de Imagem</p>
            <input type="file" id="fileInput" accept="image/*">
            <br>
            <button onclick="uploadFile()" id="btn">Converter para SVG</button>
            <div id="result"></div>
            <a id="download">Descarregar SVG</a>
        </div>

        <script>
            async function uploadFile() {
                const fileInput = document.getElementById('fileInput');
                const btn = document.getElementById('btn');
                const resultDiv = document.getElementById('result');
                const downloadBtn = document.getElementById('download');
                
                if (fileInput.files.length === 0) {
                    alert('Por favor, escolha um ficheiro primeiro.');
                    return;
                }

                btn.disabled = true;
                btn.innerText = 'A processar...';
                resultDiv.innerHTML = '';
                downloadBtn.style.display = 'none';
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                try {
                    const response = await fetch('/vectorize', { 
                        method: 'POST', 
                        body: formData 
                    });

                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || 'Erro no servidor');
                    }

                    const svgText = await response.text();
                    resultDiv.innerHTML = svgText;
                    
                    const blob = new Blob([svgText], {type: 'image/svg+xml'});
                    const url = URL.createObjectURL(blob);
                    downloadBtn.href = url;
                    downloadBtn.download = 'vector.svg';
                    downloadBtn.style.display = 'inline-block';
                } catch (error) {
                    alert('Erro: ' + error.message);
                } finally {
                    btn.disabled = false;
                    btn.innerText = 'Converter para SVG';
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/vectorize")
async def vectorize_image(file: UploadFile = File(...)):
    tmp_in = f"input_{file.filename}"
    tmp_out = f"output_{file.filename}.svg"
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        # --- TRUQUE DE PRÉ-PROCESSAMENTO ---
        # 1. Aumentar a escala (Upscaling) para dar mais "espaço" às curvas
        img_resized = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # 2. Leve desfoque para suavizar o antialiasing e degraus de pixéis
        img_blurred = cv2.GaussianBlur(img_resized, (3, 3), 0)
        
        cv2.imwrite(tmp_in, img_blurred)
        
        vtracer.convert_image_to_svg_py(
            tmp_in,
            tmp_out,
            mode='spline',
            path_precision=1,
            corner_threshold=100,
            filter_speckle=2,
            color_precision=2,
            hierarchical='stacked'
        )
        
        with open(tmp_out, "r") as f:
            svg_data = f.read()
            
        return Response(content=svg_data, media_type="image/svg+xml")

    except Exception as e:
        print(f"ERRO: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(tmp_in): os.remove(tmp_in)
        if os.path.exists(tmp_out): os.remove(tmp_out)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
