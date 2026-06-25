# Steam game recommender

Steam game recommender using Agentic AI, vector databases and other APIs

# Local Development Environment

To create a local env run the commands: 

```
python3 -m venv .venv
source .venv/bin/activate # Linux
.\.venv\Scripts\Activate.ps1 # Windows (PowerShell)
pip install -r requirements.txt
```

# Running Qdrant locally

## Docker

Qdrant runs as a container. With Docker installed, pull and start it with data persisted to a local `qdrant_storage` folder:

```
docker run -p 6333:6333 -p 6334:6334 -v ${PWD}/qdrant_storage:/qdrant/storage qdrant/qdrant
```

On Linux/macOS replace `${PWD}` with `$(pwd)`.

- REST API and web dashboard: http://localhost:6333 (dashboard at http://localhost:6333/dashboard)
- gRPC API: localhost:6334

## Without docker

Qdrant ships a native Windows binary. The release zip contains only `qdrant.exe`, so the web dashboard also needs the separate Web UI static files. Both are downloaded into the gitignored `.qdrant/` folder (PowerShell):

```powershell
$root = ".qdrant"
New-Item -ItemType Directory -Force -Path $root | Out-Null

# Server binary
Invoke-WebRequest "https://github.com/qdrant/qdrant/releases/download/v1.18.2/qdrant-x86_64-pc-windows-msvc.zip" -OutFile "$root\qdrant.zip"
Expand-Archive "$root\qdrant.zip" -DestinationPath $root -Force

# Web UI (dashboard) static files -> .qdrant/static
Invoke-WebRequest "https://github.com/qdrant/qdrant-web-ui/releases/download/v0.2.13/dist-qdrant.zip" -OutFile "$root\webui.zip"
Expand-Archive "$root\webui.zip" -DestinationPath "$root\webui_tmp" -Force
Move-Item "$root\webui_tmp\dist" "$root\static" -Force
Remove-Item "$root\qdrant.zip","$root\webui.zip" -Force; Remove-Item "$root\webui_tmp" -Recurse -Force
```

Start the server from the `.qdrant` folder (it looks for `./static` and writes data to `./storage` in the working dir):

```powershell
cd .qdrant
.\qdrant.exe
```

- REST API: http://localhost:6333
- Web dashboard: http://localhost:6333/dashboard
- gRPC API: localhost:6334

The Python client then connects with:


```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
```