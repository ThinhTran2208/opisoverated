# GPU production deployment

This branch separates the production runtime into two GPU services:

- `inference-core` on `127.0.0.1:8000`
- `vlm` on `127.0.0.1:8001`

Only `inference-core` should be reachable by the web application. Keep the VLM
service private on the Docker host/network.

## 1. Prepare the GPU host

Recommended baseline:

- Ubuntu 22.04 or 24.04, x86-64
- NVIDIA driver with `nvidia-smi` working
- Docker Engine and the Docker Compose plugin
- NVIDIA Container Toolkit configured for Docker
- enough disk space for the RF-DETR, FashionCLIP and Qwen3-VL images/models

Verify the host before registering it:

```bash
nvidia-smi
docker version
docker compose version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 2. Register the GitHub self-hosted runner

Open this repository in GitHub, then go to:

```text
Settings → Actions → Runners → New self-hosted runner
```

Choose Linux and x64, then run the exact registration commands GitHub generates
on the GPU host. Add the custom label:

```text
gpu
```

The deployment workflow expects all four labels:

```text
self-hosted, linux, x64, gpu
```

Install the runner as a system service after registration, using the
`svc.sh` commands shown by GitHub. Do not paste the short-lived registration
token into the repository, workflow files or issue comments.

## 3. Deploy the containers

Run the workflow manually:

```text
Actions → Deploy production inference to GPU → Run workflow
```

The workflow:

1. verifies CUDA, Docker and Compose;
2. builds the two production images;
3. starts `docker-compose.gpu.yml`;
4. waits for both health endpoints;
5. prints the inference-core production contract.

Local verification on the host:

```bash
curl --fail http://127.0.0.1:8001/healthz
curl --fail http://127.0.0.1:8000/healthz
```

## 4. Provide one HTTPS endpoint

The ChatGPT Site requires an HTTPS base URL that reaches
`http://127.0.0.1:8000` on the GPU host. A named Cloudflare Tunnel is preferred
because it avoids opening port 8000 directly.

Configure a public hostname such as:

```text
https://outfit-api.example.com → http://127.0.0.1:8000
```

Run the Cloudflare connector as a system service on the GPU host. Do not expose
port 8001 or the Docker network publicly.

Verify from another network:

```bash
curl --fail https://outfit-api.example.com/healthz
```

The Site runtime must then set:

```text
FASHION_API_BASE_URL=https://outfit-api.example.com
```

and publish a new Site deployment so the environment revision is applied.

## 5. Production notes

- The HTTP API accepts image uploads up to 10 MB.
- `POST /v1/analyze-outfit` may take several minutes while the VLM model is cold.
- The compatibility score is calibrated model output, not a universal fashion rating.
- Replacement recommendation is not implemented in production inference v1.
- Restrict access or add rate limiting before making the GPU endpoint broadly public.
