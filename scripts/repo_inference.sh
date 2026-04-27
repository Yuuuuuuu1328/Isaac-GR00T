#!/bin/bash

cleanup() {
    echo "Interrupt received, shutting down the server..."
    # Find the server process and kill it
    pkill -f "python scripts/inference_base.py"
    exit 0
}

# Trap interrupt signal to run cleanup
trap cleanup INT

# Check command line arguments
while getopts "s c h z" opt; do
  case $opt in
    s)
      server=true
      ;;
    c)
      client=true
      ;;
    h)
      http=true
      ;;
    z)
      zmq=true
      ;;
    \?)
      echo "Usage: $0 [-s server] [-c client] [-h http] [-z zmq]"
      exit 1
      ;;
  esac
done

# Default values for communication type and role
if [[ -z "$server" && -z "$client" ]]; then
    echo "Error: You must specify either -s (server) or -c (client)"
    exit 1
fi

if [[ -z "$http" && -z "$zmq" ]]; then
    echo "Error: You must specify either -h (http) or -z (zmq)"
    exit 1
fi

# Run server or client based on the flags
if [[ "$server" == "true" ]]; then
    if [[ "$http" == "true" ]]; then
        # HTTP server
        echo "Starting HTTP server..."
        python scripts/inference_base.py \
          --server \
          --http-server \
          --model-path "/home/jetson/Desktop/project/model/gr00t_weights/GR00T-N1.5-3B" \
          --embodiment-tag gr1 \
          --data-config fourier_gr1_arms_waist \
          --port 8000 \
          --host 0.0.0.0 \
          --denoising-steps 4 \
          --use-tensorrt \
          --trt-engine-path "gr00t_engine_fp16" \
          --vit-dtype fp16 \
          --llm-dtype fp16 \
          --dit-dtype fp16
    elif [[ "$zmq" == "true" ]]; then
        # ZMQ server
        echo "Starting ZMQ server..."
        python scripts/inference_base.py \
          --server \
          --model-path "/home/jetson/Desktop/project/model/gr00t_weights/GR00T-N1.5-3B" \
          --embodiment-tag gr1 \
          --data-config fourier_gr1_arms_waist \
          --port 5555 \
          --denoising-steps 4 \
          --use-tensorrt \
          --trt-engine-path "gr00t_engine_fp16" \
          --vit-dtype fp16 \
          --llm-dtype fp16 \
          --dit-dtype fp16
    fi
fi

if [[ "$client" == "true" ]]; then
    if [[ "$http" == "true" ]]; then
        # HTTP client
        echo "Starting HTTP client..."
        python scripts/inference_base.py \
          --client \
          --http-server \
          --host localhost \
          --port 8000 \
          --denoising-steps 4
    elif [[ "$zmq" == "true" ]]; then
        # ZMQ client
        echo "Starting ZMQ client..."
        python scripts/inference_base.py \
          --client \
          --host localhost \
          --port 5555 \
          --denoising-steps 4
    fi
fi

# Keep the script running until interrupted
while true; do
    sleep 1
done
