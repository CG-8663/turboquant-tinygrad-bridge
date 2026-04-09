# TQBridge Decode Server — multi-arch container
#
# Build:
#   docker build -t chronara/tqbridge-server deploy/
#
# Build multi-platform:
#   docker buildx build --platform linux/amd64,linux/arm64 \
#     -t chronara/tqbridge-server:latest --push deploy/
#
# Run:
#   docker run -p 9473:9473 chronara/tqbridge-server
#   docker run -p 9473:9473 chronara/tqbridge-server --head-dim 128 --seed 42

FROM debian:bookworm-slim AS build

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY src/ .

RUN cc -O2 -static -o tqbridge-server \
    tqbridge-server.c tqbridge.c tqbridge_net.c -lm

# Minimal runtime — static binary on empty image (~100KB)
FROM scratch
COPY --from=build /build/tqbridge-server /tqbridge-server

# Include CUDA kernel source for runtime compilation on any architecture
# The decode node compiles for the detected GPU (Pascal sm_61 → Blackwell sm_120)
COPY src/polar_quant.cu /kernels/polar_quant.cu

EXPOSE 9473
ENTRYPOINT ["/tqbridge-server"]
CMD ["--port", "9473"]
