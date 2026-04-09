# TQBridge Decode Server — multi-arch container
#
# Build multi-platform:
#   docker buildx build --platform linux/amd64,linux/arm64 -t chronara/tqbridge-server:latest deploy/
#
# Run:
#   docker run -p 9473:9473 chronara/tqbridge-server
#   docker run -p 9473:9473 chronara/tqbridge-server --head-dim 128 --seed 42
#   docker run --gpus all -p 9473:9473 chronara/tqbridge-server  # with CUDA

FROM debian:bookworm-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY llama-cpp-turboquant/src/tqbridge.c llama-cpp-turboquant/src/tqbridge.h \
     llama-cpp-turboquant/src/tqbridge_net.c llama-cpp-turboquant/src/tqbridge_net.h ./
COPY llama-cpp-turboquant/tools/tqbridge-server.c ./

RUN sed -i 's|"../src/tqbridge.h"|"tqbridge.h"|' tqbridge-server.c && \
    sed -i 's|"../src/tqbridge_net.h"|"tqbridge_net.h"|' tqbridge-server.c && \
    cc -O2 -static -o tqbridge-server tqbridge-server.c tqbridge.c tqbridge_net.c -lm

# Minimal runtime — static binary, no OS needed
FROM scratch
COPY --from=build /build/tqbridge-server /tqbridge-server
EXPOSE 9473
ENTRYPOINT ["/tqbridge-server"]
CMD ["--port", "9473"]
