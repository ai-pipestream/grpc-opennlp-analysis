# Runs the OpenNLP analysis service as a GraalVM native binary; no JRE in the
# final image. Stage 1 builds the binary with Gradle on a GraalVM CE 25 image
# (native-image included); stage 2 is a thin glibc layer over the executable
# (the binary links only glibc + libz).
#
#   docker build -t grpc-opennlp-analysis .
#   docker run --rm -p 50051:50051 grpc-opennlp-analysis
#   docker run --rm -p 50051:50051 -e OPENNLP_EMBEDDINGS_DIR=/models/minishlab \
#     -v /path/to/models:/models grpc-opennlp-analysis

FROM ghcr.io/graalvm/graalvm-community:25 AS build
WORKDIR /workspace
COPY . .
# The container's GraalVM JDK is the only toolchain on the image, so the
# vendor-neutral toolchain spec resolves to it without further pinning.
RUN ./gradlew nativeCompile --no-daemon

FROM debian:trixie-slim

COPY --from=build /workspace/build/native/nativeCompile/grpc-opennlp-analysis \
    /usr/local/bin/grpc-opennlp-analysis

# A compromise of the process should not own the container: run as a dedicated
# non-root user. The service itself writes nothing.
RUN useradd --system --uid 10001 --create-home --home-dir /home/analysis analysis \
    && chmod a+rx /usr/local/bin/grpc-opennlp-analysis
USER 10001
ENV HOME=/home/analysis
WORKDIR /home/analysis

EXPOSE 50051
ENTRYPOINT ["/usr/local/bin/grpc-opennlp-analysis"]
