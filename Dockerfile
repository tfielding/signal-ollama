# Build natively on the runner and cross-compile for each target platform (no QEMU needed).
FROM --platform=$BUILDPLATFORM golang:1.24-alpine AS build
ARG TARGETOS TARGETARCH
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH go build -trimpath -ldflags="-s -w" -o /signal-ollama .

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /signal-ollama /signal-ollama
COPY LICENSE THIRD_PARTY_LICENSES /licenses/
VOLUME /data
ENTRYPOINT ["/signal-ollama"]
