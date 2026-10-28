# Contributing

Thanks for helping out. Bug reports, fixes and small focused features are all welcome.

## Before you start

For anything bigger than a small fix, open an issue first so we can agree on the approach.

## Development

You need Go (see `go.mod` for the minimum version).

```bash
go build ./...
go test -race ./...
```

The tests run entirely offline against in-process fakes, so no real services are needed.

Before sending a pull request, make sure these pass (CI runs the same checks):

```bash
gofmt -l .          # should print nothing
go vet ./...
staticcheck ./...   # https://staticcheck.dev
go test -race ./...
```

## Pull requests

- Keep each pull request to one change, with tests for new behaviour.
- Follow standard Go style; `gofmt` decides formatting.
- Add a line under "Unreleased" in `CHANGELOG.md` for user-visible changes.

## Reporting security issues

Please don't open a public issue. See [SECURITY.md](SECURITY.md).
