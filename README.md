# x86_superopt

[![Release](https://img.shields.io/github/v/release/kostyk348/x86_superopt?sort=semver&color=blue)](https://github.com/kostyk348/x86_superopt/releases)
[![Last commit](https://img.shields.io/github/last-commit/kostyk348/x86_superopt?branch=main)](https://github.com/kostyk348/x86_superopt/commits/main)

Same-ISA x86_64 binary super-optimizer: SSA lifting (sdre) + AVX2 vectorization + branchless

## Status

Early-stage snapshot: the repository is public and tagged, the tree is the source of truth. Issues and pull requests are welcome.

## Build

```bash
cargo build --release
```

## Layout

- `src/` — src

## Files

- `.gitignore`
- `Cargo.lock`
- `Cargo.toml`
