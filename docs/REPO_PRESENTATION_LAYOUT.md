# Repo Presentation Layout

This note records where public-facing presentation assets should be placed so
they can be included in the GitHub release target and referenced from
`README.md`.

## Recommended Asset Locations

### Repository title

Use the full thesis title in the README:

```text
以擴散模型潛在空間為基礎之多編碼器條件擴散強健浮水印方法
```

The GitHub repository slug itself should still stay URL-friendly.

### Architecture figure

Recommended path:

```text
assets/figures/o256_stagee_overall_architecture.png
```

### Thesis or paper PDF

Recommended path:

```text
docs/thesis/thesis.pdf
```

Optional second file:

```text
docs/thesis/paper.pdf
```

## README Placement

The most portfolio-friendly README order is:

1. project title
2. one-sentence project summary
3. architecture figure
4. thesis / paper link
5. key results and repository guide

## Assembly Note

The release assembly script copies these paths into:

- the GitHub public target
- the lab handoff target

So once the files are placed in the source repository, re-running the assembly
script is enough to include them in both output folders.
