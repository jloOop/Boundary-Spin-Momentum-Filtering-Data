# Simple GitHub upload steps

This is the easiest graphical workflow.

## A. Create the repository

1. Go to GitHub.
2. Click **New repository**.
3. Repository name:

```text
Boundary-Spin-Momentum-Filtering-Data
```

4. Keep it private until the paper/release is ready, or public if you are ready.
5. Add the files from this package.

## B. Upload small files directly

Use GitHub website:

1. Open the repository.
2. Click **Add file** → **Upload files**.
3. Drag folders/files into the page.
4. Commit.

Upload directly:

```text
README.md
docs/*.md
python-scripts/*.py
notebooks/*.ipynb
small PNG files
small GIF files
CSV/JSON summary files
```

## C. Put large GIFs in a Release

For large GIFs:

1. Zip them locally, for example `omega_300_density_gifs.zip`.
2. Open the GitHub repository.
3. Click **Releases**.
4. Click **Draft a new release**.
5. Tag: `v1.0-gifs`.
6. Upload the ZIP files as assets.
7. Publish the release.
8. Copy the download links into

```text
follow-up-results-3D/3D-results/TimeEvolution-WaveFunction-Gifs/Links-to-Gifs.md
```

## D. Recommended terminal version

```bash
git clone https://github.com/jloOop/Boundary-Spin-Momentum-Filtering-Data.git
cd Boundary-Spin-Momentum-Filtering-Data

# copy files/folders from this package here

git add .
git commit -m "Add Gaussian spinor-ABC follow-up technical material"
git push
```
