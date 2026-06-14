# Preset Assets

Place local assets under this directory before training or inference.

Expected layout:

```text
preset/
  models/
    stable-diffusion-2-1-base/
    focussr.pkl
  test_datasets/
  testfolder/
    test_SR_bicubic/
    test_HR/
  gt_all_path.txt
  gt_hq_path.txt
  gt_lsdir_path.txt
  gt_ffhq_path.txt
```

The `gt_*.txt` files are local image path lists and are ignored by Git. Use the
provided `*.example.txt` files as templates.
