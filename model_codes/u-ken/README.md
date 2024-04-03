# U-KEN Do It

A U-Knowledge-Embedding-Network for universal lesion segmentation. I'm not 100% sure if this is how it's meant to work but it has been worth a shot.

Make sure to download `ULS23_DeepLesion3D` data from https://zenodo.org/records/10035161 and https://github.com/MJJdG/ULS23 and put it into a `datasets` folder.

I have the folder set up like

```bash
.
├── datasets
│   └──ULS23_DeepLesion3D
│           ├── categories
│           ├── images
|           └── labels
├── models
│   └──uken.py
├── snapshots
└── *
```

Based on Qiu, Y., Xu, J. (2022). Delving into Universal Lesion Segmentation: Method, Dataset, and Benchmark. In: Avidan, S., Brostow, G., Cissé, M., Farinella, G.M., Hassner, T. (eds) Computer Vision – ECCV 2022. ECCV 2022. Lecture Notes in Computer Science, vol 13668. Springer, Cham. https://doi.org/10.1007/978-3-031-20074-8_28
