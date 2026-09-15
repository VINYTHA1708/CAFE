# CAFE — Citations and Attribution

---

## FaceForensics++

The dataset used in this project is FaceForensics++ (C23 compression level).

```bibtex
@inproceedings{rossler2019faceforensics++,
  title     = {FaceForensics++: Learning to Detect Manipulated Facial Images},
  author    = {Rossler, Andreas and Cozzolino, Davide and Verdoliva, Luisa
               and Riess, Christian and Thies, Justus and Niessner, Matthias},
  booktitle = {Proceedings of the IEEE/CVF International Conference on
               Computer Vision (ICCV)},
  year      = {2019}
}
```

**Licence and redistribution.** The FaceForensics++ dataset is provided for
research purposes only under the terms of the official access agreement. The
dataset must NOT be redistributed. It is excluded from this repository via
`.gitignore`. To obtain access, submit the official request form:

> https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform

---

## Base detector - EfficientNet-B4 FF++

The frozen deepfake detector used by CAFE is the EfficientNet-B4 implementation
and pretrained FF++ checkpoint from the upstream
`polimi-ispl/icpr2020dfdc` repository.

The associated paper is:

> N. Bonettini, E. D. Cannas, S. Mandelli, L. Bondi, P. Bestagini and S. Tubaro,
> "Video Face Manipulation Detection Through Ensemble of CNNs,"
> 2020 25th International Conference on Pattern Recognition (ICPR),
> 2021, pp. 5012-5019, doi: 10.1109/ICPR48806.2021.9412711.

```bibtex
@INPROCEEDINGS{9412711,
  author={Bonettini, Nicolò and Cannas, Edoardo Daniele and Mandelli, Sara and Bondi, Luca and Bestagini, Paolo and Tubaro, Stefano},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
  title={Video Face Manipulation Detection Through Ensemble of CNNs},
  year={2021},
  pages={5012-5019},
  doi={10.1109/ICPR48806.2021.9412711}
}
```

Upstream repository:

https://github.com/polimi-ispl/icpr2020dfdc

The checkpoint and upstream source code are not redistributed in this
repository. Users are responsible for complying with the applicable licence
and usage conditions.

---

## BlazeFace

Face detection during preprocessing uses BlazeFace from the upstream
`detector_reference/blazeface/` directory.

```bibtex
@article{bazarevsky2019blazeface,
  title   = {BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs},
  author  = {Bazarevsky, Valentin and Kartynnik, Yury and Vakunov, Andrey
             and Raveendran, Karthik and Grundmann, Matthias},
  journal = {arXiv preprint arXiv:1907.05047},
  year    = {2019}
}
```

---

## MediaPipe Face Landmarker

Landmark extraction uses the MediaPipe FaceLandmarker task model
(`face_landmarker.task`).

> Lugaresi, C. et al. (2019). MediaPipe: A Framework for Building Perception
> Pipelines. arXiv:1906.08172.

Model download:
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

The model is subject to the Apache 2.0 licence from Google LLC. It is not
redistributed in this repository.
