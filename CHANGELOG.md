# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-02

### Added
- Initial release of pupil_track
- U-Net based pupil segmentation model
- PyQt5 GUI for interactive annotation and processing
- Batch processing workflow with ROI definition and detection phases
- Support for alternative detection methods (integrodifferential operator, starburst)
- Automatic and manual ROI detection
- Post-processing pipeline with temporal smoothing, outlier removal, and gap filling
- Export to CSV, binary masks (.npy), time-series plots, and annotated videos
- Reproducible JSON configuration for each run
- Command-line interface for training and inference
- Pre-trained model with 0.8586 Dice coefficient
- Comprehensive documentation and installation guide

[0.1.0]: https://github.com/ruixli/pupil_track/releases/tag/v0.1.0
