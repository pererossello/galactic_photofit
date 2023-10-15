# Photometric Decomposition of UGC09629

## Authors
Pere Rosselló

## Overview
This project revolves around the photometric decomposition of the galaxy UGC09629. It was initially developed as part of the subject "Física Extragaláctica" within an astrophysics program at ULL. The focus lies on unraveling the physical characteristics of UGC09629 through photometric analysis.

## Directory Structure
Brief description of the repository structure:

- `code/`: The main codebase. Contains Jupyter notebooks for fitting, plotting, and comparison, along with utility Python scripts.
- `data/`: Contains FITS files.
- `figures/`: Stored visualizations, including plots and comparison metrics.
- `results/`: Contains stored results, mainly as .nc files and .txt files detailing parameters, and posto-processed data.

## How to Run
- You'll need to have Python installed, along with some specific packages.
- Navigate to the `code/` directory for the core codebase.
- Run individual Jupyter notebooks for specific tasks like fitting or plotting.

## Special Mention
- `init_imports.py`: Initializes all imports.
- `utils.py` and `plot_utils.py`: Utility scripts to ease the process.

## Contributing
Feel free to fork and submit pull requests. For bugs and feature requests, please create an issue.

### Core Libraries Used in this Project

In this project, two essential Python libraries were integral to our analysis:

1. **pyimfit**: 
   - A powerful library tailored for the fitting of astronomical images.
   - [Official Documentation](https://pyimfit.readthedocs.io/)

2. **photutils**: 
   - A library that provides tools for detecting and performing photometry of astronomical sources.
   - [Official Documentation](https://photutils.readthedocs.io/)

