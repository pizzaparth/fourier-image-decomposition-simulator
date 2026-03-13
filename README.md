# Fourier Image Decomposition Simulator

#### A Python implementation of Fourier epicycle drawing. The program extracts image contours, converts them into a complex signal, computes the Discrete Fourier Transform, and animates the reconstruction using rotating vectors.

## Setup
### 1. Install Anaconda
#### Download the installer from:
>https://anaconda.org/
##### Choose the installer of your OS
##### Windows Installtion Method (Recommended) ------
    >>> 1. Donwload the 64-bit graphical installer.
    >>> 2. Run the .exe file.
    >>> 3. Keep default settings.
    >>> 4. After installation open the Anaconda Prompt.

### 2. Verfiy the installtion
```bash
conda --version
```

### 3. Setup the virtual encironment
```bash
conda create -n fourier-env python=3.11
```
### 4. Activate the environment
```bash
conda activate fourier-env
```
### 5. Install packages inside the environment
```bash
pip install numpy matplotlib opencv-python scipy
```
#
#
## Execute/Modify the code
### 1. Go to the directory
```bash
cd go/to/the/desired/directory
```
### 2. Inside the Anaconda Prompt
```bash
cd go/to/the/desired/directory
conda activate fourier-env
```
### 3. Run main.py
#### In Anaconda Prompt
```bash
python main.py --image /images/alpha_h.jpg --n_terms 100 --n_frames 300
```
### 4. Save the .mp4 file
#### In Anaconda Prompt
```bash
python main.py --image /images/alpha_h.jpg --n_terms 100 --n_frames 300 --save output.mp4
```

