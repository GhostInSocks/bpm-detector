# BPM Metronome

An automatic tempo (BPM) detector and synchronized metronome overlay tool designed for dance practice. This project is implemented in Python using fundamental digital signal processing (DSP) mathematics and NumPy.

## Key Features
* **Low-Pass Filtering:** Isolates bass frequencies and kick drum hits (below 150 Hz) using a 4th-order Butterworth filter to remove masking "noise" like vocals and melodies.
* **Manual STFT:** Short-Time Fourier Transform implemented manually using `for` loops combined with a Hanning window function for spectral smoothing.
* **Spectral Flux:** Computes a novelty envelope (energy onset detection) to precisely capture the exact onset moments of musical beats.
* **Autocorrelation:** Automatically estimates the global tempo of the song within a standard 60 to 200 BPM range.
* **Dynamic Peak Picking:** A phase-locked tracking loop that dynamically adjusts to slight tempo variations in live instrumentation, preventing the metronome from drifting over time.
* **Adaptive Mixing:** Automatically attenuates the original audio track by 50% before mixing, ensuring the synthesized metronome click stands out clearly as a distinct rhythmic guide for dancers.

## System Requirements
To run this project, you need Python 3.7 or higher along with the following standard libraries for data science and audio processing:
* `numpy` — for mathematical operations and Fourier matrices
* `scipy` — for designing the Butterworth filter
* `soundfile` — for reading and writing audio files
* `matplotlib` — for generating the final visual analysis charts

## Usage Instructions

### 1. Install Dependencies
Before running the script for the first time, open your terminal (or command prompt) and install all the required packages with a single command:
```bash
pip install numpy scipy soundfile matplotlib

```

### 2. Prepare Your Audio File

Place the audio track you want to analyze (e.g., in `.mp3` or `.wav` format) into the exact same project folder where your Python script is located.

### 3. Configure and Run the Script

Open the Python script (e.g., `main.py`) in any code editor or IDE. Scroll down to the bottom of the code to find the `if __name__ == "__main__":` block, and specify your audio file's name in the `vhodna_pesem` variable:

```python
if __name__ == "__main__":
    vhodna_pesem = "your_audio_file.mp3"  # Change this to your file's name
    izhodna_pesem = "rezultat_z_metronomom.wav"

```

Once updated, run the script from your terminal:

```bash
python main.py

```

### 4. Output Results

Once the algorithm finishes execution, two new files will automatically appear in your project folder:

1. `rezultat_z_metronomom.wav` — Your practice-ready audio file where the music is slightly attenuated, overlaid with clear and loud metronome clicks.
2. `bpm_analiza_graf.png` — A visual analysis chart illustrating the original audio wave, the isolated bass signal, and the final generated beat grid alongside the calculated BPM value.
