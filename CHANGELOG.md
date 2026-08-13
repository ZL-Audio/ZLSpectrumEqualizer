# ZL Spectrum Equalizer

LICENSE and CODE are available at [https://github.com/ZL-Audio/ZLSpectrumEqualizer](https://github.com/ZL-Audio/ZLSpectrumEqualizer)

# Changelog

## 0.0.3

Bug fixes

- fix unresponsive Linux UI
- fix incorrect static gain compensation when stereo mode changes
- fix external side-chain button may not get updated display
- fix dynamic curve display beyond Nyquist
- fix potential race condition for FFT collision colour
- fix potential redundant value notification of sliders/buttons/comboboxes
- fix potential lagging caused by UI resizing

New Features

- add built-in preset manager (very early stage)

Other Changes

- adjust UI setting panel
  - remove import/export functions
  - add reveal folder button
- adjust combobox UI


## 0.0.2

BREAKING CHANGES

- fix the `Relative` dynamic mode
- use separate sensitivity controls for sliders/draggers

Bug fixes

- fix slider value editor display

New Features

- add dynamic delta (per band)

Improvements

- improve DSP performance (slightly)


## 0.0.1

- first version