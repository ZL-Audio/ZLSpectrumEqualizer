# ZL Spectrum Equalizer

LICENSE and CODE are available at [https://github.com/ZL-Audio/ZLSpectrumEqualizer](https://github.com/ZL-Audio/ZLSpectrumEqualizer)

# Changelog

## 0.1.0

BREAKING CHANGES

- fix incorrect fresh rate reported by internal refresh handler
    - you may notice the FFT analyzer decays in a different speed after the fix
    - you may need to re-adjust FFT `Speed` in UI settings

Bug fixes

- fix preset folder permission issue on macOS
- fix floating window position when dynamic is ON
- fix solo shortcuts regarding right-click

New Features

- add `Flat Gain` filter type
- add gain adjustment during solo
    - to fix the gain as zero, press `Ctrl/Command` during right-click dragging

Other Changes

- change filter close button to trash icon
- improve FFT analyzer precision

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