// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#pragma once

namespace zlpanel::multilingual {
    enum TooltipLabel {
        // band basic controls
        kBandBypass,
        kBandSolo,
        kBandType,
        kBandSlope,
        kBandStereoMode,
        kBandFreq,
        kBandGain,
        kBandQ,
        kBandDynamic,
        kBandOff,
        kBandPrev,
        kBandNext,
        kBandPassScaleOn,
        kBandPassScale,

        // band dynamic controls
        kBandDynamicBypass,
        kBandDynamicDelta,
        kBandDynamicMode,
        kBandDynamicAbs,
        kBandDynamicBand,
        kBandDynamicRelative,
        kBandDynamicAbsThreshold,
        kBandDynamicBandThreshold,
        kBandDynamicRelativeThreshold,
        kBandDynamicKnee,
        kBandDynamicAttack,
        kBandDynamicRelease,

        // top controls
        kPluginLogo,
        kPresetOpen,
        kAnalyzerOpen,
        kOutputOpen,
        kBypass,
        kExternalSideChain,

        // preset browser
        kPresetFolder,
        kPresetGroupDelete,
        kPresetDelete,
        kPresetClose,

        // spectrum controls
        kSpecOpen,
        kSpecResolution,
        kSpecSmoothType,
        kSpecSmoothValue,
        kSpecTilt,
        kSpecAttackSkew,
        kSpecReleaseSkew,
        kSpecGate,

        // output controls
        kOutputGain,
        kGainScale,
        kStaticGC,
        kLoudnessGC,

        // analyzer controls
        kFFTPre,
        kFFTPost,
        kFFTSide,
        kFFTSpeed,
        kFFTSlope,
        kFFTSmoothValue,
        kFFTSmoothType,
        kFFTFreeze,
        kFFTStereo,
        kFFTCollision,
        kFFTCollisionStrength,
        kFFTValue,
        kFFTMeter,

        // curve scale & piano roll
        kScaleEQMaxDB,
        kScaleFFTTopDB,
        kScaleFFTMinDB,
        kPianoRoll,

        kLabelNum
    };
}
