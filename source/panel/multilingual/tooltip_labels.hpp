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

        kBandDynamicBypass,
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

        kBypass,
        kExternalSideChain,

        kSpecResolution,
        kSpecSmoothType,
        kSpecSmoothValue,
        kSpecTilt,
        kSpecAttackSkew,
        kSpecReleaseSkew,
        kSpecGate,

        kOutputGain,
        kGainScale,
        kStaticGC,
        kLoudnessGC,

        kFFTPre,
        kFFTPost,
        kFFTSide,
        kFFTSpeed,
        kFFTSlope,
        kFFTSmoothValue,
        kFFTSmoothType,
        kFFTFreeze,
        kFFTCollision,
        kFFTCollisionStrength,

        kPluginLogo,
    };
}
