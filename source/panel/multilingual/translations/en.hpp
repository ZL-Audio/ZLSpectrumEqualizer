// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

// This file is also dual licensed under the Apache License, Version 2.0. You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>

#pragma once

#include <array>

namespace zlpanel::multilingual::en {
    static constexpr std::array kTexts = {
        "Release: bypass the band.",
        "Press: solo the band.",
        "Choose the filter type.",
        "Choose the filter slope. A higher slope makes the filter’s response curve change more steeply.",
        "Choose the stereo mode.",
        "Control the frequency.",
        "Control the base gain and the target gain.",
        "Control the quality factor. A larger Q value makes the bandwidth narrower.",
        "Press: turn on the dynamic behavior.",
        "Click: turn off the band.",

        "Release: bypass the dynamic processing.",
        "Choose the dynamic mode.",
        "Dynamic processing uses a static threshold.",
        "Dynamic processing uses a dynamic threshold related to band side-chain loudness",
        "Dynamic processing uses a dynamic threshold related to total side-chain loudness",
        "Control the static threshold.",
        "Control the band relative threshold.",
        "Control the total relative threshold.",
        "Control the knee width of the dynamic processing.",
        "Control the attack time of the dynamic processing.",
        "Control the release time of the dynamic processing.",

        "Release: bypass the plugin.",
        "Press: use the external side-chain.\nRelease: use the internal side-chain.",

        "Choose the spectrum processing resolution.",
        "Choose the spectrum processing side-chain smooth type.",
        "Choose the spectrum processing side-chain smooth value.",
        "Control the spectrum processing side-chain tilt slope.",
        "Control the spectrum processing side-chain attack skew. A higher skew makes the attack faster at high frequency.",
        "Control the spectrum processing side-chain release skew. A higher skew makes the release faster at high frequency.",
        "Control the spectrum gate of relative loudness.",

        "Control the additional output gain.",
        "Control the scale of all filters’ base & target gain.",
        "Press: turn on Static Gain Compensation. SGC is inaccurate, but does not affect dynamics.",
        "Press: start to measure the integrated loudness of the input signal and the output signal\nRelease: update the Output Gain to the difference between the two loudness values.",

        "Press: turn on input signal spectrum analyzer",
        "Press: turn on output signal spectrum analyzer",
        "Press: turn on side-chain signal spectrum analyzer",
        "Choose the decay speed of spectrum analyzers.",
        "Choose the tilt slope of spectrum analyzers.",
        "Choose the smooth value of spectrum analyzers.",
        "Choose the smooth type of spectrum analyzers.",
        "Press: turn on the freezing feature. Hover the mouse over the analyzer to freeze the spectrum.",
        "Press: turn on the collision detection.",
        "Control the collision detection strength.",

        "Double-click: open the UI settings.",
    };
}
