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
    inline constexpr std::array kTexts = {
        // band basic controls
        "Toggles band bypass.",
        "Toggles band solo audition.",
        "Selects the filter type.",
        "Selects the filter slope. A higher slope produces a steeper frequency response curve.",
        "Selects the stereo channel mode.",
        "Adjusts the center or cutoff frequency.",
        "Adjusts the base gain and dynamic target gain.",
        "Adjusts the Q factor. A higher Q narrows the bandwidth.",
        "Toggles dynamic EQ behavior for this band.",
        "Turns off this band.",
        "Selects the previous band.",
        "Selects the next band.",
        "Toggles pass scaling for Low Pass and High Pass filters.",
        "Adjusts the magnitude scale of the pass filter.",
        // band dynamic controls
        "Toggles dynamic processing bypass for this band.",
        "Toggles output of the spectrum dynamic delta signal.",
        "Selects the spectrum dynamic mode.",
        "Uses a static threshold for dynamic processing.",
        "Uses a dynamic threshold based on band side-chain loudness.",
        "Uses a dynamic threshold based on total side-chain loudness.",
        "Adjusts the static threshold.",
        "Adjusts the band relative threshold.",
        "Adjusts the total relative threshold.",
        "Adjusts the dynamic processing knee width.",
        "Adjusts the dynamic processing attack time.",
        "Adjusts the dynamic processing release time.",
        // top controls
        "Toggles the UI settings panel.",
        "Toggles the preset browser.",
        "Toggles the analyzer settings panel.",
        "Toggles the output settings panel.",
        "Toggles plugin bypass.",
        "Toggles the external side-chain input.",
        // preset browser
        "Opens the presets folder in the system file manager.",
        "Deletes the selected preset group.",
        "Deletes the selected preset.",
        "Closes the preset browser.",
        // spectrum controls
        "Toggles the spectrum processing settings panel.",
        "Selects the spectrum processing resolution.",
        "Selects the side-chain frequency smoothing mode for spectrum processing.",
        "Adjusts the side-chain frequency smoothing amount for spectrum processing.",
        "Adjusts the side-chain tilt slope for spectrum processing.",
        "Adjusts the side-chain attack skew for spectrum processing. A higher skew speeds up the attack at high frequencies.",
        "Adjusts the side-chain release skew for spectrum processing. A higher skew speeds up the release at high frequencies.",
        "Adjusts the relative loudness gate for spectrum processing.",
        // output controls
        "Adjusts the additional output gain.",
        "Adjusts the overall scale of all filters' base and target gain.",
        "Toggles Static Gain Compensation (SGC). SGC is approximate, but does not affect dynamics.",
        "Toggles loudness gain matching. Enabling measures input and output loudness; disabling sets Output Gain to their difference.",
        // analyzer controls
        "Toggles display of the input signal spectrum.",
        "Toggles display of the output signal spectrum.",
        "Toggles display of the side-chain signal spectrum.",
        "Selects the decay speed of the spectrum analyzer.",
        "Selects the tilt slope of the spectrum analyzer.",
        "Selects the frequency smoothing width of the spectrum analyzer.",
        "Selects the frequency smoothing mode.\nOCT: constant fractional-octave smoothing.\nERB: equivalent rectangular bandwidth smoothing.",
        "Toggles spectrum freezing when hovering the pointer over the analyzer.",
        "Selects the stereo channel mode displayed by the spectrum analyzer.",
        "Toggles frequency collision detection.",
        "Adjusts the collision detection strength.",
        "Toggles visibility of the numeric value readout.",
        "Toggles visibility of the level meter panel.",
        // curve scale & piano roll
        "Selects the decibel display range for the EQ response curve.",
        "Selects the upper decibel ceiling for the spectrum analyzer display.",
        "Selects the minimum decibel floor for the spectrum analyzer display.",
        "Toggles visibility of the piano roll.",
        " "
    };
}
