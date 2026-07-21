// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include "controller.hpp"
#include "juce_helper/para_updater.hpp"

namespace zlp {
    class ChoreAttach final : private juce::AudioProcessorValueTreeState::Listener {
    public:
        explicit ChoreAttach(juce::AudioProcessor& processor,
                             juce::AudioProcessorValueTreeState& parameters,
                             Controller& controller);

        ~ChoreAttach() override;

    private:
        juce::AudioProcessorValueTreeState& parameters_;
        Controller& controller_;

        static constexpr std::array kIDs{
            PExtSide::kID, PFFTResolution::kID,
            PSpecSmooth::kID, PSpecSmoothType::kID, PSpecTilt::kID,
            PSpecSkewAttack::kID, PSpecSkewRelease::kID,
            POutputGain::kID, PStaticGain::kID
        };

        void parameterChanged(const juce::String& parameter_ID, float value) override;
    };
}
