// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#include "filter_dynamic_attach.hpp"

namespace zlp {
    FilterDynamicAttach::FilterDynamicAttach(juce::AudioProcessor&,
                                             juce::AudioProcessorValueTreeState& parameters,
                                             Controller& controller, size_t idx) :
        parameters_(parameters),
        controller_(controller),
        idx_(idx) {
        for (size_t i = 0; i < kIDs.size(); ++i) {
            const auto ID = kIDs[i] + std::to_string(idx_);
            parameters_.addParameterListener(ID, this);
            parameterChanged(ID, parameters.getRawParameterValue(ID)->load(std::memory_order::relaxed));
        }
    }

    FilterDynamicAttach::~FilterDynamicAttach() {
        for (size_t i = 0; i < kIDs.size(); ++i) {
            parameters_.removeParameterListener(kIDs[i] + std::to_string(idx_), this);
        }
    }

    void FilterDynamicAttach::parameterChanged(const juce::String& parameter_ID, const float value) {
        if (parameter_ID.startsWith(PDynamicON::kID)) {
            controller_.setDynamicON(idx_, value > .5f);
        } else if (parameter_ID.startsWith(PDynamicBypass::kID)) {
            controller_.setDynamicBypass(idx_, value > .5f);
        } else if (parameter_ID.startsWith(PDynamicMode::kID)) {
            controller_.setDynamicMode(idx_, static_cast<DynamicMode>(std::round(value)));
        } else if (parameter_ID.startsWith(PThresholdAbs::kID)) {
            controller_.setSpecThresholdAbs(idx_, value);
        } else if (parameter_ID.startsWith(PThresholdBand::kID)) {
            controller_.setSpecThresholdBand(idx_, value);
        } else if (parameter_ID.startsWith(PThresholdRel::kID)) {
            controller_.setSpecThresholdRel(idx_, value);
        } else if (parameter_ID.startsWith(PKneeW::kID)) {
            controller_.setSpecKnee(idx_, value);
        } else if (parameter_ID.startsWith(PAttack::kID)) {
            controller_.setSpecAttack(idx_, value);
        } else if (parameter_ID.startsWith(PRelease::kID)) {
            controller_.setSpecRelease(idx_, value);
        }
    }
}
