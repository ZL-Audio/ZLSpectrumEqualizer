// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#include "filter_attach.hpp"

namespace zlp {
    FilterAttach::FilterAttach(juce::AudioProcessor&,
                               juce::AudioProcessorValueTreeState& parameters,
                               Controller& controller, const size_t idx) :
        parameters_(parameters),
        controller_(controller),
        idx_(idx),
        empty_(controller.getEmptyFilters()[idx]),
        scale_(*parameters.getRawParameterValue(PGainScale::kID)),
        gain_(*parameters.getRawParameterValue(PGain::kID + std::to_string(idx))),
        empty_update_flag_(controller.getEmptyUpdateFlags()[idx]),
        spec_update_flag_(controller.getSpecResponseUpdateFlag()),
        whole_update_flag_(controller.getUpdateFlag()) {
        for (size_t i = 0; i < kIDs.size(); ++i) {
            const auto ID = kIDs[i] + std::to_string(idx_);
            parameters_.addParameterListener(ID, this);
            parameterChanged(ID, parameters.getRawParameterValue(ID)->load(std::memory_order::relaxed));
        }
        parameters_.addParameterListener(PGainScale::kID, this);
    }

    FilterAttach::~FilterAttach() {
        for (size_t i = 0; i < kIDs.size(); ++i) {
            parameters_.removeParameterListener(kIDs[i] + std::to_string(idx_), this);
        }
        parameters_.removeParameterListener(PGainScale::kID, this);
    }

    void FilterAttach::parameterChanged(const juce::String& parameter_ID, const float value) {
        if (parameter_ID.startsWith(PFilterStatus::kID)) {
            controller_.setFilterStatus(idx_, static_cast<FilterStatus>(std::round(value)));
        } else if (parameter_ID.startsWith(PFilterType::kID)) {
            empty_.setFilterType(static_cast<zldsp::filter::FilterType>(std::round(value)));
            signal();
        } else if (parameter_ID.startsWith(POrder::kID)) {
            empty_.setOrder(POrder::kOrderArray[static_cast<size_t>(std::round(value))]);
            signal();
        } else if (parameter_ID.startsWith(PLRMode::kID)) {
            controller_.setLRMS(idx_, static_cast<FilterStereo>(std::round(value)));
        } else if (parameter_ID.startsWith(PFreq::kID)) {
            empty_.setFreq(value);
            signal();
        } else if (parameter_ID.startsWith(PGain::kID)) {
            empty_.setGain(std::clamp(value * (scale_.load(std::memory_order::relaxed) / 100.f), -30.f, 30.f));
            signal();
        } else if (parameter_ID.startsWith(PQ::kID)) {
            empty_.setQ(value);
            signal();
        } else if (parameter_ID.startsWith(PGainScale::kID)) {
            empty_.setGain(std::clamp(gain_.load(std::memory_order::relaxed) * (value / 100.f), -30.f, 30.f));
            signal();
        }
    }

    void FilterAttach::signal() {
        empty_update_flag_.signal();
        spec_update_flag_.signal();
        whole_update_flag_.signal();
    }
}
