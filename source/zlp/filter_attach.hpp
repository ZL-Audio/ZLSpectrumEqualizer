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
    class FilterAttach final : private juce::AudioProcessorValueTreeState::Listener {
    public:
        explicit FilterAttach(juce::AudioProcessor& processor,
                              juce::AudioProcessorValueTreeState& parameters,
                              Controller& controller,
                              size_t idx);

        ~FilterAttach() override;

    private:
        juce::AudioProcessorValueTreeState& parameters_;
        Controller& controller_;
        size_t idx_;
        zldsp::filter::Empty& empty_;
        std::atomic<float>& scale_;
        std::atomic<float>& gain_;
        zlchore::thread::Notifier& empty_update_flag_;
        zlchore::thread::Notifier& spec_update_flag_;
        zlchore::thread::Notifier& whole_update_flag_;

        static constexpr std::array kIDs{
            PFilterStatus::kID, PFilterType::kID, POrder::kID, PLRMode::kID,
            PFreq::kID, PGain::kID, PQ::kID,
        };

        void parameterChanged(const juce::String& parameter_ID, float value) override;

        void signal();
    };
}
