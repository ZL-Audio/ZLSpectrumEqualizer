// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

#include "zlp_definitions.hpp"
#include "../dsp/filter/ideal_filter/ideal.hpp"
#include "../dsp/filter/empty_filter/empty.hpp"

#include "../dsp/splitter/inplace_ms_splitter.hpp"

namespace zlp {
    class Controller final : private juce::AsyncUpdater {
    private:
        struct ChannelFFTData {
            size_t dyn_left_{}, dyn_length_{};

            size_t side_left_{}, side_length_{};

            std::vector<size_t> dyn_not_off_bands_{};

            std::vector<size_t> dyn_on_bands_{};

            bool not_off_{false}, dyn_not_off{false};
        };

    public:
        static constexpr size_t kFilterSize = 16;
        static constexpr size_t kAnalyzerPointNum = 251;

        enum class LRMS {
            kStereo, kL, kR, kM, kS
        };

        explicit Controller(juce::AudioProcessor& p);

        void prepare(double sample_rate, size_t max_num_samples);

        template <bool bypass = false, bool ext_side = false>
        void processBuffer(std::array<float*, 2> main_buffer,
                           std::array<float*, 2> side_buffer,
                           size_t num_samples);

    private:
        static constexpr float kSqrt2Over2 = static_cast<float>(
            0.7071067811865475244008443621048490392848359376884740365883398690);

        juce::AudioProcessor& p_ref_;


        void handleAsyncUpdate() override;

        void prepareBuffer();
    };
}
