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
#include "../dsp/filter/spec_dynamic/spec_response.hpp"
#include "../dsp/filter/spec_dynamic/spec_follower.hpp"
#include "../dsp/filter/spec_dynamic/spec_dynamic.hpp"
#include "../dsp/filter/spec_dynamic/spec_smoother.hpp"
#include "../dsp/splitter/inplace_ms_splitter.hpp"
#include "../dsp/fft/zldsp_fft_include.hpp"

#include "../chore/thread/notifier.hpp"

namespace zlp {
    namespace hn = hwy::HWY_NAMESPACE;

    template <typename T, std::size_t N, typename... Args, std::size_t... I>
    constexpr std::array<T, N> make_array_of_impl(std::index_sequence<I...>, Args&&... args) {
        return {{(static_cast<void>(I), T(std::forward<Args>(args)...))...}};
    }

    template <typename T, std::size_t N, typename... Args>
    constexpr std::array<T, N> make_array_of(Args&&... args) {
        return make_array_of_impl<T, N>(std::make_index_sequence<N>{}, std::forward<Args>(args)...);
    }

    class Controller final : private juce::AsyncUpdater {
    public:
        static constexpr size_t kFilterSize = 16;

        explicit Controller(juce::AudioProcessor& p);

        void prepare(double sample_rate, size_t max_num_samples);

        void process(bool is_bypass);

        template <bool has_stereo, bool has_l, bool has_r, bool has_m, bool has_s>
        void processImpl();

    private:
        static constexpr hn::ScalableTag<float> d;
        static constexpr size_t lanes = hn::MaxLanes(d);

        enum class SideStatus {
            kNotRequired, kLR, kMS, kLRMS
        };

        struct ChannelData {
            bool is_side_required{false};
            bool is_active{false};

            std::vector<size_t> bands{};
            zldsp::vector::aligned_vector<float> static_response;

            zldsp::vector::aligned_vector<float> fft_side_abs_sqr;
            zldsp::filter::SpecSmoother<float>::SmoothBounds smooth_bounds;

            size_t dynamic_start_idx{0}, dynamic_end_idx{0};
            std::vector<size_t> dynamic_bands{};
            zldsp::vector::aligned_vector<float> dynamic_response;
        };

        static constexpr float kSqrt2Over2 = static_cast<float>(
            0.7071067811865475244008443621048490392848359376884740365883398690);

        juce::AudioProcessor& p_ref_;

        std::array<zldsp::filter::Empty, kBandNum> emptys_{};
        std::array<zlchore::thread::Notifier, kBandNum> empty_update_flags_{};
        std::array<zldsp::filter::FilterParameters, kBandNum> filter_paras_{};
        // spectrum processing
        std::array<zldsp::filter::SpecResponse<float>, kBandNum> spec_response_
            = make_array_of<zldsp::filter::SpecResponse<float>, kBandNum>();
        std::array<zldsp::filter::SpecFollower<float>, kBandNum> spec_follower_
            = make_array_of<zldsp::filter::SpecFollower<float>, kBandNum>();
        std::array<zldsp::filter::SpecDynamic<float>, kBandNum> spec_dynamic_
            = make_array_of<zldsp::filter::SpecDynamic<float>, kBandNum>();
        zldsp::filter::SpecSmoother<float> spec_smoother_;
        // fft working space
        std::unique_ptr<zldsp::fft::RFFT<float>> fft_;
        size_t fft_order_ = 12;
        size_t fft_size_ = static_cast<size_t>(1) << fft_order_;
        size_t num_bin_ = fft_size_ / 2 + 1;
        size_t num_bin_effective_ = fft_size_ / 2;
        zldsp::vector::aligned_vector<float> window1_, window2_, window_bypass_;

        std::array<zldsp::vector::aligned_vector<float>, 4> input_fifos_, output_fifos_;
        std::array<zldsp::vector::aligned_vector<float>, 4> fft_ins_;
        std::array<zldsp::vector::aligned_vector<float>, 2> fft_out_reals_, fft_out_imags_;

        size_t dispatch_mask_{0};
        ChannelData stereo_data_, l_data_, r_data_, m_data_, s_data_;

        SideStatus side_status_{SideStatus::kNotRequired};

        void processSide();

        void processSideLR();

        void processSideMS();

        void processSideLRMS();

        void processDynamicBands(ChannelData& data);

        void processMain(bool is_bypass);

        void handleAsyncUpdate() override;
    };
}
