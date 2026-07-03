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

        void process(const std::array<float*, 4>& buffer, size_t num_samples, bool is_bypass);

        template <bool has_stereo, bool has_l, bool has_r, bool has_m, bool has_s>
        void processMainImpl(bool perform_fft);

        void setFilterStatus(const size_t idx, const FilterStatus filter_status) {
            a_filter_status_[idx].store(filter_status, std::memory_order::relaxed);
            to_update_status_.signal();
            to_update_.signal();
        }

        void setLRMS(const size_t idx, const FilterStereo filter_stereo) {
            a_lrms_[idx].store(filter_stereo, std::memory_order::relaxed);
            to_update_lrms_.signal();
            to_update_.signal();
        }

        void setExtSide(bool is_ext_side) {
            a_is_ext_side_.store(is_ext_side, std::memory_order::relaxed);
            to_update_.signal();
        }

        auto& getEmptyFilters() {
            return emptys_;
        }

        auto& getEmptyUpdateFlags() {
            return empty_update_flags_;
        }

        auto& getUpdateFlag() {
            return to_update_;
        }

        bool isSideRequired() const {
            return side_status_ != SideStatus::kNotRequired;
        }

    private:
        enum class SideStatus {
            kNotRequired, kLR, kMS, kLRMS
        };

        static constexpr hn::ScalableTag<float> d;
        static constexpr size_t lanes = hn::MaxLanes(d);

        struct ChannelData {
            bool is_active{false};

            std::vector<size_t> bands{};
            zldsp::vector::aligned_vector<float> static_response;

            zldsp::vector::aligned_vector<float> fft_side_abs_sqr;
            zldsp::filter::SpecSmoother<float>::SmoothBounds smooth_bounds;

            size_t dynamic_start_idx{0}, dynamic_end_idx{0};
            std::vector<size_t> dynamic_bands{};
            zldsp::vector::aligned_vector<float> dynamic_response;
        };

        juce::AudioProcessor& p_ref_;
        zlchore::thread::Notifier to_update_{false};
        // fft resolution
        std::atomic<FFTResolution> a_fft_resolution_{FFTResolution::kMedium};
        zlchore::thread::Notifier resolution_{false};
        // filter status
        std::array<std::atomic<FilterStatus>, kBandNum> a_filter_status_{};
        std::array<FilterStatus, kBandNum> filter_status_{};
        std::vector<size_t> not_off_total_{};
        zlchore::thread::Notifier to_update_status_{false};
        // filter l/r/m/s
        std::array<std::atomic<FilterStereo>, kBandNum> a_lrms_{};
        std::array<FilterStereo, kBandNum> lrms_{};
        zlchore::thread::Notifier to_update_lrms_{false};
        // empty filters for holding atomic parameters
        std::array<zldsp::filter::Empty, kBandNum> emptys_{};
        std::array<zlchore::thread::Notifier, kBandNum> empty_update_flags_{};
        std::array<zldsp::filter::FilterParameters, kBandNum> filter_paras_{};
        // filter dynamic flags
        std::array<std::atomic<bool>, kBandNum> a_dynamic_on_{};
        std::array<std::atomic<bool>, kBandNum> a_dynamic_bypass_{};
        std::array<bool, kBandNum> dynamic_on_{};
        std::array<bool, kBandNum> dynamic_bypass_{};
        zlchore::thread::Notifier to_update_dynamic_{false};

        // filters for calculating prototype response and biquad response
        std::array<zldsp::filter::Ideal<float, kFilterSize>, kBandNum> ideals_{};
        // spectrum processing
        std::array<zldsp::filter::SpecResponse<float>, kBandNum> spec_response_
            = make_array_of<zldsp::filter::SpecResponse<float>, kBandNum>();
        std::array<zldsp::filter::SpecFollower<float>, kBandNum> spec_follower_
            = make_array_of<zldsp::filter::SpecFollower<float>, kBandNum>();
        std::array<zldsp::filter::SpecDynamic<float>, kBandNum> spec_dynamic_
            = make_array_of<zldsp::filter::SpecDynamic<float>, kBandNum>();
        zldsp::filter::SpecSmoother<float> spec_smoother_;
        // fft working space
        std::unique_ptr<zldsp::fft::RFFT<float>> fft_low_;
        std::unique_ptr<zldsp::fft::RFFT<float>> fft_medium_;
        std::unique_ptr<zldsp::fft::RFFT<float>> fft_high_;
        std::unique_ptr<zldsp::fft::RFFT<float>> fft_extreme_;
        zldsp::fft::RFFT<float>* fft_{nullptr};
        size_t fft_order_ = 13;
        size_t fft_size_ = static_cast<size_t>(1) << fft_order_;
        size_t num_bin_ = fft_size_ / 2 + 1;
        size_t num_bin_effective_ = fft_size_ / 2;
        zldsp::vector::aligned_vector<float> window1_, window2_, window_bypass_;
        size_t fft_count_ = 0;
        size_t fft_pos_ = 0;
        size_t fft_hop_size_ = fft_size_ / 4;

        std::array<zldsp::vector::aligned_vector<float>, 4> input_fifos_, output_fifos_;
        std::array<zldsp::vector::aligned_vector<float>, 4> fft_ins_;
        std::array<zldsp::vector::aligned_vector<float>, 2> fft_out_reals_, fft_out_imags_;

        size_t dispatch_mask_{0};
        ChannelData stereo_data_, l_data_, r_data_, m_data_, s_data_;

        SideStatus side_status_{SideStatus::kNotRequired};

        std::atomic<bool> a_is_ext_side_{false};
        bool is_ext_side_{false};

        void processFrame(bool is_bypass);

        void processSide();

        void computeSideAbsSqrFromMain();

        void processDualChannelSide(ChannelData& ch1, ChannelData& ch2);

        void processSideLR();

        void processSideMS();

        void processSideLRMS();

        void processDynamicBands(ChannelData& data);

        void processMain(bool is_bypass, bool perform_fft);

        void multiplyWithWindow(float* HWY_RESTRICT in1_ptr, float* HWY_RESTRICT in2_ptr,
                                const float* HWY_RESTRICT window_ptr) const;

        void handleAsyncUpdate() override;
    };
}
