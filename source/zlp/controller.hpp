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
#include "../dsp/filter/spec_dynamic/spec_tilter.hpp"
#include "../dsp/fft/zldsp_fft_include.hpp"
#include "../dsp/loudness/lufs_matcher.hpp"
#include "../dsp/gain/gain.hpp"

#include "../chore/thread/notifier.hpp"

#include "../dsp/analyzer/analyzer_base/analyzer_sender_base.hpp"

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

        void prepareBuffer();

        void process(const std::array<float*, 4>& buffer, size_t num_samples, bool is_bypass);

        template <bool has_stereo, bool has_l, bool has_r, bool has_m, bool has_s>
        void processMainImpl(bool perform_fft);

        void setFilterStatus(const size_t idx, const FilterStatus filter_status) {
            a_filter_status_[idx].store(filter_status, std::memory_order::relaxed);
            to_update_filter_status_.signal();
            to_update_.signal();
        }

        void setLRMS(const size_t idx, const FilterStereo filter_stereo) {
            a_lrms_[idx].store(filter_stereo, std::memory_order::relaxed);
            to_update_lrms_.signal();
            to_update_.signal();
        }

        void setExtSide(const bool is_ext_side) {
            a_is_ext_side_.store(is_ext_side, std::memory_order::relaxed);
            to_update_.signal();
        }

        bool getExtSide() const {
            return is_ext_side_;
        }

        auto& getAnalyzerSender() {
            return analyzer_sender_;
        }

        void setLoudnessMatchON(const bool f) {
            loudness_matcher_on_.store(f, std::memory_order::relaxed);
        }

        double getLUFSMatcherDiff() const {
            return loudness_matcher_.getDiff();
        }

        void setTargetGain(const size_t band, const float gain) {
            empty_target_gains_[band].store(gain, std::memory_order::relaxed);
            to_update_empty_targets_[band].signal();
            to_update_spec_response_.signal();
            to_update_.signal();
        }

        void setSpecSmoothValue(const float smooth) {
            a_spec_smooth_value_.store(smooth, std::memory_order::relaxed);
            to_update_spec_smooth_.signal();
            to_update_.signal();
        }

        void setSpecSmoothType(const zldsp::filter::SpecSmoother<float>::SmoothMethod method) {
            a_spec_smooth_type_.store(method, std::memory_order::relaxed);
            to_update_spec_smooth_.signal();
            to_update_.signal();
        }

        void setSpecTiltSlope(const float slope) {
            a_spec_tilt_slope_.store(slope, std::memory_order::relaxed);
            to_update_spec_tilt_.signal();
            to_update_.signal();
        }

        void setOutputGain(const float gain) {
            a_output_gain_.store(gain, std::memory_order::relaxed);
            to_update_output_gain_.signal();
            to_update_.signal();
        }

        void setSpecAttack(const size_t idx, const float attack) {
            a_spec_attack_[idx].store(attack, std::memory_order::relaxed);
            to_update_spec_attack_[idx].signal();
            to_update_.signal();
        }

        void setSpecRelease(const size_t idx, const float release) {
            a_spec_release_[idx].store(release, std::memory_order::relaxed);
            to_update_spec_release_[idx].signal();
            to_update_.signal();
        }

        void setSpecFollowerSkewAttack(const float skew) {
            a_spec_skew_attack_.store(skew, std::memory_order::relaxed);
            to_update_spec_skew_.signal();
            to_update_.signal();
        }

        void setSpecFollowerSkewRelease(const float skew) {
            a_spec_skew_release_.store(skew, std::memory_order::relaxed);
            to_update_spec_skew_.signal();
            to_update_.signal();
        }

        void setFFTResolution(const FFTResolution resolution) {
            a_fft_resolution_.store(resolution, std::memory_order::relaxed);
            to_update_fft_resolution_.signal();
            to_update_.signal();
        }

        void setDynamicON(const size_t idx, const bool dynamic_on) {
            a_dynamic_on_[idx].store(dynamic_on, std::memory_order::relaxed);
            to_update_dynamic_status_.signal();
            to_update_.signal();
        }

        void setDynamicBypass(const size_t idx, const bool dynamic_bypass) {
            a_dynamic_bypass_[idx].store(dynamic_bypass, std::memory_order::relaxed);
            to_update_dynamic_status_.signal();
            to_update_.signal();
        }

        void setDynamicMode(const size_t idx, const DynamicMode mode) {
            a_dynamic_mode_[idx].store(mode, std::memory_order::relaxed);
            to_update_dynamic_status_.signal();
            to_update_.signal();
        }

        void setSpecThresholdAbs(const size_t idx, const float threshold) {
            a_spec_threshold_abs_[idx].store(threshold, std::memory_order::relaxed);
            to_update_spec_threshold_[idx].signal();
            to_update_.signal();
        }

        void setSpecThresholdBand(const size_t idx, const float threshold) {
            a_spec_threshold_band_[idx].store(threshold, std::memory_order::relaxed);
            to_update_spec_threshold_[idx].signal();
            to_update_.signal();
        }

        void setSpecThresholdRel(const size_t idx, const float threshold) {
            a_spec_threshold_rel_[idx].store(threshold, std::memory_order::relaxed);
            to_update_spec_threshold_[idx].signal();
            to_update_.signal();
        }

        void setSpecKnee(const size_t idx, const float knee) {
            a_spec_knee_[idx].store(knee, std::memory_order::relaxed);
            to_update_spec_knee_[idx].signal();
            to_update_.signal();
        }

        auto& getEmptyFilters() {
            return emptys_;
        }

        auto& getEmptyUpdateFlags() {
            return to_update_empty_bases_;
        }

        auto& getUpdateFlag() {
            return to_update_;
        }

        auto& getSpecResponseUpdateFlag() {
            return to_update_spec_response_;
        }

        auto& getChannelDataUpdateFlag() {
            return to_update_channel_data_;
        }

        void setSGCON(const float f) {
            sgc_on_.store(f, std::memory_order::relaxed);
            to_update_sgc_on_.signal();
            to_update_.signal();
        }

        double getDisplayedGain() const {
            return displayed_gain_.load(std::memory_order::relaxed);
        }

    private:
        enum class SideStatus {
            kNotRequired, kLR, kMS, kLRMS
        };

        static constexpr hn::ScalableTag<float> d;
        static constexpr size_t lanes = hn::MaxLanes(d);

        struct ChannelData {
            std::vector<size_t> bands{};
            zldsp::vector::aligned_vector<float> static_response;

            zldsp::vector::aligned_vector<float> fft_side_abs_sqr;
            zldsp::filter::SpecSmoother<float>::SmoothBounds smooth_bounds;

            size_t dynamic_start_idx{0}, dynamic_end_idx{0};
            std::vector<size_t> dynamic_bands{};
            zldsp::vector::aligned_vector<float> dynamic_response;
            bool require_relative{false};
        };

        juce::AudioProcessor& p_ref_;
        zlchore::thread::Notifier to_update_{false};
        // fft resolution
        std::atomic<FFTResolution> a_fft_resolution_{FFTResolution::kMedium};
        zlchore::thread::Notifier to_update_fft_resolution_{false};
        std::atomic<int> latency_{0};
        // filter status
        std::array<std::atomic<FilterStatus>, kBandNum> a_filter_status_{};
        std::array<FilterStatus, kBandNum> filter_status_{FilterStatus::kOff};
        zlchore::thread::Notifier to_update_filter_status_{false};
        std::vector<size_t> on_bands_{};
        // filter l/r/m/s
        std::array<std::atomic<FilterStereo>, kBandNum> a_lrms_{};
        std::array<FilterStereo, kBandNum> lrms_{};
        zlchore::thread::Notifier to_update_lrms_{false};
        // empty filters for holding atomic parameters
        std::array<zldsp::filter::Empty, kBandNum> emptys_{};
        std::array<zlchore::thread::Notifier, kBandNum> to_update_empty_bases_{};
        std::array<std::atomic<float>, kBandNum> empty_target_gains_{};
        std::array<zlchore::thread::Notifier, kBandNum> to_update_empty_targets_{};
        // filter dynamic flags
        std::array<std::atomic<DynamicMode>, kBandNum> a_dynamic_mode_{};
        std::array<DynamicMode, kBandNum> dynamic_mode_{};
        std::array<std::atomic<bool>, kBandNum> a_dynamic_on_{};
        std::array<std::atomic<bool>, kBandNum> a_dynamic_bypass_{};
        std::array<bool, kBandNum> dynamic_on_{};
        std::array<bool, kBandNum> dynamic_bypass_{};
        zlchore::thread::Notifier to_update_dynamic_status_{false};

        // filter dynamic parameters
        std::atomic<float> a_spec_smooth_value_{0.0f};
        std::atomic<zldsp::filter::SpecSmoother<float>::SmoothMethod> a_spec_smooth_type_{zldsp::filter::SpecSmoother<float>::SmoothMethod::kOCT};
        zlchore::thread::Notifier to_update_spec_smooth_{false};

        std::atomic<float> a_spec_tilt_slope_{0.0f};
        zlchore::thread::Notifier to_update_spec_tilt_{false};

        std::atomic<float> a_output_gain_{0.0f};
        zlchore::thread::Notifier to_update_output_gain_{false};

        std::array<std::atomic<float>, kBandNum> a_spec_attack_{};
        std::array<zlchore::thread::Notifier, kBandNum> to_update_spec_attack_{};
        std::array<std::atomic<float>, kBandNum> a_spec_release_{};
        std::array<zlchore::thread::Notifier, kBandNum> to_update_spec_release_{};
        std::atomic<float> a_spec_skew_attack_{0.9f};
        std::atomic<float> a_spec_skew_release_{0.2f};
        zlchore::thread::Notifier to_update_spec_skew_{false};

        std::array<std::atomic<float>, kBandNum> a_spec_threshold_abs_{};
        std::array<std::atomic<float>, kBandNum> a_spec_threshold_band_{};
        std::array<std::atomic<float>, kBandNum> a_spec_threshold_rel_{};
        std::array<zlchore::thread::Notifier, kBandNum> to_update_spec_threshold_{};
        std::array<std::atomic<float>, kBandNum> a_spec_knee_{};
        std::array<zlchore::thread::Notifier, kBandNum> to_update_spec_knee_{};

        // filters for calculating prototype response and biquad response
        zldsp::vector::aligned_vector<float> ws_;
        zldsp::filter::Ideal<float, kFilterSize> ideal_{};
        std::array<bool, kBandNum> to_update_bases_{false};
        zlchore::thread::Notifier to_update_spec_response_{false};

        // spectrum processing
        std::array<zldsp::filter::SpecResponse<float>, kBandNum> spec_response_
            = make_array_of<zldsp::filter::SpecResponse<float>, kBandNum>();
        std::array<zldsp::filter::SpecFollower<float>, kBandNum> spec_follower_
            = make_array_of<zldsp::filter::SpecFollower<float>, kBandNum>();
        std::array<zldsp::filter::SpecDynamic<float>, kBandNum> spec_dynamic_
            = make_array_of<zldsp::filter::SpecDynamic<float>, kBandNum>();
        zldsp::filter::SpecSmoother<float> spec_smoother_;
        zldsp::filter::SpecTilter<float> spec_tilter_;
        std::vector<double> spec_follower_scaling_{};
        std::array<float, kBandNum> band_avgs_{};

        // fft working space
        double sample_rate_{48000.0};
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
        std::array<ChannelData, 5> channel_datas_{};
        std::array<bool, 5> to_update_channel_static_{false};
        std::array<bool, 5> to_update_channel_smooth_bounds_{false};
        zlchore::thread::Notifier to_update_channel_data_{false};
        ChannelData& stereo_data_{channel_datas_[0]};
        ChannelData& l_data_{channel_datas_[1]};
        ChannelData& r_data_{channel_datas_[2]};
        ChannelData& m_data_{channel_datas_[3]};
        ChannelData& s_data_{channel_datas_[4]};

        SideStatus side_status_{SideStatus::kNotRequired};

        std::atomic<bool> a_is_ext_side_{false};
        bool is_ext_side_{false};

        zldsp::analyzer::AnalyzerSenderBase<float, 3> analyzer_sender_{};
        std::array<std::vector<float>, 2> pre_analyzer_temp_;
        std::array<std::vector<float>, 2> post_analyzer_temp_;
        std::array<std::vector<float>, 2> side_analyzer_temp_;
        std::array<float*, 2> pre_analyzer_ptrs_{};
        std::array<float*, 2> post_analyzer_ptrs_{};
        std::array<float*, 2> side_analyzer_ptrs_{};

        // loudness matcher
        std::atomic<bool> loudness_matcher_on_{false};
        bool c_loudness_matcher_on_{false};
        zldsp::loudness::LUFSMatcher<float, true> loudness_matcher_{};

        std::atomic<float> sgc_on_{false};
        zlchore::thread::Notifier to_update_sgc_on_{false};
        bool c_sgc_on_{false};
        std::array<double, kBandNum> sgc_values_{};
        float c_sgc_gain_linear_{1.f};
        zldsp::gain::Gain<float> sgc_gain_{};
        std::atomic<double> displayed_gain_{1.};

        // output gain
        zldsp::gain::Gain<float> output_gain_dsp_{};

        void prepareFFTPlans();

        void resizeWorkingSpace();

        void processFrame(bool is_bypass);

        void processSide();

        void computeSideAbsSqrFromMain();

        void processSideLR();

        void processSideMS();

        void processSideLRMS();

        void processDualChannelSide(ChannelData& ch1, ChannelData& ch2);

        void processDynamicBands(ChannelData& data);

        void processMain(bool is_bypass, bool perform_fft);

        void multiplyWithWindow(float* HWY_RESTRICT in1_ptr, float* HWY_RESTRICT in2_ptr,
                                const float* HWY_RESTRICT window_ptr) const;

        void updateFFTResolution();

        void updateFilterStatus();

        void updateDynamicStatus();

        void updateSpecSmooth();

        void updateSpecTilt();

        void updateSpecFollower();

        void updateSpecDynamic();

        void updateSpecResponse();

        void updateLRMS();

        void updateChannelData();

        void updateOutputGain();
        void updateSGC();

        void handleAsyncUpdate() override;
    };
}
