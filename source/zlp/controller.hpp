// Copyright (C) 2025 - zsliu98
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

#include "../dsp/fft/fft.hpp"

#include "../dsp/spectrum/spectrum_smoother.hpp"
#include "../dsp/spectrum/dynamic_spectrum.hpp"

namespace zlp {
    class Controller final : private juce::AsyncUpdater {
    private:
        struct ChannelFFTData {
            size_t count_{0};
            size_t pos_{0};

            kfr::univector<float> ch0_input_fifo_{}, ch0_output_fifo_{};
            kfr::univector<float> ch1_input_fifo_{}, ch1_output_fifo_{};
            kfr::univector<float> ch0_ext_input_fifo_{}, ch0_ext_output_fifo_{};
            kfr::univector<float> ch1_ext_input_fifo_{}, ch1_ext_output_fifo_{};

            kfr::univector<std::complex<float>> ch0_cspectrum_{}, ch1_cspectrum_{};
            kfr::univector<float> ch0_side_spectrum_{}, ch1_side_spectrum_{};

            kfr::univector<float> ch0_static_gain_{}, ch1_static_gain_{};
            kfr::univector<float> ch0_dyn_gain_{}, ch1_dyn_gain_{};

            size_t ch0_dyn_left_{}, ch0_dyn_length_{};
            size_t ch1_dyn_left_{}, ch1_dyn_length_{};

            size_t ch0_side_left_{}, ch0_side_length_{};
            size_t ch1_side_left_{}, ch1_side_length_{};

            std::vector<size_t> ch0_dyn_bands_{};
            std::vector<size_t> ch1_dyn_bands_{};

            bool is_ext_side_{false};

            bool not_off_{false};

            bool ch0_fft_required_{false}, ch1_fft_required_{false};

            bool ch0_ext_fft_required_{false}, ch1_ext_fft_required_{false};

            bool ch0_side_required_{false}, ch1_side_required_{false};

            bool ch0_not_off_{false}, ch1_not_off_{false};

            bool stereo_dyn_not_off_{false};

            bool ch0_dyn_not_off_{false}, ch1_dyn_not_off_{false};
        };

    public:
        static constexpr size_t kFilterSize = 16;
        static constexpr size_t kAnalyzerPointNum = 251;

        explicit Controller(juce::AudioProcessor& p);

        void prepare(double sample_rate, size_t max_num_samples);

    private:
        juce::AudioProcessor& p_ref_;

        std::array<kfr::univector<float>, kBandNum> base_gains_{};
        kfr::univector<float> target_sqr_gain_{};
        std::array<kfr::univector<float>, 4> lrms_gains_{};
        zldsp::filter::Ideal<float, kFilterSize> ideal_filter_{};

        std::array<bool, kBandNum> lrms_side_shuffle_{};

        ChannelFFTData lr_fft_data_{}, ms_fft_data_{};
        kfr::univector<float> stereo_side_spectrum_{};
        size_t stereo_left_{}, stereo_right_{};
        std::vector<size_t> stereo_dyn_bands_{};
        size_t stereo_dyn_left_{}, stereo_dyn_length_{};
        kfr::univector<float> stereo_dyn_gain_{};

        std::unique_ptr<kfr::dft_plan_real<float>> fft_plan_;
        kfr::univector<float> fft_in_;
        kfr::univector<std::complex<float>> fft_out_;
        kfr::univector<kfr::u8> fft_temp_buffer_;
        kfr::univector<float> window1_, window2_;
        size_t fft_order_{0}, fft_size_{0};
        size_t num_bin_{0}, hop_size_{0};
        static constexpr float kWindowCorrection = 2.0f / 3.0f;
        static constexpr float kBypassCorrection = 1.0f / 4.0f;

        zldsp::spectrum::SpectrumSmoother<float> spectrum_smoother_;

        std::array<zldsp::spectrum::DynamicSpectrum<float>, kBandNum> dynamic_spectrums_{};

        std::atomic<bool> update_flag_{};

        std::atomic<int> latency_{0};

        void handleAsyncUpdate() override;

        void prepareBuffer();

        void processBuffer(ChannelFFTData& fft_data, size_t num_samples,
                           float* __restrict ch0, float* __restrict ch1,
                           float* __restrict side_ch0, float* __restrict side_ch1);

        template <bool is_ext_side = false>
        static void fifoPushPop(float* __restrict buffer, float* __restrict input_fifo, float* __restrict output_fifo,
                                size_t buffer_pos, size_t fifo_pos, size_t copy_size);

        template <bool bypass = false>
        void processSpectrum(ChannelFFTData& fft_data);

        void forwardFFT(const kfr::univector<float>& fifo, size_t fifo_pos,
                        kfr::univector<std::complex<float>>& cspectrum);

        void backwardFFT(kfr::univector<float>& fifo, size_t fifo_pos,
                         const kfr::univector<std::complex<float>>& cspectrum);
    };
}
