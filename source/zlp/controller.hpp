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

#include "../dsp/splitter/inplace_ms_splitter.hpp"

namespace zlp {
    class Controller final : private juce::AsyncUpdater {
    private:
        struct ChannelFFTData {
            kfr::univector<float> static_gain_{};

            kfr::univector<float> dyn_gain_{};
            kfr::univector<float> final_gain_{};
            size_t dyn_left_{}, dyn_length_{};

            size_t side_left_{}, side_length_{};

            std::vector<size_t> dyn_not_off_bands_{};

            std::vector<size_t> dyn_on_bands_{};

            bool not_off_{false}, dyn_not_off{false};
        };

    public:
        static constexpr size_t kFilterSize = 16;
        static constexpr size_t kAnalyzerPointNum = 251;

        explicit Controller(juce::AudioProcessor& p);

        void prepare(double sample_rate, size_t max_num_samples);

        template <bool bypass = false>
        void processBuffer(std::array<float*, 2> main_buffer,
                           std::array<float*, 2> side_buffer,
                           size_t num_samples);

    private:
        static constexpr float kSqrt2Over2 = static_cast<float>(
            0.7071067811865475244008443621048490392848359376884740365883398690);

        juce::AudioProcessor& p_ref_;

        std::array<kfr::univector<float>, kBandNum> base_gains_{};
        kfr::univector<float> target_sqr_gain_{};
        std::array<kfr::univector<float>, 4> lrms_gains_{};
        zldsp::filter::Ideal<float, kFilterSize> ideal_filter_{};

        std::array<bool, kBandNum> dynamic_bypass_{};

        ChannelFFTData stereo_data_{}, l_data_{}, r_data_{}, m_data_{}, s_data_{};

        std::unique_ptr<kfr::dft_plan_real<float>> fft_plan_;
        kfr::univector<float> fft_in_;
        kfr::univector<std::complex<float>> fft_out_;
        kfr::univector<kfr::u8> fft_temp_buffer_;
        kfr::univector<float> window1_, window2_;
        size_t fft_order_{0}, fft_size_{0};
        size_t num_bin_{0}, hop_size_{0};
        size_t fifo_count_{0};
        size_t fifo_pos_{0};
        static constexpr float kWindowCorrection = 2.0f / 3.0f;
        static constexpr float kBypassCorrection = 1.0f / 4.0f;
        std::array<kfr::univector<float>, 2> input_fifo_{};
        std::array<kfr::univector<float>, 2> output_fifo_{};
        std::array<kfr::univector<float>, 2> ext_fifo_{};
        std::array<kfr::univector<std::complex<float>>, 2> cspectrum_{};
        std::array<kfr::univector<std::complex<float>>, 2> side_cspectrum_{};
        kfr::univector<float> side_spectrum_{};

        zldsp::spectrum::SpectrumSmoother<float> spectrum_smoother_;

        std::array<zldsp::spectrum::DynamicSpectrum<float>, kBandNum> dynamic_spectrums_{};

        bool dyn_not_off_{false};

        bool ext_side_{false};

        enum class LRMSStatus {
            kStereo, kLR, kMS, kLRMS
        };

        std::atomic<bool> update_flag_{};

        std::atomic<int> latency_{0};

        void handleAsyncUpdate() override;

        void prepareBuffer();

        template <bool bypass = false, bool ext_side = false>
        void processBufferInternal(std::array<float*, 2> main_buffer,
                                   std::array<float*, 2> side_buffer,
                                   size_t num_samples);

        template <bool bypass = false, bool ext_side = false>
        void processFrame();

        template <bool bypass = false, bool ext_side = false>
        void processSpectrumStereo();

        void calculateSideSpectrumStereo(kfr::univector<std::complex<float>>& side0_cspectrum,
                                         kfr::univector<std::complex<float>>& side1_cspectrum);

        template <bool bypass = false, bool ext_side = false>
        void processSpectrumLR();

        template <bool bypass = false, bool ext_side = false>
        void processSpectrumMS();

        template <bool bypass = false, bool ext_side = false>
        void processSpectrumLRMS();

        void calculateSideSpectrumChannel(const ChannelFFTData& fft_data,
                                          kfr::univector<float>& side_spectrum,
                                          kfr::univector<std::complex<float>>& side_cspectrum);

        template <bool bypass = false>
        void processSideSpectrum(ChannelFFTData& fft_data, kfr::univector<float>& side_spectrum);

        void calculateFinalGainChannel(ChannelFFTData& fft_data);
    };
}
