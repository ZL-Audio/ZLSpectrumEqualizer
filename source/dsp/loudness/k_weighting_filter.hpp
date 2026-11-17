// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <algorithm>
#include <array>
#include <span>
#include <vector>

#include "../filter/filter_design/filter_design.hpp"
#include "../filter/iir_filter/coeff/ivantsov_coeff.hpp"

namespace zldsp::loudness {
    /**
     * a K-weighting filter used for integrated loudness measurement
     * @tparam FloatType the float type of input audio buffer
     */
    template <typename FloatType>
    class KWeightingFilter {
    public:
        /**
         *
         * @param use_low_pass whether to use an extra lowpass filter at 22,000 Hz
         */
        explicit KWeightingFilter(const bool use_low_pass = false) :
            use_low_pass_(use_low_pass) {
        }

        void prepare(const double sample_rate, const size_t num_channels) {
            high_pass_ = designBiquad(filter::kHighPass, sample_rate, 38.13713296248405, 0.0, 0.500242812458813);
            high_shelf_ = designBiquad(filter::kHighShelf, sample_rate, 1500.6868667368922,
                                       3.9993623475151354, 0.7096433028107384);
            low_pass_enabled_ = use_low_pass_ && sample_rate > 40000.0;
            if (low_pass_enabled_) {
                low_pass_ = designBiquad(filter::kLowPass, sample_rate, std::min(22000.0, 0.49964 * sample_rate),
                                         0.0, 0.7071067811865476);
            }
            states_.assign(num_channels, ChannelState{});
        }

        void reset() {
            std::ranges::fill(states_, ChannelState{});
        }

        void process(std::span<FloatType*> buffer, const size_t num_samples) {
            if (low_pass_enabled_) {
                processSections<true>(buffer, num_samples);
            } else {
                processSections<false>(buffer, num_samples);
            }
        }

    private:
        using Coefficients = std::array<double, 5>;

        static Coefficients designBiquad(filter::FilterType type, double sample_rate, double freq, double gain,
                                         double q) {
            Coefficients coeffs{};
            filter::FilterDesign::updateCoeffs<filter::IvantsovCoeff>(
                type, 2, freq, sample_rate, gain, q, std::span{&coeffs, 1});
            return coeffs;
        }

        struct BiquadState {
            FloatType s1{}, s2{};

            FloatType process(const Coefficients& coeffs, FloatType sample) {
                const auto input = static_cast<double>(sample);
                const auto output = input * coeffs[2] + static_cast<double>(s1);
                s1 = static_cast<FloatType>(input * coeffs[3] - output * coeffs[0]) + s2;
                s2 = static_cast<FloatType>(input * coeffs[4] - output * coeffs[1]);
                return static_cast<FloatType>(output);
            }
        };

        struct ChannelState {
            BiquadState high_pass, high_shelf, low_pass;
        };

        template <bool with_low_pass>
        void processSections(std::span<FloatType*> buffer, size_t num_samples) {
            for (size_t i = 0; i < num_samples; ++i) {
                for (size_t ch = 0; ch < buffer.size(); ++ch) {
                    auto& state = states_[ch];
                    auto sample = state.high_pass.process(high_pass_, buffer[ch][i]);
                    sample = state.high_shelf.process(high_shelf_, sample);
                    if constexpr (with_low_pass) {
                        sample = state.low_pass.process(low_pass_, sample);
                    }
                    buffer[ch][i] = sample * kBias;
                }
            }
        }

        bool use_low_pass_{false};
        bool low_pass_enabled_{false};
        Coefficients high_pass_{}, high_shelf_{}, low_pass_{};
        std::vector<ChannelState> states_;
        static constexpr FloatType kBias = static_cast<FloatType>(1.0051643348917434);
    };
}
