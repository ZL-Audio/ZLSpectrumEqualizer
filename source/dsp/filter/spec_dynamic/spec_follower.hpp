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
#include <cmath>
#include <numbers>
#include <span>
#include "../../vector/vector.hpp"

namespace zldsp::filter {
    namespace hn = hwy::HWY_NAMESPACE;

    template <typename FloatType>
    class SpecFollower {
    public:
        explicit SpecFollower() = default;

        static void fillScaling(const double bin_width, const double skew, const std::span<double> curve) {
            constexpr double kReferenceHz = 1000.0;
            static constexpr hn::ScalableTag<double> d;
            static constexpr size_t lanes = hn::MaxLanes(d);
            const auto v_width = hn::Set(d, bin_width);
            const auto v_skew = hn::Set(d, skew);
            const auto v_reference = hn::Set(d, kReferenceHz);

            const size_t upper_octave = (curve.size() + 1) / 2;
            for (size_t i = upper_octave; i < curve.size(); i += lanes) {
                const auto v_freq = hn::Mul(hn::Iota(d, static_cast<double>(i)), v_width);
                const auto v_log = hn::Log(d, hn::Div(v_reference, v_freq));
                const auto v_scale = hn::Exp(d, hn::Mul(v_skew, v_log));
                hn::StoreN(v_scale, d, curve.data() + i, std::min(lanes, curve.size() - i));
            }

            const double octave_ratio = std::exp2(skew);
            for (size_t i = upper_octave; i > 1; --i) {
                const size_t bin = i - 1;
                curve[bin] = curve[2 * bin] * octave_ratio;
            }
            if (curve.size() > 1) {
                curve[0] = curve[1] * octave_ratio * octave_ratio;
            } else if (!curve.empty()) {
                curve[0] = std::pow(kReferenceHz / (0.25 * bin_width), skew);
            }
        }

        void prepare(const double sample_rate, const size_t fft_size) {
            const double frame_rate = sample_rate / static_cast<double>(fft_size / 4);
            exp_factor = -2.0 * std::numbers::pi * 1000.0 / frame_rate;

            const auto spec_size = fft_size / 2;
            attacks_.resize(spec_size);
            releases_.resize(spec_size);
        }

        void updateAttack(const double attack_time, const std::span<const double> scaling) {
            updateAR(attack_time, attacks_, scaling);
        }

        void updateRelease(const double release_time, const std::span<const double> scaling) {
            updateAR(release_time, releases_, scaling);
        }

        auto& getAttack() {
            return attacks_;
        }

        auto& getRelease() {
            return releases_;
        }

    private:
        double exp_factor{0.};
        vector::aligned_vector<FloatType> attacks_;
        vector::aligned_vector<FloatType> releases_;

        void updateAR(const double ar_time, vector::aligned_vector<FloatType>& ar,
                      const std::span<const double> scaling) {
            for (size_t i = 0; i < ar.size(); ++i) {
                const double scaled_time = ar_time * scaling[i];
                if (scaled_time < static_cast<FloatType>(0.001)) {
                    ar[i] = static_cast<FloatType>(0);
                } else {
                    ar[i] = static_cast<FloatType>(std::exp(exp_factor / scaled_time));
                }
            }
        }
    };
}
