// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <cmath>
#include <numbers>
#include "../../vector/vector.hpp"

namespace zldsp::filter {
    namespace hn = hwy::HWY_NAMESPACE;

    template <typename FloatType>
    class SpecFollower {
    public:
        static void fillScaling(const double sample_rate, const size_t fft_size, std::span<double> scaling) {
            const auto spec_size = fft_size / 2;
            const double bin_width = sample_rate / static_cast<double>(fft_size);
            static constexpr double kRefFreq = 1000.0;
            {
                const double bin_freq = 0.25 * bin_width;
                scaling[0] = kRefFreq / bin_freq;
            }
            for (size_t i = 1; i < spec_size; ++i) {
                const double bin_freq = static_cast<double>(i) * bin_width;
                scaling[i] = kRefFreq / bin_freq;
            }
        }

        explicit SpecFollower() = default;

        void prepare(const double sample_rate, const size_t fft_size) {
            const double frame_rate = sample_rate / static_cast<double>(fft_size / 4);
            exp_factor = -2.0 * std::numbers::pi * 1000.0 / frame_rate;

            const auto spec_size = fft_size / 2;
            attacks_.resize(spec_size);
            releases_.resize(spec_size);
        }

        void updateAttack(const double attack_time, const double skew, const std::span<double>& scaling) {
            updateAR(attack_time, attacks_, skew, scaling);
        }

        void updateRelease(const double release_time, const double skew, const std::span<double>& scaling) {
            updateAR(release_time, releases_, skew, scaling);
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
                      const double skew, const std::span<double>& scaling) {
            for (size_t i = 0; i < ar.size(); ++i) {
                const auto final_scale = 1.0 + skew * (scaling[i] - 1.0);
                const double scaled_time = ar_time * final_scale;
                if (scaled_time < static_cast<FloatType>(0.001)) {
                    ar[i] = static_cast<FloatType>(0);
                } else {
                    ar[i] = static_cast<FloatType>(std::exp(exp_factor / scaled_time));
                }
            }
        }
    };
}
