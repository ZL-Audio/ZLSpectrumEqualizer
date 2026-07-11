// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <span>
#include <cmath>
#include "../../vector/vector.hpp"

namespace zldsp::filter {
    namespace hn = hwy::HWY_NAMESPACE;

    template <typename FloatType>
    class SpecTilter {
    public:
        explicit SpecTilter() = default;

        void prepare(const size_t fft_size) {
            tilt_shift_.resize(fft_size / 2 + 1);
        }

        void setTiltSlope(const double sample_rate, const double slope_per_oct) {
            const size_t num_bins = tilt_shift_.size();
            const double delta = sample_rate * 0.5 / static_cast<double>(num_bins - 1);
            const double power = slope_per_oct * std::log2(10.0) / 10.0;
            const double delta_khz = delta / 1000.0;
            for (size_t i = 1; i < num_bins; ++i) {
                const double freq_khz = static_cast<double>(i) * delta_khz;
                tilt_shift_[i] = static_cast<FloatType>(std::pow(freq_khz, power));
            }
            if (num_bins > 1) {
                tilt_shift_[0] = tilt_shift_[1];
            }
        }

        auto& getTilt() {
            return tilt_shift_;
        }

        const auto& getTilt() const {
            return tilt_shift_;
        }

    private:
        vector::aligned_vector<FloatType> tilt_shift_{};
    };
}
