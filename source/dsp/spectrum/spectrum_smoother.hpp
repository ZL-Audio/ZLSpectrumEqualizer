// Copyright (C) 2025 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <span>
#include <vector>
#include <algorithm>

#include "../vector/vector.hpp"

namespace zldsp::spectrum {
    template <typename FloatType>
    class SpectrumSmoother final {
    public:
        SpectrumSmoother() = default;

        /**
         * prepare for a maximum spectrum size
         * @param max_spectrum_size
         */
        void prepare(const size_t max_spectrum_size) {
            prefix_sum_.reserve(max_spectrum_size + 1);
            low_idx_.reserve(max_spectrum_size);
            high_idx_.reserve(max_spectrum_size);
            scale_.reserve(max_spectrum_size);
        }

        /**
         * set current spectrum size
         * @param spectrum_size
         */
        void setSpectrumSize(const size_t spectrum_size) {
            prefix_sum_.resize(spectrum_size + 1);
            low_idx_.resize(spectrum_size);
            high_idx_.resize(spectrum_size);
            scale_.resize(spectrum_size);
            updateLowHighIdx();
        }

        /**
         * set spectrum smoothing parameter (should be greater or equal to 1.0)
         * @param smooth
         */
        void setSmooth(const FloatType smooth) {
            smooth_ = std::max(FloatType(1), smooth);
            updateLowHighIdx();
        }

        /**
         * smooth the spectrum [start_idx, end_idx] by two-pass box blur
         * @param spectrum
         * @param start_idx
         * @param end_idx
         */
        void smooth(std::span<FloatType> spectrum, const size_t start_idx, const size_t end_idx) {
            const size_t pass1_start = low_idx_[start_idx];
            const size_t pass1_end = std::min(high_idx_[end_idx], spectrum.size() - 1);
            internal_smooth(spectrum, pass1_start, pass1_end);
            internal_smooth(spectrum, start_idx, end_idx);
            auto v1 = kfr::make_univector(spectrum.data() + start_idx, end_idx - start_idx + 1);
            auto v2 = kfr::make_univector(scale_.data() + start_idx, end_idx - start_idx + 1);
            v1 = v1 * v2;
        }

    private:
        std::vector<FloatType> prefix_sum_{};
        std::vector<size_t> low_idx_{}, high_idx_{};
        std::vector<FloatType> scale_{};
        FloatType smooth_{FloatType(1)};

        void updateLowHighIdx() {
            const auto spectrum_size = low_idx_.size();
            const auto smooth = static_cast<double>(smooth_);
            const auto smooth_r = 1.0 / smooth;
            for (size_t idx = 0; idx < spectrum_size; ++idx) {
                low_idx_[idx] = static_cast<size_t>(std::round(static_cast<double>(idx) * smooth_r));
                high_idx_[idx] = std::min(static_cast<size_t>(std::round(static_cast<double>(idx) * smooth) + 1.0),
                                          spectrum_size);
                const auto diff = static_cast<double>(high_idx_[idx] - low_idx_[idx]);
                scale_[idx] = static_cast<FloatType>(1.0 / (diff * diff));
            }
        }

        void internal_smooth(std::span<FloatType> spectrum, const size_t start_idx, const size_t end_idx) {
            // calculate prefix sum
            const auto low = low_idx_[start_idx];
            const auto high = high_idx_[end_idx];
            double sum{0.0};
            for (auto idx = low; idx < high; ++idx) {
                prefix_sum_[idx] = static_cast<FloatType>(sum);
                sum += static_cast<double>(spectrum[idx]);
            }
            prefix_sum_[high] = static_cast<FloatType>(sum);
            // calculate moving average
            for (auto idx = start_idx; idx <= end_idx; ++idx) {
                spectrum[idx] = prefix_sum_[high_idx_[idx]] - prefix_sum_[low_idx_[idx]];
            }
        }
    };
}
