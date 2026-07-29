// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <cstddef>

namespace zlp {
    inline constexpr double getFrequencyParameterMax(const double sample_rate) noexcept {
        if (sample_rate >= 40000.0 && sample_rate < 50000.0) {
            return 30000.0;
        }
        return 0.49964 * sample_rate;
    }

    inline constexpr std::size_t getMediumFFTOrder(const double sample_rate) noexcept {
        if (sample_rate < 12500.0) {
            return 10;
        }
        if (sample_rate < 25000.0) {
            return 11;
        }
        if (sample_rate < 50000.0) {
            return 12;
        }
        if (sample_rate < 100000.0) {
            return 13;
        }
        if (sample_rate < 200000.0) {
            return 14;
        }
        if (sample_rate < 400000.0) {
            return 15;
        }
        return 16;
    }
}
