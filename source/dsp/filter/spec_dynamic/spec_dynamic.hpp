// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include "spec_response.hpp"
#include "spec_follower.hpp"

namespace zldsp::filter {
    namespace hn = hwy::HWY_NAMESPACE;

    template <typename FloatType>
    class SpecDynamic {
    public:
        explicit SpecDynamic() = default;

        void prepare(const size_t fft_size) {
            states_.resize(fft_size);
        }

        template <bool to_add = true, bool bypass = false>
        void process(FloatType* HWY_RESTRICT side_log_sqr,
                     FloatType* HWY_RESTRICT dynamic_db,
                     SpecResponse<FloatType>& response,
                     SpecFollower<FloatType>& follower) {
            FloatType* HWY_RESTRICT diffs{response.getDiffResponse().data()};
            FloatType* HWY_RESTRICT attacks{follower.getAttack().data()};
            FloatType* HWY_RESTRICT releases{follower.getRelease().data()};
            FloatType* HWY_RESTRICT states{states_.data()};
            static constexpr hn::ScalableTag<FloatType> d;
            static constexpr size_t lanes = hn::MaxLanes(d);

            const auto v_coeffa = hn::Set(d, coeff_a_);
            const auto v_coeffb = hn::Set(d, coeff_b_);
            const auto v_zero = hn::Set(d, static_cast<FloatType>(0));
            const auto v_one = hn::Set(d, static_cast<FloatType>(1));
            const auto i_start = response.getDiffStartIdx();
            const auto i_stop = response.getDiffEndIdx();
            for (size_t i = i_start; i < i_stop; i+= lanes) {
                // threshold & knee
                const auto v_side = hn::Load(d, side_log_sqr + i);
                auto v_x = hn::MulAdd(v_side, v_coeffa, v_coeffb);
                v_x = hn::Clamp(v_x, v_zero, v_one);
                v_x = hn::Mul(v_x, v_x);
                // attach & release
                const auto v_release = hn::Load(d, releases + i);
                const auto v_attack = hn::Load(d, attacks + i);
                auto v_state = hn::Load(d, states + (i << 1));
                auto v_y = hn::Load(d, states + (i << 1) + lanes);
                v_state = hn::Max(v_x, hn::MulAdd(v_release, hn::Sub(v_state, v_x), v_x));
                v_y = hn::MulAdd(v_attack, hn::Sub(v_y,v_state), v_state);
                hn::Store(v_state, d, states + (i << 1));
                hn::Store(v_y, d, states + (i << 1) + lanes);
                // calculate diff
                if constexpr (!bypass) {
                    const auto v_diff = hn::Mul(hn::Load(d, diffs + i), v_y);
                    if constexpr (to_add) {
                        const auto v_dyn = hn::Load(d, dynamic_db + i);
                        hn::Store(hn::Add(v_diff, v_dyn), d, dynamic_db + i);
                    } else {
                        hn::Store(v_diff, d, dynamic_db + i);
                    }
                }
            }
            if constexpr (to_add && bypass) {
                std::fill(dynamic_db + i_start, dynamic_db + i_stop, static_cast<FloatType>(0));
            }
        }

        void updateTK(const double threshold, const double knee) {
            const auto safe_knee = std::max(knee, 0.001);
            coeff_a_ = static_cast<FloatType>(kLogScale / safe_knee);
            coeff_b_ = static_cast<FloatType>((safe_knee - threshold) / (2.0 * safe_knee));
        }

    private:
        FloatType coeff_a_, coeff_b_;
        vector::aligned_vector<FloatType> states_;
        static constexpr double kLogScale = 2.1714724095162588; // 5.0 / ln(10.0)
    };
}
