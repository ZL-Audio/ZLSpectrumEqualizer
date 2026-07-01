// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include "../../vector/vector.hpp"
#include "../ideal_filter/ideal.hpp"

namespace zldsp::filter {
    namespace hn = hwy::HWY_NAMESPACE;

    template <typename FloatType>
    class SpecResponse {
    public:
        explicit SpecResponse() = default;

        void prepare(const size_t fft_size) {
            const auto spec_size = fft_size / 2;
            base_response_.resize(spec_size);
            diff_response_.resize(spec_size);
            diff_start_idx_ = 0;
            diff_size_ = 0;
        }

        template <size_t kFilterSize>
        void updateBaseResponse(zldsp::filter::FilterParameters paras,
                                zldsp::filter::Ideal<FloatType, kFilterSize>& ideal,
                                vector::aligned_vector<FloatType>& ws) {
            ideal.forceUpdate(paras);
            ideal.updateMagnitudeSquare(ws, base_response_);
            static constexpr hn::ScalableTag<FloatType> d;
            static constexpr size_t lanes = hn::MaxLanes(d);

            const auto v_min = hn::Set(d, kLogSqrMin);
            for (size_t i = 0; i < base_response_.size(); i += lanes) {
                const auto v = hn::Load(d, base_response_.data() + i);
                const auto v_log = hn::Log(hn::Max(v, v_min));
                hn::Store(v_log, d, base_response_.data() + i);
            }
        }

        template <size_t kFilterSize>
        void updateDiffResponse(zldsp::filter::FilterParameters paras,
                                zldsp::filter::Ideal<FloatType, kFilterSize>& ideal,
                                vector::aligned_vector<FloatType>& ws) {
            ideal.forceUpdate(paras);
            ideal.updateMagnitudeSquare(ws, diff_response_);
            static constexpr hn::ScalableTag<FloatType> d;
            static constexpr size_t lanes = hn::MaxLanes(d);

            const auto v_min = hn::Set(d, kLogSqrMin);
            size_t i = 0;
            for (; i < base_response_.size(); i += lanes) {
                const auto v = hn::Load(d, diff_response_.data() + i);
                const auto v_log = hn::Log(d, hn::Max(v, v_min));
                const auto r_log = hn::Load(d, base_response_.data() + i);
                const auto diff = hn::Sub(v_log, r_log);
                if (hn::ReduceMax(hn::Abs(diff)) > kDiffMin) {
                    break;
                }
            }
            diff_start_idx_ = i;
            for (; i < base_response_.size(); i += lanes) {
                const auto v = hn::Load(d, diff_response_.data() + i);
                const auto v_log = hn::Log(d, hn::Max(v, v_min));
                const auto r_log = hn::Load(d, base_response_.data() + i);
                const auto diff = hn::Sub(v_log, r_log);
                if (hn::ReduceMax(hn::Abs(diff)) < kDiffMin) {
                    break;
                }
                hn::Store(diff, d, diff_response_.data() + i);
            }
            diff_size_ = i - diff_start_idx_;
            if (diff_size_ == 0) {
                diff_start_idx_ = 0;
            }
        }

        auto& getBaseResponse() {
            return base_response_;
        }

        auto& getDiffResponse() {
            return diff_response_;
        }

        [[nodiscard]] auto getDiffStartIdx() const {
            return diff_start_idx_;
        }

        [[nodiscard]] auto getDiffEndIdx() const {
            return diff_end_idx_;
        }

        [[nodiscard]] auto getDiffSize() const {
            return diff_size_;
        }

    private:
        vector::aligned_vector<FloatType> base_response_;
        vector::aligned_vector<FloatType> diff_response_;
        size_t diff_start_idx_{0}, diff_end_idx_{0}, diff_size_{0};

        static constexpr auto kLogSqrMin = static_cast<FloatType>(1e-24);
        static constexpr auto kDiffMin = static_cast<FloatType>(1e-2);
    };
}
