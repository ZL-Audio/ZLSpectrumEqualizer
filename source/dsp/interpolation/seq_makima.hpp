// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <vector>

namespace zldsp::interpolation {
    /**
     * modified Akima spline interpolation with increasing input/output X
     * @tparam FloatType the float type of input/output
     */
    template <typename FloatType>
    class SeqMakima {
    public:
        SeqMakima() = default;

        explicit SeqMakima(const size_t max_input_size) {
            reserve(max_input_size);
        }

        explicit SeqMakima(const FloatType* x, const FloatType* y, const size_t point_num,
                           const FloatType left_derivative = 0, const FloatType right_derivative = 0) {
            prepare(x, y, point_num, left_derivative, right_derivative);
        }

        void reserve(const size_t max_input_size) {
            if (derivatives_.size() < max_input_size) {
                derivatives_.resize(max_input_size);
                deltas_.resize(max_input_size > 0 ? max_input_size - 1 : 0);
            }
        }

        void prepare() {
            if (input_size_ < 2) {
                return;
            }
            if (derivatives_.size() < input_size_) {
                derivatives_.resize(input_size_);
                deltas_.resize(input_size_ - 1);
            }
            for (size_t i = 0; i < input_size_ - 1; ++i) {
                deltas_[i] = (ys_[i + 1] - ys_[i]) / (xs_[i + 1] - xs_[i]);
            }
            const auto left_delta = FloatType(2) * deltas_[0] - deltas_[1];
            derivatives_[0] = left_derivative_;
            derivatives_[input_size_ - 1] = right_derivative_;

            if (input_size_ >= 4) {
                const auto right_delta = FloatType(2) * deltas_[input_size_ - 2] - deltas_[input_size_ - 3];
                derivatives_[1] = calculateD(left_delta, deltas_[0], deltas_[1], deltas_[2]);
                for (size_t i = 2; i < input_size_ - 2; ++i) {
                    derivatives_[i] = calculateD(deltas_[i - 2], deltas_[i - 1], deltas_[i], deltas_[i + 1]);
                }
                derivatives_[input_size_ - 2] = calculateD(deltas_[input_size_ - 4], deltas_[input_size_ - 3], deltas_[input_size_ - 2], right_delta);
            } else if (input_size_ == 3) {
                derivatives_[1] = (deltas_[0] + deltas_[1]) / FloatType(2);
            }
        }

        void prepare(const FloatType* x, const FloatType* y, const size_t point_num,
                     const FloatType left_derivative = 0, const FloatType right_derivative = 0) {
            xs_ = x;
            ys_ = y;
            input_size_ = point_num;
            left_derivative_ = left_derivative;
            right_derivative_ = right_derivative;
            prepare();
        }

        /**
         * evaluate the spline at output X
         * @param x output X pointer
         * @param y output Y pointer
         * @param point_num number of output points
         */
        void eval(const FloatType* x, FloatType* y, const size_t point_num) const {
            if (input_size_ < 2) {
                return;
            }
            size_t current_pos = 0;
            size_t start_idx = 0, end_idx = point_num - 1;
            while (start_idx <= end_idx && x[start_idx] <= xs_[0]) {
                y[start_idx] = ys_[0];
                start_idx += 1;
            }
            while (end_idx > start_idx && x[end_idx] >= xs_[input_size_ - 1]) {
                y[end_idx] = ys_[input_size_ - 1];
                end_idx -= 1;
            }
            for (size_t i = start_idx; i <= end_idx; ++i) {
                while (current_pos + 2 < input_size_ && x[i] >= xs_[current_pos + 1]) {
                    current_pos += 1;
                }
                const auto dx = xs_[current_pos + 1] - xs_[current_pos];
                const auto t = (x[i] - xs_[current_pos]) / dx;
                y[i] = h00(t) * ys_[current_pos] +
                    h10(t) * dx * derivatives_[current_pos] +
                    h01(t) * ys_[current_pos + 1] +
                    h11(t) * dx * derivatives_[current_pos + 1];
            }
        }

    private:
        const FloatType* xs_{nullptr};
        const FloatType* ys_{nullptr};
        size_t input_size_{0};
        std::vector<FloatType> derivatives_{};
        std::vector<FloatType> deltas_{};
        FloatType left_derivative_{0};
        FloatType right_derivative_{0};

        static FloatType h00(FloatType t) {
            return (FloatType(1) + FloatType(2) * t) * (FloatType(1) - t) * (FloatType(1) - t);
        }

        static FloatType h10(FloatType t) {
            return t * (FloatType(1) - t) * (FloatType(1) - t);
        }

        static FloatType h01(FloatType t) {
            return t * t * (FloatType(3) - FloatType(2) * t);
        }

        static FloatType h11(FloatType t) {
            return t * t * (t - FloatType(1));
        }

        static FloatType calculateD(const FloatType& delta0, const FloatType& delta1,
                                    const FloatType& delta2, const FloatType& delta3) {
            const auto w1 = std::abs(delta3 - delta2) + std::abs(delta3 + delta2) * FloatType(0.5);
            const auto w2 = std::abs(delta1 - delta0) + std::abs(delta1 + delta0) * FloatType(0.5);
            const auto w_sum = w1 + w2 + static_cast<FloatType>(1e-24);
            const auto w = w1 / w_sum;
            return w * delta1 + (FloatType(1) - w) * delta2;
        }
    };
}
