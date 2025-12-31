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
#include "../vector/kfr_import.hpp"

namespace zldsp::spectrum {
    template <typename FloatType>
    class DynamicSpectrum {
    public:
        DynamicSpectrum() = default;

        void prepare(const size_t max_spectrum_size) {
            diff_db_.reserve(max_spectrum_size);
            dynamic_diff_db_.reserve(max_spectrum_size);
            temp_db_.reserve(max_spectrum_size);
        }

        void setSpectrumSize(const size_t spectrum_size) {
            diff_db_.resize(spectrum_size);
            dynamic_diff_db_.resize(spectrum_size);
            temp_db_.resize(spectrum_size);

            std::fill(dynamic_diff_db_.begin(), dynamic_diff_db_.end(), FloatType(0));
        }

        void setDiffDB(std::span<FloatType> diff_db) {
            left_ = diff_db.size();
            for (size_t i = 0; i < diff_db.size(); ++i) {
                if (std::abs(diff_db[i]) > FloatType(0.01)) {
                    left_ = i;
                    break;
                }
            }
            length_ = 0;
            for (size_t i = left_ + 1; i < diff_db.size(); ++i) {
                if (std::abs(diff_db[i]) < FloatType(0.01)) {
                    length_ = i - left_;
                    break;
                }
            }
            if (length_ == 0) {
                left_ = 0;
            }

            auto v0 = kfr::make_univector(diff_db.data() + left_, length_);
            auto v1 = kfr::make_univector(diff_db_.data() + left_, length_);
            v1 = 0.05f * v0;
        }

        void setThreshold(const FloatType threshold) {
            threshold_ = threshold;
            updateTK();
        }

        void setKnee(const FloatType knee) {
            knee_ = knee;
            updateTK();
        }

        void setAttack(const FloatType attack) {
            attack_ = attack;
        }

        void setRelease(const FloatType release) {
            release_ = release;
        }

        void process(std::span<FloatType> side_spectrum_sqr_log) {
            if (length_ == 0) {
                return;
            }
            auto side_v = kfr::make_univector(side_spectrum_sqr_log.data() + left_, length_);
            auto temp_side_v = kfr::make_univector(temp_db_.data() + left_, length_);
            auto diff_v = kfr::make_univector(diff_db_.data() + left_, length_);
            temp_side_v = diff_v * kfr::sqr(kfr::clamp((side_v - low_sqr_) * slope_sqr_, FloatType(0), FloatType(1)));

            for (size_t i = left_; i < left_ + length_; i++) {
                const auto x = temp_db_[i];
                const auto y = dynamic_diff_db_[i];
                dynamic_diff_db_[i] = x > y ? attack_ * (y - x) + x : release_ * (y - x) + x;
            }
        }

        void update(std::span<FloatType> main_diff_db) {
            auto main_diff_v = kfr::make_univector(main_diff_db.data() + left_, length_);
            auto dynamic_diff_db_v = kfr::make_univector(dynamic_diff_db_.data() + left_, length_);
            main_diff_v = main_diff_v + dynamic_diff_db_v;
        }

    private:
        std::vector<FloatType> diff_db_{}, dynamic_diff_db_{};
        std::vector<FloatType> temp_db_{};
        size_t left_{}, length_{};

        FloatType threshold_{}, knee_{};
        FloatType low_sqr_{}, slope_sqr_{};

        FloatType attack_{}, release_{};

        void updateTK() {
            const auto low = threshold_ - knee_;
            const auto slope = static_cast<FloatType>(0.5) / knee_;

            low_sqr_ = low / static_cast<FloatType>(10);
            slope_sqr_ = slope * static_cast<FloatType>(10);
        }
    };
}