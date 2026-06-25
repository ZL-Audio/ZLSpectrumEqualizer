// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#include "controller.hpp"

namespace zlp {
    Controller::Controller(juce::AudioProcessor& p) :
        p_ref_(p) {
    }

    void Controller::prepare(const double sample_rate, const size_t max_num_samples) {

    }

    void Controller::processStereoStatic() {
        zldsp::vector::multiply(fft_ins_[0].data(), window1_.data(), fft_size_);
        zldsp::vector::multiply(fft_ins_[1].data(), window1_.data(), fft_size_);

        fft_->forward(fft_ins_[0].data(), {fft_out_reals_[0].data(), fft_out_imags_[0].data()});
        fft_->forward(fft_ins_[1].data(), {fft_out_reals_[1].data(), fft_out_imags_[1].data()});

        auto* HWY_RESTRICT l_real_ptr = fft_out_reals_[0].data();
        auto* HWY_RESTRICT l_imag_ptr = fft_out_imags_[0].data();
        auto* HWY_RESTRICT r_real_ptr = fft_out_reals_[1].data();
        auto* HWY_RESTRICT r_imag_ptr = fft_out_imags_[1].data();

        const auto* HWY_RESTRICT mag_ptr = l_data_.static_response_linear.data();

        for (size_t i = 0; i < num_bin_effective_; i += lanes) {
            const auto v_mag = hn::Load(d, mag_ptr + i);
            {
                const auto vl_real = hn::Load(d, l_real_ptr + i);
                hn::Store(hn::Mul(vl_real, v_mag), d, l_real_ptr + i);
                const auto vl_imag = hn::Load(d, l_imag_ptr + i);
                hn::Store(hn::Mul(vl_imag, v_mag), d, l_imag_ptr + i);
            }
            {
                const auto vr_real = hn::Load(d, r_real_ptr + i);
                hn::Store(hn::Mul(vr_real, v_mag), d, r_real_ptr + i);
                const auto vr_imag = hn::Load(d, r_imag_ptr + i);
                hn::Store(hn::Mul(vr_imag, v_mag), d, r_imag_ptr + i);
            }
        }

        fft_->backward({fft_out_reals_[0].data(), fft_out_imags_[0].data()}, fft_ins_[0].data());
        fft_->backward({fft_out_reals_[1].data(), fft_out_imags_[1].data()}, fft_ins_[1].data());

        zldsp::vector::multiply(fft_ins_[0].data(), window2_.data(), fft_size_);
        zldsp::vector::multiply(fft_ins_[1].data(), window2_.data(), fft_size_);
    }

    void Controller::processLRStatic() {
        if (l_data_.is_static_active_) {
            zldsp::vector::multiply(fft_ins_[0].data(), window1_.data(), fft_size_);
            fft_->forward(fft_ins_[0].data(), {fft_out_reals_[0].data(), fft_out_imags_[0].data()});
            auto* HWY_RESTRICT l_real_ptr = fft_out_reals_[0].data();
            auto* HWY_RESTRICT l_imag_ptr = fft_out_imags_[0].data();
            const auto* HWY_RESTRICT l_mag_ptr = l_data_.static_response_linear.data();
            for (size_t i = 0; i < num_bin_effective_; i += lanes) {
                const auto vl_mag = hn::Load(d, l_mag_ptr + i);
                const auto vl_real = hn::Load(d, l_real_ptr + i);
                hn::Store(hn::Mul(vl_real, vl_mag), d, l_real_ptr + i);
                const auto vl_imag = hn::Load(d, l_imag_ptr + i);
                hn::Store(hn::Mul(vl_imag, vl_mag), d, l_imag_ptr + i);
            }
            fft_->backward({fft_out_reals_[0].data(), fft_out_imags_[0].data()}, fft_ins_[0].data());
            zldsp::vector::multiply(fft_ins_[0].data(), window2_.data(), fft_size_);
        } else {
            zldsp::vector::multiply(fft_ins_[0].data(), window_bypass_.data(), fft_size_);
        }
        if (r_data_.is_static_active_) {
            zldsp::vector::multiply(fft_ins_[1].data(), window1_.data(), fft_size_);
            fft_->forward(fft_ins_[1].data(), {fft_out_reals_[1].data(), fft_out_imags_[1].data()});
            auto* HWY_RESTRICT r_real_ptr = fft_out_reals_[1].data();
            auto* HWY_RESTRICT r_imag_ptr = fft_out_imags_[1].data();
            const auto* HWY_RESTRICT r_mag_ptr = r_data_.static_response_linear.data();
            for (size_t i = 0; i < num_bin_effective_; i += lanes) {
                const auto vr_mag = hn::Load(d, r_mag_ptr + i);
                const auto vr_real = hn::Load(d, r_real_ptr + i);
                hn::Store(hn::Mul(vr_real, vr_mag), d, r_real_ptr + i);
                const auto vr_imag = hn::Load(d, r_imag_ptr + i);
                hn::Store(hn::Mul(vr_imag, vr_mag), d, r_imag_ptr + i);
            }
            fft_->backward({fft_out_reals_[1].data(), fft_out_imags_[1].data()}, fft_ins_[1].data());
            zldsp::vector::multiply(fft_ins_[1].data(), window2_.data(), fft_size_);
        } else {
            zldsp::vector::multiply(fft_ins_[1].data(), window_bypass_.data(), fft_size_);
        }
    }

    void Controller::processMSStatic() {
        zldsp::splitter::InplaceMSSplitter<float>::split(fft_ins_[0].data(), fft_ins_[1].data(), fft_size_);
        if (m_data_.is_static_active_) {
            zldsp::vector::multiply(fft_ins_[0].data(), window1_.data(), fft_size_);
            fft_->forward(fft_ins_[0].data(), {fft_out_reals_[0].data(), fft_out_imags_[0].data()});
            auto* HWY_RESTRICT m_real_ptr = fft_out_reals_[0].data();
            auto* HWY_RESTRICT m_imag_ptr = fft_out_imags_[0].data();
            const auto* HWY_RESTRICT m_mag_ptr = m_data_.static_response_linear.data();
            for (size_t i = 0; i < num_bin_effective_; i += lanes) {
                const auto vl_mag = hn::Load(d, m_mag_ptr + i);
                const auto vl_real = hn::Load(d, m_real_ptr + i);
                hn::Store(hn::Mul(vl_real, vl_mag), d, m_real_ptr + i);
                const auto vl_imag = hn::Load(d, m_imag_ptr + i);
                hn::Store(hn::Mul(vl_imag, vl_mag), d, m_imag_ptr + i);
            }
            fft_->backward({fft_out_reals_[0].data(), fft_out_imags_[0].data()}, fft_ins_[0].data());
            zldsp::vector::multiply(fft_ins_[0].data(), window2_.data(), fft_size_);
        } else {
            zldsp::vector::multiply(fft_ins_[0].data(), window_bypass_.data(), fft_size_);
        }
        if (s_data_.is_static_active_) {
            zldsp::vector::multiply(fft_ins_[1].data(), window1_.data(), fft_size_);
            fft_->forward(fft_ins_[1].data(), {fft_out_reals_[1].data(), fft_out_imags_[1].data()});
            auto* HWY_RESTRICT s_real_ptr = fft_out_reals_[1].data();
            auto* HWY_RESTRICT s_imag_ptr = fft_out_imags_[1].data();
            const auto* HWY_RESTRICT s_mag_ptr = s_data_.static_response_linear.data();
            for (size_t i = 0; i < num_bin_effective_; i += lanes) {
                const auto vr_mag = hn::Load(d, s_mag_ptr + i);
                const auto vr_real = hn::Load(d, s_real_ptr + i);
                hn::Store(hn::Mul(vr_real, vr_mag), d, s_real_ptr + i);
                const auto vr_imag = hn::Load(d, s_imag_ptr + i);
                hn::Store(hn::Mul(vr_imag, vr_mag), d, s_imag_ptr + i);
            }
            fft_->backward({fft_out_reals_[1].data(), fft_out_imags_[1].data()}, fft_ins_[1].data());
            zldsp::vector::multiply(fft_ins_[1].data(), window2_.data(), fft_size_);
        } else {
            zldsp::vector::multiply(fft_ins_[1].data(), window_bypass_.data(), fft_size_);
        }
        zldsp::splitter::InplaceMSSplitter<float>::combine(fft_ins_[0].data(), fft_ins_[1].data(), fft_size_);
    }

    void Controller::processLRMSStatic() {
        zldsp::vector::multiply(fft_ins_[0].data(), window1_.data(), fft_size_);
        zldsp::vector::multiply(fft_ins_[1].data(), window1_.data(), fft_size_);

        fft_->forward(fft_ins_[0].data(), {fft_out_reals_[0].data(), fft_out_imags_[0].data()});
        fft_->forward(fft_ins_[1].data(), {fft_out_reals_[1].data(), fft_out_imags_[1].data()});

        auto* HWY_RESTRICT l_real_ptr = fft_out_reals_[0].data();
        auto* HWY_RESTRICT l_imag_ptr = fft_out_imags_[0].data();
        auto* HWY_RESTRICT r_real_ptr = fft_out_reals_[1].data();
        auto* HWY_RESTRICT r_imag_ptr = fft_out_imags_[1].data();

        const auto* HWY_RESTRICT l_mag_ptr = l_data_.static_response_linear.data();
        const auto* HWY_RESTRICT r_mag_ptr = r_data_.static_response_linear.data();
        const auto* HWY_RESTRICT m_mag_ptr = m_data_.static_response_linear.data();
        const auto* HWY_RESTRICT s_mag_ptr = s_data_.static_response_linear.data();

        const auto v_half = hn::Set(d, 0.5f);

        for (size_t i = 0; i < num_bin_effective_; i += lanes) {
            auto vl_real = hn::Load(d, l_real_ptr + i);
            auto vr_real = hn::Load(d, r_real_ptr + i);
            auto vl_imag = hn::Load(d, l_imag_ptr + i);
            auto vr_imag = hn::Load(d, r_imag_ptr + i);

            const auto vl_mag = hn::Load(d, l_mag_ptr + i);
            const auto vr_mag = hn::Load(d, r_mag_ptr + i);

            vl_real = hn::Mul(vl_real, vl_mag);
            vl_imag = hn::Mul(vl_imag, vl_mag);
            vr_real = hn::Mul(vr_real, vr_mag);
            vr_imag = hn::Mul(vr_imag, vr_mag);

            auto vm_real = hn::Mul(hn::Add(vl_real, vr_real), v_half);
            auto vs_real = hn::Mul(hn::Sub(vl_real, vr_real), v_half);
            auto vm_imag = hn::Mul(hn::Add(vl_imag, vr_imag), v_half);
            auto vs_imag = hn::Mul(hn::Sub(vl_imag, vr_imag), v_half);

            const auto vm_mag = hn::Load(d, m_mag_ptr + i);
            const auto vs_mag = hn::Load(d, s_mag_ptr + i);

            vm_real = hn::Mul(vm_real, vm_mag);
            vm_imag = hn::Mul(vm_imag, vm_mag);
            vs_real = hn::Mul(vs_real, vs_mag);
            vs_imag = hn::Mul(vs_imag, vs_mag);

            hn::Store(hn::Add(vm_real, vs_real), d, l_real_ptr + i);
            hn::Store(hn::Sub(vm_real, vs_real), d, r_real_ptr + i);
            hn::Store(hn::Add(vm_imag, vs_imag), d, l_imag_ptr + i);
            hn::Store(hn::Sub(vm_imag, vs_imag), d, r_imag_ptr + i);
        }

        fft_->backward({fft_out_reals_[0].data(), fft_out_imags_[0].data()}, fft_ins_[0].data());
        fft_->backward({fft_out_reals_[1].data(), fft_out_imags_[1].data()}, fft_ins_[1].data());

        zldsp::vector::multiply(fft_ins_[0].data(), window2_.data(), fft_size_);
        zldsp::vector::multiply(fft_ins_[1].data(), window2_.data(), fft_size_);
    }
}
