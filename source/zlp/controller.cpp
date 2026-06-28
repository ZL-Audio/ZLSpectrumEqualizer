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

    void Controller::processSideLR() {
        if (l_data_.is_side_required_ || stereo_data_.is_side_required_) {
            zldsp::vector::multiply(fft_ins_[2].data(), window1_.data(), fft_size_);
            fft_->forward_sqr_mag(fft_ins_[2].data(), l_data_.fft_side_abs_sqr_.data());
        }
        if (r_data_.is_side_required_ || stereo_data_.is_side_required_) {
            zldsp::vector::multiply(fft_ins_[3].data(), window1_.data(), fft_size_);
            fft_->forward_sqr_mag(fft_ins_[3].data(), r_data_.fft_side_abs_sqr_.data());
        }
        if (stereo_data_.is_side_required_) {
            auto* HWY_RESTRICT stereo_abs_sqr = stereo_data_.fft_side_abs_sqr_.data();
            const auto* HWY_RESTRICT l_abs_sqr = l_data_.fft_side_abs_sqr_.data();
            const auto* HWY_RESTRICT r_abs_sqr = r_data_.fft_side_abs_sqr_.data();
            for (size_t i = stereo_data_.smooth_bounds.pass1_start;
                 i < stereo_data_.smooth_bounds.pass1_end;
                 i += lanes) {
                const auto l_v = hn::Load(d, l_abs_sqr + i);
                const auto r_v = hn::Load(d, r_abs_sqr + i);
                hn::Store(hn::Add(l_v, r_v), d, stereo_abs_sqr + i);
            }
            spec_smoother_.smoothRange(stereo_data_.fft_side_abs_sqr_, stereo_data_.smooth_bounds);
        }
        if (l_data_.is_side_required_) {
            spec_smoother_.smoothRange(l_data_.fft_side_abs_sqr_, l_data_.smooth_bounds);
        }
        if (r_data_.is_side_required_) {
            spec_smoother_.smoothRange(r_data_.fft_side_abs_sqr_, r_data_.smooth_bounds);
        }
    }

    void Controller::processSideMS() {
        zldsp::splitter::InplaceMSSplitter<float>::split(fft_ins_[2].data(), fft_ins_[3].data(), fft_size_);
        if (m_data_.is_side_required_ || stereo_data_.is_side_required_) {
            zldsp::vector::multiply(fft_ins_[2].data(), window1_.data(), fft_size_);
            fft_->forward_sqr_mag(fft_ins_[2].data(), m_data_.fft_side_abs_sqr_.data());
        }
        if (s_data_.is_side_required_ || stereo_data_.is_side_required_) {
            zldsp::vector::multiply(fft_ins_[3].data(), window1_.data(), fft_size_);
            fft_->forward_sqr_mag(fft_ins_[3].data(), s_data_.fft_side_abs_sqr_.data());
        }
        if (stereo_data_.is_side_required_) {
            auto* HWY_RESTRICT stereo_abs_sqr = stereo_data_.fft_side_abs_sqr_.data();
            const auto* HWY_RESTRICT m_abs_sqr = m_data_.fft_side_abs_sqr_.data();
            const auto* HWY_RESTRICT s_abs_sqr = s_data_.fft_side_abs_sqr_.data();
            for (size_t i = stereo_data_.smooth_bounds.pass1_start;
                 i < stereo_data_.smooth_bounds.pass1_end;
                 i += lanes) {
                const auto m_v = hn::Load(d, m_abs_sqr + i);
                const auto s_v = hn::Load(d, s_abs_sqr + i);
                hn::Store(hn::Add(m_v, s_v), d, stereo_abs_sqr + i);
            }
            spec_smoother_.smoothRange(stereo_data_.fft_side_abs_sqr_, stereo_data_.smooth_bounds);
        }
        if (m_data_.is_side_required_) {
            spec_smoother_.smoothRange(m_data_.fft_side_abs_sqr_, m_data_.smooth_bounds);
        }
        if (s_data_.is_side_required_) {
            spec_smoother_.smoothRange(s_data_.fft_side_abs_sqr_, s_data_.smooth_bounds);
        }
    }

    void Controller::processSideLRMS() {
        auto* HWY_RESTRICT l_real_ptr = fft_out_reals_[0].data();
        auto* HWY_RESTRICT l_imag_ptr = fft_out_imags_[0].data();
        auto* HWY_RESTRICT r_real_ptr = fft_out_reals_[1].data();
        auto* HWY_RESTRICT r_imag_ptr = fft_out_imags_[1].data();

        zldsp::vector::multiply(fft_ins_[2].data(), window1_.data(), fft_size_);
        zldsp::vector::multiply(fft_ins_[3].data(), window1_.data(), fft_size_);
        fft_->forward(fft_ins_[2].data(), {l_real_ptr, l_imag_ptr});
        fft_->forward(fft_ins_[3].data(), {r_real_ptr, r_imag_ptr});

        if (l_data_.is_side_required_) {
            auto* HWY_RESTRICT l_abs_sqr = l_data_.fft_side_abs_sqr_.data();
            for (size_t i = l_data_.smooth_bounds.pass1_start;
                 i < l_data_.smooth_bounds.pass1_end;
                 i += lanes) {
                const auto real_v = hn::Load(d, l_real_ptr + i);
                const auto imag_v = hn::Load(d, l_imag_ptr + i);

                const auto abs_sqr_v = hn::MulAdd(real_v, real_v, hn::Mul(imag_v, imag_v));
                hn::Store(abs_sqr_v, d, l_abs_sqr + i);
            }
            spec_smoother_.smoothRange(l_data_.fft_side_abs_sqr_, l_data_.smooth_bounds);
        }
        if (r_data_.is_side_required_) {
            auto* HWY_RESTRICT r_abs_sqr = r_data_.fft_side_abs_sqr_.data();
            for (size_t i = r_data_.smooth_bounds.pass1_start; i < r_data_.smooth_bounds.pass1_end; i += lanes) {
                const auto real_v = hn::Load(d, r_real_ptr + i);
                const auto imag_v = hn::Load(d, r_imag_ptr + i);

                const auto abs_sqr_v = hn::MulAdd(real_v, real_v, hn::Mul(imag_v, imag_v));
                hn::Store(abs_sqr_v, d, r_abs_sqr + i);
            }
            spec_smoother_.smoothRange(r_data_.fft_side_abs_sqr_, r_data_.smooth_bounds);
        }
        if (stereo_data_.is_side_required_) {
            auto* HWY_RESTRICT st_abs_sqr = stereo_data_.fft_side_abs_sqr_.data();
            for (size_t i = stereo_data_.smooth_bounds.pass1_start; i < stereo_data_.smooth_bounds.pass1_end; i +=
                 lanes) {
                const auto l_real_v = hn::Load(d, l_real_ptr + i);
                const auto l_imag_v = hn::Load(d, l_imag_ptr + i);
                const auto l_abs_sqr_v = hn::MulAdd(l_real_v, l_real_v, hn::Mul(l_imag_v, l_imag_v));

                const auto r_real_v = hn::Load(d, r_real_ptr + i);
                const auto r_imag_v = hn::Load(d, r_imag_ptr + i);
                const auto r_abs_sqr_v = hn::MulAdd(r_real_v, r_real_v, hn::Mul(r_imag_v, r_imag_v));

                const auto abs_sqr_v = hn::Add(l_abs_sqr_v, r_abs_sqr_v);
                hn::Store(abs_sqr_v, d, st_abs_sqr + i);
            }
            spec_smoother_.smoothRange(stereo_data_.fft_side_abs_sqr_, stereo_data_.smooth_bounds);
        }

        if (m_data_.is_side_required_) {
            auto* HWY_RESTRICT m_abs_sqr = m_data_.fft_side_abs_sqr_.data();
            const auto quarter_v = hn::Set(d, 0.25f);

            for (size_t i = m_data_.smooth_bounds.pass1_start; i < m_data_.smooth_bounds.pass1_end; i += lanes) {
                const auto l_real_v = hn::Load(d, l_real_ptr + i);
                const auto r_real_v = hn::Load(d, r_real_ptr + i);
                const auto m_real_v = hn::Add(l_real_v, r_real_v);

                const auto l_imag_v = hn::Load(d, l_imag_ptr + i);
                const auto r_imag_v = hn::Load(d, r_imag_ptr + i);
                const auto m_imag_v = hn::Add(l_imag_v, r_imag_v);

                const auto abs_sqr_v = hn::MulAdd(m_real_v, m_real_v, hn::Mul(m_imag_v, m_imag_v));
                hn::Store(hn::Mul(abs_sqr_v, quarter_v), d, m_abs_sqr + i);
            }
            spec_smoother_.smoothRange(m_data_.fft_side_abs_sqr_, m_data_.smooth_bounds);
        }

        if (s_data_.is_side_required_) {
            auto* HWY_RESTRICT s_abs_sqr = s_data_.fft_side_abs_sqr_.data();
            const auto quarter_v = hn::Set(d, 0.25f);

            for (size_t i = s_data_.smooth_bounds.pass1_start; i < s_data_.smooth_bounds.pass1_end; i += lanes) {
                const auto l_real_v = hn::Load(d, l_real_ptr + i);
                const auto r_real_v = hn::Load(d, r_real_ptr + i);
                const auto s_real_v = hn::Sub(l_real_v, r_real_v);

                const auto l_imag_v = hn::Load(d, l_imag_ptr + i);
                const auto r_imag_v = hn::Load(d, r_imag_ptr + i);
                const auto s_imag_v = hn::Sub(l_imag_v, r_imag_v);

                const auto abs_sqr_v = hn::MulAdd(s_real_v, s_real_v, hn::Mul(s_imag_v, s_imag_v));
                hn::Store(hn::Mul(abs_sqr_v, quarter_v), d, s_abs_sqr + i);
            }
            spec_smoother_.smoothRange(s_data_.fft_side_abs_sqr_, s_data_.smooth_bounds);
        }
    }
}
