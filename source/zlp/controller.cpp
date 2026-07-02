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
    namespace {
        template <size_t Mask>
        void dispatchMask(Controller* instance) {
            instance->processMainImpl<
                (Mask & 1) != 0,
                (Mask & 2) != 0,
                (Mask & 4) != 0,
                (Mask & 8) != 0,
                (Mask & 16) != 0
            >();
        }

        template <size_t... Is>
        constexpr std::array<void (*)(Controller*), sizeof...(Is)> makeDispatchTable(std::index_sequence<Is...>) {
            return {&dispatchMask<Is>...};
        }

        constexpr auto dispatcher = makeDispatchTable(std::make_index_sequence<32>{});
    }

    Controller::Controller(juce::AudioProcessor& p) :
        p_ref_(p) {
    }

    void Controller::prepare(const double sample_rate, const size_t max_num_samples) {

    }

    void Controller::process(const std::array<float*, 4>& buffer, const size_t num_samples, const bool is_bypass) {
        if (fft_ == nullptr) {
            return;
        }
        size_t samples_processed = 0;
        const bool requires_side = (side_status_ != SideStatus::kNotRequired);
        while (samples_processed < num_samples) {
            // copy data to FIFO until reach a hop size or buffer size
            const size_t chunk = std::min(num_samples - samples_processed, fft_hop_size_ - fft_count_);
            const size_t chunk1 = std::min(chunk, fft_size_ - fft_pos_);
            const size_t chunk2 = chunk - chunk1;
            for (size_t chan = 0; chan < 2; ++chan) {
                std::copy_n(buffer[chan] + samples_processed, chunk1,
                            input_fifos_[chan].data() + fft_pos_);
                if (chunk2 > 0) {
                    std::copy_n(buffer[chan] + samples_processed + chunk1, chunk2,
                                input_fifos_[chan].data());
                }
                std::copy_n(output_fifos_[chan].data() + fft_pos_, chunk1,
                            buffer[chan] + samples_processed);
                if (chunk2 > 0) {
                    std::copy_n(output_fifos_[chan].data(), chunk2,
                                buffer[chan] + samples_processed + chunk1);
                }
                std::fill_n(output_fifos_[chan].data() + fft_pos_, chunk1, 0.0f);
                if (chunk2 > 0) {
                    std::fill_n(output_fifos_[chan].data(), chunk2, 0.0f);
                }
            }
            if (requires_side) {
                for (size_t chan = 2; chan < 4; ++chan) {
                    std::copy_n(buffer[chan] + samples_processed, chunk1,
                                input_fifos_[chan].data() + fft_pos_);
                    if (chunk2 > 0) {
                        std::copy_n(buffer[chan] + samples_processed + chunk1, chunk2,
                                    input_fifos_[chan].data());
                    }
                }
            }
            // forward FFT pos
            fft_pos_ += chunk;
            if (fft_pos_ >= fft_size_) {
                fft_pos_ -= fft_size_;
            }
            fft_count_ += chunk;
            // if reach a hop size, process spectrum
            if (fft_count_ >= fft_hop_size_) {
                fft_count_ = 0;
                // copy to FFT working space
                for (size_t chan = 0; chan < 2; ++chan) {
                    std::copy_n(input_fifos_[chan].data() + fft_pos_, fft_size_ - fft_pos_,
                                fft_ins_[chan].data());
                    if (fft_pos_ > 0) {
                        std::copy_n(input_fifos_[chan].data(), fft_pos_,
                                    fft_ins_[chan].data() + fft_size_ - fft_pos_);
                    }
                }
                if (requires_side) {
                    for (size_t chan = 2; chan < 4; ++chan) {
                        std::copy_n(input_fifos_[chan].data() + fft_pos_, fft_size_ - fft_pos_,
                                    fft_ins_[chan].data());
                        if (fft_pos_ > 0) {
                            std::copy_n(input_fifos_[chan].data(), fft_pos_,
                                        fft_ins_[chan].data() + fft_size_ - fft_pos_);
                        }
                    }
                    processSide();
                    processDynamicBands(stereo_data_);
                    processDynamicBands(l_data_);
                    processDynamicBands(r_data_);
                    processDynamicBands(m_data_);
                    processDynamicBands(s_data_);
                }
                processMain(is_bypass);
                // overlap-add
                const size_t range1 = fft_size_ - fft_pos_;
                for (size_t chan = 0; chan < 2; ++chan) {
                    {
                        auto* HWY_RESTRICT out_ptr = output_fifos_[chan].data() + fft_pos_;
                        const auto* HWY_RESTRICT in_ptr = fft_ins_[chan].data();
                        for (size_t k = 0; k < range1; k += lanes) {
                            const auto v_out = hn::Load(d, out_ptr + k);
                            const auto v_in = hn::Load(d, in_ptr + k);
                            hn::Store(hn::Add(v_out, v_in), d, out_ptr + k);
                        }
                    }
                    {
                        auto* HWY_RESTRICT out_ptr = output_fifos_[chan].data();
                        const auto* HWY_RESTRICT in_ptr = fft_ins_[chan].data() + range1;
                        for (size_t k = 0; k < fft_pos_; k += lanes) {
                            const auto v_out = hn::Load(d, out_ptr + k);
                            const auto v_in = hn::Load(d, in_ptr + k);
                            hn::Store(hn::Add(v_out, v_in), d, out_ptr + k);
                        }
                    }
                }
            }
            samples_processed += chunk;
        }
    }

    void Controller::processSide() {
        if (side_status_ == SideStatus::kLR) {
            processSideLR();
        } else if (side_status_ == SideStatus::kMS) {
            processSideMS();
        } else {
            processSideLRMS();
        }
    }

    void Controller::processMain(const bool is_bypass) {
        if (is_bypass) {
            dispatcher[0](this);
        } else {
            dispatcher[dispatch_mask_](this);
        }
    }

    void Controller::processSideLR() {
        if (l_data_.is_side_required || stereo_data_.is_side_required) {
            zldsp::vector::multiply(fft_ins_[2].data(), window1_.data(), fft_size_);
            fft_->forward_sqr_mag(fft_ins_[2].data(), l_data_.fft_side_abs_sqr.data()); // NOLINT
        }
        if (r_data_.is_side_required || stereo_data_.is_side_required) {
            zldsp::vector::multiply(fft_ins_[3].data(), window1_.data(), fft_size_);
            fft_->forward_sqr_mag(fft_ins_[3].data(), r_data_.fft_side_abs_sqr.data()); // NOLINT
        }
        if (stereo_data_.is_side_required) {
            auto* HWY_RESTRICT stereo_abs_sqr = stereo_data_.fft_side_abs_sqr.data();
            const auto* HWY_RESTRICT l_abs_sqr = l_data_.fft_side_abs_sqr.data();
            const auto* HWY_RESTRICT r_abs_sqr = r_data_.fft_side_abs_sqr.data();
            const auto start_idx = stereo_data_.smooth_bounds.pass1_start;
            const auto end_idx = stereo_data_.smooth_bounds.pass1_end;
            for (size_t i = start_idx; i < end_idx; i += lanes) {
                const auto l_v = hn::Load(d, l_abs_sqr + i);
                const auto r_v = hn::Load(d, r_abs_sqr + i);
                hn::Store(hn::Add(l_v, r_v), d, stereo_abs_sqr + i);
            }
            spec_smoother_.smoothRange(stereo_data_.fft_side_abs_sqr, stereo_data_.smooth_bounds);
        }
        if (l_data_.is_side_required) {
            spec_smoother_.smoothRange(l_data_.fft_side_abs_sqr, l_data_.smooth_bounds);
        }
        if (r_data_.is_side_required) {
            spec_smoother_.smoothRange(r_data_.fft_side_abs_sqr, r_data_.smooth_bounds);
        }
    }

    void Controller::processSideMS() {
        zldsp::splitter::InplaceMSSplitter<float>::split(fft_ins_[2].data(), fft_ins_[3].data(), fft_size_);
        if (m_data_.is_side_required || stereo_data_.is_side_required) {
            zldsp::vector::multiply(fft_ins_[2].data(), window1_.data(), fft_size_);
            fft_->forward_sqr_mag(fft_ins_[2].data(), m_data_.fft_side_abs_sqr.data()); // NOLINT
        }
        if (s_data_.is_side_required || stereo_data_.is_side_required) {
            zldsp::vector::multiply(fft_ins_[3].data(), window1_.data(), fft_size_);
            fft_->forward_sqr_mag(fft_ins_[3].data(), s_data_.fft_side_abs_sqr.data()); // NOLINT
        }
        if (stereo_data_.is_side_required) {
            auto* HWY_RESTRICT stereo_abs_sqr = stereo_data_.fft_side_abs_sqr.data();
            const auto* HWY_RESTRICT m_abs_sqr = m_data_.fft_side_abs_sqr.data();
            const auto* HWY_RESTRICT s_abs_sqr = s_data_.fft_side_abs_sqr.data();
            const auto start_idx = stereo_data_.smooth_bounds.pass1_start;
            const auto end_idx = stereo_data_.smooth_bounds.pass1_end;
            for (size_t i = start_idx; i < end_idx; i += lanes) {
                const auto m_v = hn::Load(d, m_abs_sqr + i);
                const auto s_v = hn::Load(d, s_abs_sqr + i);
                hn::Store(hn::Add(m_v, s_v), d, stereo_abs_sqr + i);
            }
            spec_smoother_.smoothRange(stereo_data_.fft_side_abs_sqr, stereo_data_.smooth_bounds);
        }
        if (m_data_.is_side_required) {
            spec_smoother_.smoothRange(m_data_.fft_side_abs_sqr, m_data_.smooth_bounds);
        }
        if (s_data_.is_side_required) {
            spec_smoother_.smoothRange(s_data_.fft_side_abs_sqr, s_data_.smooth_bounds);
        }
    }

    void Controller::processSideLRMS() {
        auto* HWY_RESTRICT l_real_ptr = fft_out_reals_[0].data();
        auto* HWY_RESTRICT l_imag_ptr = fft_out_imags_[0].data();
        auto* HWY_RESTRICT r_real_ptr = fft_out_reals_[1].data();
        auto* HWY_RESTRICT r_imag_ptr = fft_out_imags_[1].data();

        zldsp::vector::multiply(fft_ins_[2].data(), window1_.data(), fft_size_);
        zldsp::vector::multiply(fft_ins_[3].data(), window1_.data(), fft_size_);
        fft_->forward(fft_ins_[2].data(), {l_real_ptr, l_imag_ptr}); // NOLINT
        fft_->forward(fft_ins_[3].data(), {r_real_ptr, r_imag_ptr}); // NOLINT

        if (l_data_.is_side_required) {
            auto* HWY_RESTRICT l_abs_sqr = l_data_.fft_side_abs_sqr.data();
            const auto start_idx = l_data_.smooth_bounds.pass1_start;
            const auto end_idx = l_data_.smooth_bounds.pass1_end;
            for (size_t i = start_idx; i < end_idx; i += lanes) {
                const auto real_v = hn::Load(d, l_real_ptr + i);
                const auto imag_v = hn::Load(d, l_imag_ptr + i);

                const auto abs_sqr_v = hn::MulAdd(real_v, real_v, hn::Mul(imag_v, imag_v));
                hn::Store(abs_sqr_v, d, l_abs_sqr + i);
            }
            spec_smoother_.smoothRange(l_data_.fft_side_abs_sqr, l_data_.smooth_bounds);
        }
        if (r_data_.is_side_required) {
            auto* HWY_RESTRICT r_abs_sqr = r_data_.fft_side_abs_sqr.data();
            const auto start_idx = r_data_.smooth_bounds.pass1_start;
            const auto end_idx = r_data_.smooth_bounds.pass1_end;
            for (size_t i = start_idx; i < end_idx; i += lanes) {
                const auto real_v = hn::Load(d, r_real_ptr + i);
                const auto imag_v = hn::Load(d, r_imag_ptr + i);

                const auto abs_sqr_v = hn::MulAdd(real_v, real_v, hn::Mul(imag_v, imag_v));
                hn::Store(abs_sqr_v, d, r_abs_sqr + i);
            }
            spec_smoother_.smoothRange(r_data_.fft_side_abs_sqr, r_data_.smooth_bounds);
        }
        if (stereo_data_.is_side_required) {
            auto* HWY_RESTRICT st_abs_sqr = stereo_data_.fft_side_abs_sqr.data();
            const auto start_idx = stereo_data_.smooth_bounds.pass1_start;
            const auto end_idx = stereo_data_.smooth_bounds.pass1_end;
            for (size_t i = start_idx; i < end_idx; i += lanes) {
                const auto l_real_v = hn::Load(d, l_real_ptr + i);
                const auto l_imag_v = hn::Load(d, l_imag_ptr + i);
                const auto l_abs_sqr_v = hn::MulAdd(l_real_v, l_real_v, hn::Mul(l_imag_v, l_imag_v));

                const auto r_real_v = hn::Load(d, r_real_ptr + i);
                const auto r_imag_v = hn::Load(d, r_imag_ptr + i);
                const auto r_abs_sqr_v = hn::MulAdd(r_real_v, r_real_v, hn::Mul(r_imag_v, r_imag_v));

                const auto abs_sqr_v = hn::Add(l_abs_sqr_v, r_abs_sqr_v);
                hn::Store(abs_sqr_v, d, st_abs_sqr + i);
            }
            spec_smoother_.smoothRange(stereo_data_.fft_side_abs_sqr, stereo_data_.smooth_bounds);
        }
        if (m_data_.is_side_required) {
            auto* HWY_RESTRICT m_abs_sqr = m_data_.fft_side_abs_sqr.data();
            const auto quarter_v = hn::Set(d, 0.25f);
            const auto start_idx = m_data_.smooth_bounds.pass1_start;
            const auto end_idx = m_data_.smooth_bounds.pass1_end;
            for (size_t i = start_idx; i < end_idx; i += lanes) {
                const auto l_real_v = hn::Load(d, l_real_ptr + i);
                const auto r_real_v = hn::Load(d, r_real_ptr + i);
                const auto m_real_v = hn::Add(l_real_v, r_real_v);

                const auto l_imag_v = hn::Load(d, l_imag_ptr + i);
                const auto r_imag_v = hn::Load(d, r_imag_ptr + i);
                const auto m_imag_v = hn::Add(l_imag_v, r_imag_v);

                const auto abs_sqr_v = hn::MulAdd(m_real_v, m_real_v, hn::Mul(m_imag_v, m_imag_v));
                hn::Store(hn::Mul(abs_sqr_v, quarter_v), d, m_abs_sqr + i);
            }
            spec_smoother_.smoothRange(m_data_.fft_side_abs_sqr, m_data_.smooth_bounds);
        }
        if (s_data_.is_side_required) {
            auto* HWY_RESTRICT s_abs_sqr = s_data_.fft_side_abs_sqr.data();
            const auto quarter_v = hn::Set(d, 0.25f);
            const auto start_idx = s_data_.smooth_bounds.pass1_start;
            const auto end_idx = s_data_.smooth_bounds.pass1_end;
            for (size_t i = start_idx; i < end_idx; i += lanes) {
                const auto l_real_v = hn::Load(d, l_real_ptr + i);
                const auto r_real_v = hn::Load(d, r_real_ptr + i);
                const auto s_real_v = hn::Sub(l_real_v, r_real_v);

                const auto l_imag_v = hn::Load(d, l_imag_ptr + i);
                const auto r_imag_v = hn::Load(d, r_imag_ptr + i);
                const auto s_imag_v = hn::Sub(l_imag_v, r_imag_v);

                const auto abs_sqr_v = hn::MulAdd(s_real_v, s_real_v, hn::Mul(s_imag_v, s_imag_v));
                hn::Store(hn::Mul(abs_sqr_v, quarter_v), d, s_abs_sqr + i);
            }
            spec_smoother_.smoothRange(s_data_.fft_side_abs_sqr, s_data_.smooth_bounds);
        }
    }

    void Controller::processDynamicBands(ChannelData& data) {
        if (data.dynamic_bands.empty()) {
            return;
        }
        const auto start_idx = data.dynamic_start_idx;
        const auto end_idx = data.dynamic_end_idx;
        auto* HWY_RESTRICT side_ptr = data.fft_side_abs_sqr.data();
        auto* HWY_RESTRICT dynamic_ptr = data.dynamic_response.data();
        // convert side from abs sqr to log
        {
            const auto v_min = hn::Set(d, 1e-24f);
            for (size_t i = start_idx; i < end_idx; i += lanes) {
                const auto v = hn::Load(d, side_ptr + i);
                hn::Store(hn::Log(d, hn::Max(v, v_min)), d, side_ptr + i);
            }
        }
        // process each dynamic band
        {
            const auto band = data.dynamic_bands[0];
            spec_dynamic_[band].process<false>(side_ptr, dynamic_ptr,
                                               spec_response_[band], spec_follower_[band]);
            std::fill(dynamic_ptr + spec_response_[band].getDiffEndIdx(), dynamic_ptr + end_idx, 0.f);
        }
        for (size_t i = 1; i < data.dynamic_bands.size(); ++i) {
            const auto band = data.dynamic_bands[i];
            spec_dynamic_[band].process<true>(side_ptr, dynamic_ptr,
                                              spec_response_[band], spec_follower_[band]);
        }
        // convert dynamic response from db to linear
        auto* HWY_RESTRICT static_ptr = data.static_response.data();
        for (size_t i = start_idx; i < end_idx; i += lanes) {
            const auto v_dynamic = hn::Load(d, dynamic_ptr + i);
            const auto v_static = hn::Load(d, static_ptr + i);
            hn::Store(hn::Mul(hn::Exp(d, v_dynamic), v_static), d, dynamic_ptr + i);
        }
    }

    template <bool has_stereo, bool has_l, bool has_r, bool has_m, bool has_s>
    void Controller::processMainImpl() {
        if constexpr (!(has_stereo || has_l || has_r || has_m || has_s)) {
            zldsp::vector::multiply(fft_ins_[0].data(), window_bypass_.data(), fft_size_);
            zldsp::vector::multiply(fft_ins_[1].data(), window_bypass_.data(), fft_size_);
            return;
        }

        zldsp::vector::multiply(fft_ins_[0].data(), window1_.data(), fft_size_);
        zldsp::vector::multiply(fft_ins_[1].data(), window1_.data(), fft_size_);

        fft_->forward(fft_ins_[0].data(), {fft_out_reals_[0].data(), fft_out_imags_[0].data()}); // NOLINT
        fft_->forward(fft_ins_[1].data(), {fft_out_reals_[1].data(), fft_out_imags_[1].data()}); // NOLINT

        auto* HWY_RESTRICT l_real_ptr = fft_out_reals_[0].data();
        auto* HWY_RESTRICT l_imag_ptr = fft_out_imags_[0].data();
        auto* HWY_RESTRICT r_real_ptr = fft_out_reals_[1].data();
        auto* HWY_RESTRICT r_imag_ptr = fft_out_imags_[1].data();

        const float* HWY_RESTRICT stereo_res_ptr = stereo_data_.dynamic_response.data();
        const float* HWY_RESTRICT l_res_ptr = l_data_.dynamic_response.data();
        const float* HWY_RESTRICT r_res_ptr = r_data_.dynamic_response.data();
        const float* HWY_RESTRICT m_res_ptr = m_data_.dynamic_response.data();
        const float* HWY_RESTRICT s_res_ptr = s_data_.dynamic_response.data();

        const auto v_half = hn::Set(d, 0.5f);

        for (size_t i = 0; i < num_bin_effective_; i += lanes) {
            auto vl_real = hn::Load(d, l_real_ptr + i);
            auto vl_imag = hn::Load(d, l_imag_ptr + i);
            auto vr_real = hn::Load(d, r_real_ptr + i);
            auto vr_imag = hn::Load(d, r_imag_ptr + i);

            if constexpr (has_stereo) {
                const auto v_res = hn::Load(d, stereo_res_ptr + i);
                vl_real = hn::Mul(vl_real, v_res);
                vl_imag = hn::Mul(vl_imag, v_res);
                vr_real = hn::Mul(vr_real, v_res);
                vr_imag = hn::Mul(vr_imag, v_res);
            }

            if constexpr (has_l) {
                const auto vl_res = hn::Load(d, l_res_ptr + i);
                vl_real = hn::Mul(vl_real, vl_res);
                vl_imag = hn::Mul(vl_imag, vl_res);
            }

            if constexpr (has_r) {
                const auto vr_res = hn::Load(d, r_res_ptr + i);
                vr_real = hn::Mul(vr_real, vr_res);
                vr_imag = hn::Mul(vr_imag, vr_res);
            }

            if constexpr (has_m || has_s) {
                auto vm_real = hn::Mul(hn::Add(vl_real, vr_real), v_half);
                auto vs_real = hn::Mul(hn::Sub(vl_real, vr_real), v_half);
                auto vm_imag = hn::Mul(hn::Add(vl_imag, vr_imag), v_half);
                auto vs_imag = hn::Mul(hn::Sub(vl_imag, vr_imag), v_half);

                if constexpr (has_m) {
                    const auto vm_res = hn::Load(d, m_res_ptr + i);
                    vm_real = hn::Mul(vm_real, vm_res);
                    vm_imag = hn::Mul(vm_imag, vm_res);
                }

                if constexpr (has_s) {
                    const auto vs_res = hn::Load(d, s_res_ptr + i);
                    vs_real = hn::Mul(vs_real, vs_res);
                    vs_imag = hn::Mul(vs_imag, vs_res);
                }

                vl_real = hn::Add(vm_real, vs_real);
                vr_real = hn::Sub(vm_real, vs_real);
                vl_imag = hn::Add(vm_imag, vs_imag);
                vr_imag = hn::Sub(vm_imag, vs_imag);
            }

            hn::Store(vl_real, d, l_real_ptr + i);
            hn::Store(vl_imag, d, l_imag_ptr + i);
            hn::Store(vr_real, d, r_real_ptr + i);
            hn::Store(vr_imag, d, r_imag_ptr + i);
        }

        fft_->backward({fft_out_reals_[0].data(), fft_out_imags_[0].data()}, fft_ins_[0].data()); // NOLINT
        fft_->backward({fft_out_reals_[1].data(), fft_out_imags_[1].data()}, fft_ins_[1].data()); // NOLINT

        zldsp::vector::multiply(fft_ins_[0].data(), window2_.data(), fft_size_);
        zldsp::vector::multiply(fft_ins_[1].data(), window2_.data(), fft_size_);
    }

    void Controller::handleAsyncUpdate() {

    }
}
