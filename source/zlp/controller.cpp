// Copyright (C) 2025 - zsliu98
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

    void Controller::prepare(double sample_rate, size_t max_num_samples) {

    }

    void Controller::prepareBuffer() {

    }

    void Controller::handleAsyncUpdate() {
        p_ref_.setLatencySamples(latency_.load(std::memory_order_relaxed));
    }

    void Controller::processBuffer(ChannelFFTData& fft_data, size_t num_samples,
                                   float* __restrict ch0, float* __restrict ch1,
                                   float* __restrict side_ch0, float* __restrict side_ch1) {
        size_t buffer_pos = 0;
        while (num_samples > 0) {
            const size_t num_to_process = std::min(num_samples, hop_size_ - fft_data.count_);
            const size_t copy_size_bytes = num_to_process * sizeof(float);

            fifoPushPop<false>(ch0, fft_data.ch0_input_fifo_.data(), fft_data.ch0_output_fifo_.data(),
                               buffer_pos, fft_data.pos_, copy_size_bytes);
            fifoPushPop<false>(ch1, fft_data.ch1_input_fifo_.data(), fft_data.ch1_output_fifo_.data(),
                               buffer_pos, fft_data.pos_, copy_size_bytes);

            if (fft_data.is_ext_side_) {
                fifoPushPop<true>(side_ch0, fft_data.ch0_ext_input_fifo_.data(), fft_data.ch0_ext_output_fifo_.data(),
                                  buffer_pos, fft_data.pos_, copy_size_bytes);
                fifoPushPop<true>(side_ch1, fft_data.ch1_ext_input_fifo_.data(), fft_data.ch1_ext_output_fifo_.data(),
                                  buffer_pos, fft_data.pos_, copy_size_bytes);
            }

            buffer_pos += num_to_process;
            num_samples -= num_to_process;

            fft_data.pos_ += num_to_process;
            fft_data.count_ += num_to_process;

            if (fft_data.count_ >= hop_size_) {
                fft_data.count_ = 0;
                if (fft_data.pos_ >= fft_size_) {
                    fft_data.pos_ = 0;
                }
                processSpectrum(fft_data);
            }
        }
    }

    template <bool is_ext_side>
    void Controller::fifoPushPop(float* __restrict buffer, float* __restrict input_fifo, float* __restrict output_fifo,
                                 const size_t buffer_pos, const size_t fifo_pos, const size_t copy_size) {
        std::memcpy(input_fifo + fifo_pos, buffer + buffer_pos, copy_size);
        std::memcpy(buffer + buffer_pos, output_fifo + fifo_pos, copy_size);
        if constexpr (is_ext_side) {
            // ext side, directly copy from input fifo
            std::memcpy(output_fifo + fifo_pos, input_fifo + fifo_pos, copy_size);
        } else {
            // not ext side, reset otuput fifo to zero for overlap-add
            std::memset(output_fifo + fifo_pos, 0, copy_size);
        }
    }

    template <bool bypass>
    void Controller::processSpectrum(ChannelFFTData& fft_data) {
        if (fft_data.ch0_fft_required_) {
            forwardFFT(fft_data.ch0_input_fifo_, fft_data.pos_, fft_data.ch0_cspectrum_);
        }
        if (fft_data.ch1_fft_required_) {
            forwardFFT(fft_data.ch1_input_fifo_, fft_data.pos_, fft_data.ch1_cspectrum_);
        }
        if (fft_data.ch0_side_required_) {
            if (fft_data.ch0_ext_fft_required_) {
                forwardFFT(fft_data.ch0_ext_input_fifo_, fft_data.pos_, fft_out_);
                fft_data.ch0_side_spectrum_ = kfr::cabssqr(fft_out_);
            } else {
                fft_data.ch0_side_spectrum_ = kfr::cabssqr(fft_data.ch0_cspectrum_);
            }
        }
        if (fft_data.ch1_side_required_) {
            if (fft_data.ch1_ext_fft_required_) {
                forwardFFT(fft_data.ch1_ext_input_fifo_, fft_data.pos_, fft_out_);
                fft_data.ch1_side_spectrum_ = kfr::cabssqr(fft_out_);
            } else {
                fft_data.ch1_side_spectrum_ = kfr::cabssqr(fft_data.ch1_cspectrum_);
            }
        }
        // if stereo is assigned to current channels
        if (fft_data.stereo_dyn_not_off_) {
            // reset stereo dynamic dbs
            auto g = kfr::make_univector(stereo_dyn_gain_.data() + stereo_dyn_left_,
                                         stereo_dyn_length_);
            if constexpr (!bypass) {
                std::ranges::fill(g, 0.0f);
            }
            // calculate stereo side
            auto s = kfr::make_univector(stereo_side_spectrum_.data() + stereo_dyn_left_,
                                         stereo_dyn_length_);
            auto s0 = kfr::make_univector(fft_data.ch0_side_spectrum_.data() + stereo_dyn_left_,
                                          stereo_dyn_length_);
            auto s1 = kfr::make_univector(fft_data.ch1_side_spectrum_.data() + stereo_dyn_left_,
                                          stereo_dyn_length_);
            s = s0 + s1;
            // smooth stereo side
            spectrum_smoother_.smooth(stereo_side_spectrum_, stereo_dyn_left_, stereo_dyn_length_);
            s = kfr::log10(kfr::max(s, 1e-24f));
            // update stereo dynamic dbs
            for (const auto& band : stereo_dyn_bands_) {
                dynamic_spectrums_[band].process(stereo_side_spectrum_);
                if constexpr (!bypass) {
                    dynamic_spectrums_[band].update(stereo_dyn_gain_);
                }
            }
            // convert stereo dynamic dbs to gains and apply
            if constexpr (!bypass) {
                g = kfr::exp10(g);
                auto c0 = kfr::make_univector(fft_data.ch0_cspectrum_.data() + stereo_dyn_left_,
                    stereo_dyn_length_);
                c0 = c0 * g;
                auto c1 = kfr::make_univector(fft_data.ch1_cspectrum_.data() + stereo_dyn_left_,
                    stereo_dyn_length_);
                c1 = c1 * g;
            }
        }
        // smooth channel0 side
        if (fft_data.ch0_side_required_) {
            spectrum_smoother_.smooth(fft_data.ch0_side_spectrum_, fft_data.ch0_side_left_, fft_data.ch0_side_length_);
            auto s = kfr::make_univector(fft_data.ch0_side_spectrum_.data() + fft_data.ch0_side_left_,
                                         fft_data.ch0_side_length_);
            s = kfr::log10(kfr::max(s, 1e-24f));
        }
        // smooth channel1 side
        if (fft_data.ch1_side_required_) {
            spectrum_smoother_.smooth(fft_data.ch1_side_spectrum_, fft_data.ch1_side_left_, fft_data.ch1_side_length_);
            auto s = kfr::make_univector(fft_data.ch1_side_spectrum_.data() + fft_data.ch1_side_left_,
                                         fft_data.ch1_side_length_);
            s = kfr::log10(kfr::max(s, 1e-24f));
        }
        if (fft_data.ch0_not_off_) {
            if (fft_data.ch0_dyn_not_off_) {
                // reset channel dynamic dbs
                auto g = kfr::make_univector(fft_data.ch0_dyn_gain_.data() + fft_data.ch0_dyn_left_,
                                                 fft_data.ch0_dyn_length_);
                if constexpr (!bypass) {
                    std::ranges::fill(g, 0.0f);
                }
                // update channel dynamic dbs
                for (const auto& band : fft_data.ch0_dyn_bands_) {
                    if (lrms_side_shuffle_[band]) {
                        dynamic_spectrums_[band].process(fft_data.ch1_side_spectrum_);
                    } else {
                        dynamic_spectrums_[band].process(fft_data.ch0_side_spectrum_);
                    }
                    if constexpr (!bypass) {
                        dynamic_spectrums_[band].update(stereo_dyn_gain_);
                    }
                }
                // convert channel dynamic dbs to gains and apply
                if constexpr (!bypass) {
                    g = kfr::exp10(g);
                    auto c0 = kfr::make_univector(fft_data.ch0_cspectrum_.data() + fft_data.ch0_dyn_left_,
                                                  fft_data.ch0_dyn_length_);
                    c0 = c0 * g;
                }
            }
            if constexpr (!bypass) {
                fft_data.ch0_cspectrum_ *= fft_data.ch0_static_gain_;
                backwardFFT(fft_data.ch0_output_fifo_, fft_data.pos_, fft_data.ch0_cspectrum_);
            } else {
                fft_data.ch0_output_fifo_ += fft_data.ch0_input_fifo_ * kBypassCorrection;
            }
        } else {
            fft_data.ch0_output_fifo_ += fft_data.ch0_input_fifo_ * kBypassCorrection;
        }
        if (fft_data.ch1_not_off_) {
            if (fft_data.ch1_dyn_not_off_) {
                // reset channel dynamic dbs
                auto g = kfr::make_univector(fft_data.ch1_dyn_gain_.data() + fft_data.ch1_dyn_left_,
                                                 fft_data.ch1_dyn_length_);
                if constexpr (!bypass) {
                    std::ranges::fill(g, 0.0f);
                }
                // update channel dynamic dbs
                for (const auto& band : fft_data.ch1_dyn_bands_) {
                    if (lrms_side_shuffle_[band]) {
                        dynamic_spectrums_[band].process(fft_data.ch0_side_spectrum_);
                    } else {
                        dynamic_spectrums_[band].process(fft_data.ch1_side_spectrum_);
                    }
                    if constexpr (!bypass) {
                        dynamic_spectrums_[band].update(stereo_dyn_gain_);
                    }
                }
                // convert channel dynamic dbs to gains and apply
                if constexpr (!bypass) {
                    g = kfr::exp10(g);
                    auto c0 = kfr::make_univector(fft_data.ch1_cspectrum_.data() + fft_data.ch1_dyn_left_,
                                                  fft_data.ch1_dyn_length_);
                    c0 = c0 * g;
                }
            }
            if constexpr (!bypass) {
                fft_data.ch1_cspectrum_ *= fft_data.ch1_static_gain_;
                backwardFFT(fft_data.ch1_output_fifo_, fft_data.pos_, fft_data.ch1_cspectrum_);
            } else {
                fft_data.ch1_output_fifo_ += fft_data.ch1_input_fifo_ * kBypassCorrection;
            }
        } else {
            fft_data.ch1_output_fifo_ += fft_data.ch1_input_fifo_ * kBypassCorrection;
        }
    }

    void Controller::forwardFFT(const kfr::univector<float>& fifo, const size_t fifo_pos,
                                kfr::univector<std::complex<float>>& cspectrum) {
        std::memcpy(fft_in_.data(), fifo.data() + fifo_pos, (fft_size_ - fifo_pos) * sizeof(float));
        std::memcpy(fft_in_.data() + fft_size_ - fifo_pos, fifo.data(), fifo_pos * sizeof(float));
        fft_in_ = fft_in_ * window1_;
        fft_plan_->execute(cspectrum, fft_in_, fft_temp_buffer_);
    }

    void Controller::backwardFFT(kfr::univector<float>& fifo, size_t fifo_pos,
        const kfr::univector<std::complex<float>>& cspectrum) {
        fft_plan_->execute(fft_in_, cspectrum, fft_temp_buffer_);
        fft_in_ = fft_in_ * window2_;
        if (fifo_pos > 0) {
            auto v0 = kfr::make_univector(fifo.data(), fifo_pos);
            const auto f0 = kfr::make_univector(fft_in_.data() + fft_size_ - fifo_pos, fifo_pos);
            v0 = v0 + f0;
        }
        auto v1 = kfr::make_univector(fifo.data() + fifo_pos, fft_size_ - fifo_pos);
        const auto f1 = kfr::make_univector(fft_in_.data(), fft_size_ - fifo_pos);
        v1 = v1 + f1;
    }
}
