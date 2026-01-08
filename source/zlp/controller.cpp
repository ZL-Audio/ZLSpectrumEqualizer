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

    template <bool bypass>
    void Controller::processBuffer(const std::array<float*, 2> main_buffer,
                                   const std::array<float*, 2> side_buffer,
                                   const size_t num_samples) {
        if (ext_side_) {
            processBufferInternal<bypass, true>(main_buffer, side_buffer, num_samples);
        } else {
            processBufferInternal<bypass, false>(main_buffer, side_buffer, num_samples);
        }
    }

    void Controller::handleAsyncUpdate() {
        p_ref_.setLatencySamples(latency_.load(std::memory_order_relaxed));
    }

    template <bool bypass, bool ext_side>
    void Controller::processBufferInternal(const std::array<float*, 2> main_buffer,
                                           const std::array<float*, 2> side_buffer,
                                           size_t num_samples) {
        size_t buffer_pos = 0;
        while (num_samples > 0) {
            const size_t num_to_process = std::min(num_samples, hop_size_ - fifo_count_);
            const size_t copy_size = num_to_process * sizeof(float);

            for (size_t chan = 0; chan < 2; ++chan) {
                std::memcpy(input_fifo_[chan].data() + fifo_pos_, main_buffer[chan] + buffer_pos, copy_size);
                std::memcpy(main_buffer[chan] + buffer_pos, output_fifo_[chan].data() + fifo_pos_, copy_size);
                std::memset(output_fifo_[chan].data() + fifo_pos_, 0, copy_size);
            }
            if constexpr (ext_side) {
                for (size_t chan = 0; chan < 2; ++chan) {
                    std::memcpy(ext_fifo_[chan].data() + fifo_pos_, side_buffer[chan] + buffer_pos, copy_size);
                }
            }
            buffer_pos += num_to_process;
            num_samples -= num_to_process;

            fifo_pos_ += num_to_process;
            fifo_count_ += num_to_process;

            if (fifo_count_ >= hop_size_) {
                fifo_count_ = 0;
                if (fifo_pos_ >= fft_size_) {
                    fifo_pos_ = 0;
                }
                processFrame<bypass, ext_side>();
            }
        }
    }

    template <bool bypass, bool ext_side>
    void Controller::processFrame() {
        // main FFT forward
        for (size_t chan = 0; chan < 2; ++chan) {
            std::memcpy(fft_in_.data(), input_fifo_[chan].data() + fifo_pos_,
                        (fft_size_ - fifo_pos_) * sizeof(float));
            std::memcpy(fft_in_.data() + fft_size_ - fifo_pos_, input_fifo_[chan].data(),
                        fifo_pos_ * sizeof(float));
            fft_in_ = fft_in_ * window1_;
            fft_plan_->execute(cspectrum_[chan], fft_in_, fft_temp_buffer_);
        }
        // if external side and dynamic is not off, side FFT forward
        if constexpr (ext_side) {
            if (dyn_not_off_) {
                for (size_t chan = 0; chan < 2; ++chan) {
                    std::memcpy(fft_in_.data(), ext_fifo_[chan].data() + fifo_pos_,
                                (fft_size_ - fifo_pos_) * sizeof(float));
                    std::memcpy(fft_in_.data() + fft_size_ - fifo_pos_, ext_fifo_[chan].data(),
                                fifo_pos_ * sizeof(float));
                    fft_in_ = fft_in_ * window1_;
                    fft_plan_->execute(side_cspectrum_[chan], fft_in_, fft_temp_buffer_);
                }
            }
        }
        // process spectrum
        processSpectrumStereo<bypass, ext_side>();
        // main backward FFT & overlap-add
        for (size_t chan = 0; chan < 2; ++chan) {
            fft_plan_->execute(fft_in_, cspectrum_[chan], fft_temp_buffer_);
            fft_in_ = fft_in_ * window2_;
            if (fifo_pos_ > 0) {
                auto v0 = kfr::make_univector(output_fifo_[chan].data(), fifo_pos_);
                const auto f0 = kfr::make_univector(fft_in_.data() + fft_size_ - fifo_pos_, fifo_pos_);
                v0 = v0 + f0;
            }
            auto v1 = kfr::make_univector(output_fifo_[chan].data() + fifo_pos_, fft_size_ - fifo_pos_);
            const auto f1 = kfr::make_univector(fft_in_.data(), fft_size_ - fifo_pos_);
            v1 = v1 + f1;
        }
    }

    template <bool bypass, bool ext_side>
    void Controller::processSpectrumStereo() {
        if (dyn_not_off_) {
            // calculate side spectrum, smooth it and convert to log domain
            calculateSideSpectrumStereo(ext_side ? side_cspectrum_[0] : cspectrum_[0],
                                        ext_side ? side_cspectrum_[1] : cspectrum_[1]);
            // process all dynamic not off bands
            processSideSpectrum<bypass>(stereo_data_, side_spectrum_);
            // update dynamic gain & final gain
            if constexpr (!bypass) {
                auto gd = stereo_data_.dyn_gain_.slice(stereo_data_.dyn_left_, stereo_data_.dyn_length_);
                auto g0 = stereo_data_.static_gain_.slice(stereo_data_.dyn_left_, stereo_data_.dyn_length_);
                auto g = stereo_data_.final_gain_.slice(stereo_data_.dyn_left_, stereo_data_.dyn_length_);
                g = g0 * kfr::exp10(gd);
                // apply final gain
                for (size_t chan = 0; chan < 2; ++chan) {
                    cspectrum_[chan] = cspectrum_[chan] * stereo_data_.final_gain_[chan];
                }
            }
        } else {
            if constexpr (!bypass) {
                // apply static gain
                for (size_t chan = 0; chan < 2; ++chan) {
                    cspectrum_[chan] = cspectrum_[chan] * stereo_data_.static_gain_;
                }
            }
        }
    }

    void Controller::calculateSideSpectrumStereo(kfr::univector<std::complex<float>>& side0_cspectrum,
                                                 kfr::univector<std::complex<float>>& side1_cspectrum) {
        // calculate side spectrum and smooth it
        auto s = side_spectrum_.slice(stereo_data_.side_left_, stereo_data_.side_length_);
        auto c0 = side0_cspectrum.slice(stereo_data_.side_left_, stereo_data_.side_length_);
        auto c1 = side1_cspectrum.slice(stereo_data_.side_left_, stereo_data_.side_length_);
        s = kfr::cabssqr(c0) + kfr::cabssqr(c1);
        spectrum_smoother_.smooth(side_spectrum_, stereo_data_.side_left_, stereo_data_.side_length_);
        // convert to log domain
        s = kfr::log10(kfr::max(s, 1e-24f));
    }

    template <bool bypass, bool ext_side>
    void Controller::processSpectrumLR() {
        if (stereo_data_.dyn_not_off || l_data_.dyn_not_off || r_data_.dyn_not_off) {
            if (stereo_data_.dyn_not_off) {
                // calculate side spectrum, smooth it and convert to log domain
                calculateSideSpectrumStereo(ext_side ? side_cspectrum_[0] : cspectrum_[0],
                                            ext_side ? side_cspectrum_[1] : cspectrum_[1]);
                // process all dynamic not off bands
                processSideSpectrum<bypass>(stereo_data_, side_spectrum_);
            }
            if (l_data_.dyn_not_off) {
                // calculate side spectrum, smooth it and convert to log domain
                calculateSideSpectrumChannel(l_data_, side_spectrum_, ext_side ? side_cspectrum_[0] : cspectrum_[0]);
                // process all dynamic not off bands
                processSideSpectrum<bypass>(l_data_, side_spectrum_);
            }
            if (r_data_.dyn_not_off) {
                // calculate side spectrum, smooth it and convert to log domain
                calculateSideSpectrumChannel(r_data_, side_spectrum_, ext_side ? side_cspectrum_[1] : cspectrum_[1]);
                // process all dynamic not off bands
                processSideSpectrum<bypass>(r_data_, side_spectrum_);
            }
            if constexpr (!bypass) {
                if (stereo_data_.dyn_not_off || l_data_.dyn_not_off) {
                    calculateFinalGainChannel(l_data_);
                    cspectrum_[0] = cspectrum_[0] * l_data_.final_gain_;
                } else {
                    cspectrum_[0] = cspectrum_[0] * l_data_.static_gain_;
                }
                if (stereo_data_.dyn_not_off || r_data_.dyn_not_off) {
                    calculateFinalGainChannel(r_data_);
                    cspectrum_[1] = cspectrum_[1] * r_data_.final_gain_;
                } else {
                    cspectrum_[1] = cspectrum_[1] * r_data_.static_gain_;
                }
            }
        } else {
            if constexpr (!bypass) {
                // apply static gain
                cspectrum_[0] = cspectrum_[0] * l_data_.static_gain_;
                cspectrum_[1] = cspectrum_[1] * r_data_.static_gain_;
            }
        }
    }

    template <bool bypass, bool ext_side>
    void Controller::processSpectrumMS() {
        if (stereo_data_.dyn_not_off || m_data_.dyn_not_off || s_data_.dyn_not_off) {
            if (stereo_data_.dyn_not_off) {
                // calculate side spectrum, smooth it and convert to log domain
                calculateSideSpectrumStereo(ext_side ? side_cspectrum_[0] : cspectrum_[0],
                                            ext_side ? side_cspectrum_[1] : cspectrum_[1]);
                // process all dynamic not off bands
                processSideSpectrum<bypass>(stereo_data_, side_spectrum_);
            }
            if (m_data_.dyn_not_off) {
                // calculate side spectrum, smooth it and convert to log domain
                calculateSideSpectrumChannel(m_data_, side_spectrum_, ext_side ? side_cspectrum_[0] : cspectrum_[0]);
                // process all dynamic not off bands
                processSideSpectrum<bypass>(m_data_, side_spectrum_);
            }
            if (s_data_.dyn_not_off) {
                // calculate side spectrum, smooth it and convert to log domain
                calculateSideSpectrumChannel(s_data_, side_spectrum_, ext_side ? side_cspectrum_[1] : cspectrum_[1]);
                // process all dynamic not off bands
                processSideSpectrum<bypass>(s_data_, side_spectrum_);
            }
            if constexpr (!bypass) {
                if (stereo_data_.dyn_not_off || m_data_.dyn_not_off) {
                    calculateFinalGainChannel(m_data_);
                    cspectrum_[0] = cspectrum_[0] * m_data_.final_gain_;
                } else {
                    cspectrum_[0] = cspectrum_[0] * m_data_.static_gain_;
                }
                if (stereo_data_.dyn_not_off || s_data_.dyn_not_off) {
                    calculateFinalGainChannel(s_data_);
                    cspectrum_[1] = cspectrum_[1] * s_data_.final_gain_;
                } else {
                    cspectrum_[1] = cspectrum_[1] * s_data_.static_gain_;
                }
            }
        } else {
            if constexpr (!bypass) {
                // apply static gain
                cspectrum_[0] = cspectrum_[0] * m_data_.static_gain_;
                cspectrum_[1] = cspectrum_[1] * s_data_.static_gain_;
            }
        }
    }

    template <bool bypass, bool ext_side>
    void Controller::processSpectrumLRMS() {
        processSpectrumLR<bypass, ext_side>();
        // split main MS
        for (size_t i = 0; i < cspectrum_[0].size(); ++i) {
            auto c0 = kSqrt2Over2 * cspectrum_[0][i];
            auto c1 = kSqrt2Over2 * cspectrum_[1][i];
            cspectrum_[0][i] = c0 + c1;
            cspectrum_[1][i] = c0 - c1;
        }
        if (m_data_.dyn_not_off || s_data_.dyn_not_off) {
            // split side MS
            if constexpr (ext_side) {
                for (size_t i = 0; i < side_cspectrum_[0].size(); ++i) {
                    auto c0 = kSqrt2Over2 * side_cspectrum_[0][i];
                    auto c1 = kSqrt2Over2 * side_cspectrum_[1][i];
                    side_cspectrum_[0][i] = c0 + c1;
                    side_cspectrum_[1][i] = c0 - c1;
                }
            }
            if (m_data_.dyn_not_off) {
                // calculate side spectrum, smooth it and convert to log domain
                calculateSideSpectrumChannel(m_data_, side_spectrum_, ext_side ? side_cspectrum_[0] : cspectrum_[0]);
                // process all dynamic not off bands
                processSideSpectrum<bypass>(m_data_, side_spectrum_);
            }
            if (s_data_.dyn_not_off) {
                // calculate side spectrum, smooth it and convert to log domain
                calculateSideSpectrumChannel(s_data_, side_spectrum_, ext_side ? side_cspectrum_[1] : cspectrum_[1]);
                // process all dynamic not off bands
                processSideSpectrum<bypass>(s_data_, side_spectrum_);
            }
            if constexpr (!bypass) {
                if (m_data_.dyn_not_off) {
                    calculateFinalGainChannel(m_data_);
                    cspectrum_[0] = cspectrum_[0] * m_data_.final_gain_;
                } else {
                    cspectrum_[0] = cspectrum_[0] * m_data_.static_gain_;
                }
                if (s_data_.dyn_not_off) {
                    calculateFinalGainChannel(s_data_);
                    cspectrum_[1] = cspectrum_[1] * s_data_.final_gain_;
                } else {
                    cspectrum_[1] = cspectrum_[1] * s_data_.static_gain_;
                }
            }
        } else {
            if constexpr (!bypass) {
                // apply static gain
                cspectrum_[0] = cspectrum_[0] * m_data_.static_gain_;
                cspectrum_[1] = cspectrum_[1] * s_data_.static_gain_;
            }
        }
    }

    void Controller::calculateSideSpectrumChannel(const ChannelFFTData& fft_data,
                                                  kfr::univector<float>& side_spectrum,
                                                  kfr::univector<std::complex<float>>& side_cspectrum) {
        // calculate side spectrum and smooth it
        auto s = side_spectrum.slice(fft_data.side_left_, fft_data.side_length_);
        auto c = side_cspectrum.slice(fft_data.side_left_, fft_data.side_length_);
        s = kfr::cabssqr(c);
        spectrum_smoother_.smooth(side_spectrum, fft_data.side_left_, fft_data.side_length_);
        // convert to log domain
        s = kfr::log10(kfr::max(s, 1e-24f));
    }

    template <bool bypass>
    void Controller::processSideSpectrum(ChannelFFTData& fft_data, kfr::univector<float>& side_spectrum) {
        std::memset(fft_data.dyn_gain_.data() + fft_data.dyn_left_, 0, fft_data.dyn_length_ * sizeof(float));
        for (const auto& band : fft_data.dyn_not_off_bands_) {
            if constexpr (bypass) {
                dynamic_spectrums_[band].process<true>(side_spectrum, fft_data.dyn_gain_);
            } else {
                if (dynamic_bypass_[band]) {
                    dynamic_spectrums_[band].process<true>(side_spectrum, fft_data.dyn_gain_);
                } else {
                    dynamic_spectrums_[band].process<false>(side_spectrum, fft_data.dyn_gain_);
                }
            }
        }
    }

    void Controller::calculateFinalGainChannel(ChannelFFTData& fft_data) {
        auto gd = fft_data.dyn_gain_.slice(fft_data.dyn_left_, fft_data.dyn_length_);
        auto g0 = fft_data.static_gain_.slice(fft_data.dyn_left_, fft_data.dyn_length_);
        auto g = fft_data.final_gain_.slice(fft_data.dyn_left_, fft_data.dyn_length_);
        auto gds = stereo_data_.dyn_gain_.slice(fft_data.dyn_left_, fft_data.dyn_length_);
        if (stereo_data_.dyn_not_off && fft_data.dyn_not_off) {
            g = g0 * kfr::exp10(gd + gds);
        } else if (stereo_data_.dyn_not_off) {
            g = g0 * kfr::exp10(gds);
        } else {
            g = g0 * kfr::exp10(gd);
        }
    }
}
