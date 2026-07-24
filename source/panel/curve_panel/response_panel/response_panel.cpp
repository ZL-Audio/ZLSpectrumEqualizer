// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#include "response_panel.hpp"

namespace zlpanel {
    ResponsePanel::ResponsePanel(PluginProcessor& p,
                                 zlgui::UIBase& base,
                                 const multilingual::TooltipHelper& tooltip_helper) :
        Thread("response"),
        p_ref_(p), base_(base),
        gain_scale_(*p.parameters_.getRawParameterValue(zlp::PGainScale::kID)),
        single_panel_(p, base, message_not_off_indices_),
        sum_panel_(p, base),
        dragger_panel_(p, base, tooltip_helper),
        solo_panel_(p, base),
        eq_max_db_idx_ref_(*p.parameters_NA_.getRawParameterValue(zlstate::PEQMaxDB::kID)) {
        juce::ignoreUnused(base_, tooltip_helper);
        for (size_t band = 0; band < zlp::kBandNum; ++band) {
            const auto band_str = std::to_string(band);
            for (const auto& ID : kIDs) {
                const auto band_ID = ID + band_str;
                p_ref_.parameters_.addParameterListener(band_ID, this);
                parameterChanged(band_ID,
                                 p_ref_.parameters_.getRawParameterValue(band_ID)->load(std::memory_order::relaxed));
            }
        }
        p_ref_.parameters_.addParameterListener(zlp::PGainScale::kID, this);
        p_ref_.parameters_.addParameterListener(zlp::PSpecResolution::kID, this);

        xs_.resize(kNumPoints);
        ws_.resize(kNumPoints);
        for (size_t i = 0; i < zlp::kBandNum; ++i) {
            base_mags_[i].resize(kNumPoints);
            target_mags_[i].resize(kNumPoints);
            dynamic_mags_[i].resize(kNumPoints);
        }
        for (size_t i = 0; i < 5; ++i) {
            sum_mags_[i].resize(kNumPoints);
            on_lr_indices_[i].reserve(zlp::kBandNum);
            interpolated_deltas_[i].resize(kNumPoints);
        }
        addAndMakeVisible(single_panel_);
        addAndMakeVisible(sum_panel_);
        addAndMakeVisible(dragger_panel_);
        dragger_panel_.addChildComponent(solo_panel_);
        solo_panel_.setAlwaysOnTop(true);
        dragger_panel_.getFloatPopPanel().setAlwaysOnTop(true);
        dragger_panel_.getFloatPopPanel().toFront(true);
        dragger_panel_.getRightClickPanel().setAlwaysOnTop(true);
        dragger_panel_.getRightClickPanel().toFront(true);
        setInterceptsMouseClicks(false, true);
    }

    ResponsePanel::~ResponsePanel() {
        dragger_panel_.removeChildComponent(&solo_panel_);
        for (size_t band = 0; band < zlp::kBandNum; ++band) {
            const auto band_str = std::to_string(band);
            for (const auto& ID : kIDs) {
                p_ref_.parameters_.removeParameterListener(ID + band_str, this);
            }
        }
        p_ref_.parameters_.removeParameterListener(zlp::PGainScale::kID, this);
        p_ref_.parameters_.removeParameterListener(zlp::PSpecResolution::kID, this);
    }

    void ResponsePanel::paint(juce::Graphics& g) {
        const auto should_alpha = static_cast<float>(base_.getPanelProperty(zlgui::kCurveShouldTransparent)) > .5f;
        if (should_alpha) {
            g.beginTransparencyLayer(.5f);
        }
        single_panel_.paintDifferentStereo(g);
        sum_panel_.paintDifferentStereo(g);
        single_panel_.paintSameStereo(g);
        sum_panel_.paintSameStereo(g);
        if (should_alpha) {
            g.endTransparencyLayer();
        }
    }

    void ResponsePanel::resized() {
        const auto bound = getLocalBounds();
        const auto font_size = base_.getFontSize();
        const auto bottom_height = getBottomAreaHeight(font_size);
        single_panel_.setBounds(bound);
        sum_panel_.setBounds(bound);
        dragger_panel_.setBounds(bound);
        solo_panel_.setBounds(bound);
        side_y_ = static_cast<float>(bound.getHeight() - bottom_height) - font_size * kDraggerScale * .5f;
        width_.store(static_cast<float>(bound.getWidth()), std::memory_order::relaxed);
        height_.store(static_cast<float>(bound.getHeight()), std::memory_order::relaxed);
        font_size_.store(base_.getFontSize(), std::memory_order::relaxed);
        to_update_bound_.signal();
    }

    void ResponsePanel::repaintCallBack() {
        updateDraggerPositions();
        updateDrawingParas();
        dragger_panel_.repaintCallBack();
    }

    void ResponsePanel::updateDraggerPositions() {
        updateSoloPosition();
        if (!message_to_update_draggers_total_.check()) {
            return;
        }
        for (size_t band = 0; band < zlp::kBandNum; ++band) {
            if (message_to_update_draggers_[band].check()) {
                const auto filter_type = empty_[band].getFilterType();
                dragger_panel_.updateFilterType(band, filter_type);
                dragger_panel_.getDragger(band).updateButton(
                {points_[band][0].load(std::memory_order::relaxed),
                 points_[band][4].load(std::memory_order::relaxed)});
            }
        }
        updateFloatingPosition();
        updateTargetPosition();
        updateSidePosition();
    }

    void ResponsePanel::updateSoloPosition() {
        if (!solo_panel_.isVisible()) {
            return;
        }
        if (const auto band = base_.getSelectedBand(); band < zlp::kBandNum) {
            if (solo_panel_.isSoloSide()) {
                solo_panel_.updateX(side_points_[band][1].load(std::memory_order::relaxed),
                                    side_points_[band][2].load(std::memory_order::relaxed));
            } else {
                solo_panel_.updateX(points_[band][1].load(std::memory_order::relaxed),
                                    points_[band][2].load(std::memory_order::relaxed));
            }
        }
    }

    void ResponsePanel::updateTargetPosition() {
        if (const auto band = base_.getSelectedBand(); band < zlp::kBandNum) {
            dragger_panel_.getTargetDragger().updateButton(
            {points_[band][0].load(std::memory_order::relaxed),
             points_[band][5].load(std::memory_order::relaxed)});
        }
    }

    void ResponsePanel::updateSidePosition() {
    }

    void ResponsePanel::updateFloatingPosition() {
        if (const auto band = base_.getSelectedBand(); band < zlp::kBandNum) {
            dragger_panel_.getFloatPopPanel().updatePosition(
            {points_[band][0].load(std::memory_order::relaxed),
             points_[band][4].load(std::memory_order::relaxed)});
        }
    }

    void ResponsePanel::updateDrawingParas() {
        if (!message_to_update_panels_.check()) {
            return;
        }
        message_not_off_indices_.clear();
        const auto selected_band = base_.getSelectedBand();
        const auto selected_lr_mode = selected_band < zlp::kBandNum
            ? lr_modes_[selected_band].load(std::memory_order::relaxed)
            : 0;
        for (size_t band = 0; band < zlp::kBandNum; ++band) {
            const auto filter_status = filter_status_[band].load(std::memory_order::relaxed);
            if (filter_status != zlp::FilterStatus::kOff) {
                message_not_off_indices_.emplace_back(band);
                const auto dynamic_on = dynamic_ons_[band].load(std::memory_order::relaxed);
                const auto lr_mode = lr_modes_[band].load(std::memory_order::relaxed);
                const auto is_same_stereo = selected_band < zlp::kBandNum ? lr_mode == selected_lr_mode : true;
                single_panel_.updateDrawingParas(band, filter_status, dynamic_on, is_same_stereo);
                dragger_panel_.updateDrawingParas(band, filter_status, dynamic_on, is_same_stereo, lr_mode);
                dragger_panel_.getDragger(band).updateButton(
                {points_[band][0].load(std::memory_order::relaxed),
                 points_[band][4].load(std::memory_order::relaxed)});
            } else {
                single_panel_.updateDrawingParas(band, zlp::FilterStatus::kOff, false, false);
                dragger_panel_.updateDrawingParas(band, zlp::FilterStatus::kOff, false, false, 0);
                dragger_panel_.getDragger(band).updateButton({-1000.f, -1000.f});
            }
        }
        updateFloatingPosition();
        updateTargetPosition();
        updateSidePosition();
        for (int lr = 0; lr < 5; ++lr) {
            if (selected_band < zlp::kBandNum) {
                sum_panel_.updateDrawingParas(lr, lr == selected_lr_mode);
            } else {
                sum_panel_.updateDrawingParas(lr, true);
            }
        }
    }

    void ResponsePanel::repaintCallBackSlow() {
        dragger_panel_.repaintCallBackSlow();
    }

    void ResponsePanel::updateBand() {
        message_to_update_panels_.signal();
        dragger_panel_.updateBand();
        solo_panel_.updateBand();
        updateFloatingPosition();
        updateTargetPosition();
        updateSidePosition();
    }

    void ResponsePanel::updateSampleRate(const double sample_rate) {
        sample_rate_.store(sample_rate, std::memory_order::relaxed);
        dragger_panel_.updateSampleRate(sample_rate);
    }

    void ResponsePanel::run() {
        while (!threadShouldExit()) {
            const auto flag = wait(-1);
            juce::ignoreUnused(flag);
            if (threadShouldExit()) {
                break;
            }
            updateCurveParas();
            if (threadShouldExit()) {
                break;
            }
            if (!updateCurveMags()) {
                break;
            }
            for (size_t band = 0; band < zlp::kBandNum; ++band) {
                single_panel_.run(band, c_filter_status_[band],
                                  to_update_base_y_flags_[band],
                                  to_update_target_y_flags_[band],
                                  xs_, c_k_, c_b_,
                                  base_mags_[band], target_mags_[band],
                                  points_[band][0].load(std::memory_order::relaxed),
                                  points_[band][3].load(std::memory_order::relaxed),
                                  points_[band][4].load(std::memory_order::relaxed),
                                  ideal_[band].getParas().filter_type == zldsp::filter::kAllPass,
                                  ideal_[band].getParas().order == 1);
                if (threadShouldExit()) {
                    break;
                }
            }
            auto& tri_buffers = p_ref_.getController().getDynamicResponseTriBuffers();

            for (size_t lr = 0; lr < 5; ++lr) {
                bool channel_has_dynamic = false;
                for (const auto band : on_lr_indices_[lr]) {
                    if (c_dynamic_ons_[band]) {
                        channel_has_dynamic = true;
                        break;
                    }
                }
                bool valid_size = false;
                {
                    const std::lock_guard lock(p_ref_.getController().getTriBufferLock());
                    tri_buffers[lr].pull();
                    const auto& shared_buffer = tri_buffers[lr].getReader();
                    valid_size = (shared_buffer.delta.size() == ws_dsp_.size() && !ws_dsp_.empty());
                    if (valid_size) {
                        static constexpr float kLogToDB = 8.685889638065037f;
                        std::fill(delta_dsp_.data(), delta_dsp_.data() + shared_buffer.dyn_start, 0.f);
                        for (size_t i = shared_buffer.dyn_start; i < shared_buffer.dyn_end; ++i) {
                            delta_dsp_[i] = shared_buffer.delta[i] * kLogToDB;
                        }
                        std::fill(delta_dsp_.data() + shared_buffer.dyn_end, delta_dsp_.data() + delta_dsp_.size(), 0.f);
                    }
                }
                if (valid_size) {
                    makima_.prepare(ws_dsp_.data(), delta_dsp_.data(), ws_dsp_.size());
                    makima_.eval(ws_.data(), interpolated_deltas_[lr].data(), kNumPoints);
                }

                const bool should_update_sum = to_update_lr_flags_[lr] || (channel_has_dynamic);
                if (should_update_sum && channel_has_dynamic && !ws_dsp_.empty()) {
                    if (valid_size) {
                        sum_panel_.run(lr, true, is_lr_not_off_flags_[lr],
                                       on_lr_indices_[lr],
                                       xs_, c_k_, c_b_,
                                       dynamic_mags_,
                                       std::span<const float>(interpolated_deltas_[lr]));
                    } else {
                        sum_panel_.run(lr, should_update_sum, is_lr_not_off_flags_[lr],
                                       on_lr_indices_[lr],
                                       xs_, c_k_, c_b_,
                                       dynamic_mags_);
                    }
                } else {
                    sum_panel_.run(lr, should_update_sum, is_lr_not_off_flags_[lr],
                                   on_lr_indices_[lr],
                                   xs_, c_k_, c_b_,
                                   dynamic_mags_);
                }
                to_update_lr_flags_[lr] = false;
                if (threadShouldExit()) {
                    break;
                }
            }
        }
    }

    void ResponsePanel::parameterChanged(const juce::String& parameter_ID, const float value) {
        if (parameter_ID.startsWith(zlp::PGainScale::kID)) {
            for (size_t band = 0; band < zlp::kBandNum; ++band) {
                empty_[band].setGain(
                    std::clamp(
                        original_base_gains_[band].load(std::memory_order::relaxed) * value / 100.f, -30.f, 30.f));
                to_update_empty_flags_[band].signal();
                target_gains_[band].store(
                    std::clamp(
                        original_target_gains_[band].load(std::memory_order::relaxed) * value / 100.f, -30.f, 30.f),
                    std::memory_order::relaxed);
                to_update_target_gain_flags_[band].signal();
            }
            return;
        }
        const auto band = static_cast<size_t>(parameter_ID.getTrailingIntValue());
        if (parameter_ID.startsWith(zlp::PFilterStatus::kID)) {
            filter_status_[band].store(static_cast<zlp::FilterStatus>(std::round(value)), std::memory_order::relaxed);
            to_update_filter_status_.signal();
        } else if (parameter_ID.startsWith(zlp::PLRMode::kID)) {
            lr_modes_[band].store(static_cast<int>(std::round(value)), std::memory_order::relaxed);
            to_update_lr_modes_.signal();
        } else if (parameter_ID.startsWith(zlp::PFilterType::kID)) {
            empty_[band].setFilterType(static_cast<zldsp::filter::FilterType>(std::round(value)));
            to_update_empty_flags_[band].signal();
        } else if (parameter_ID.startsWith(zlp::POrder::kID)) {
            empty_[band].setOrder(zlp::POrder::kOrderArray[static_cast<size_t>(std::round(value))]);
            to_update_empty_flags_[band].signal();
        } else if (parameter_ID.startsWith(zlp::PFreq::kID)) {
            empty_[band].setFreq(value);
            to_update_empty_flags_[band].signal();
        } else if (parameter_ID.startsWith(zlp::PGain::kID)) {
            original_base_gains_[band].store(value, std::memory_order::relaxed);
            empty_[band].setGain(
                std::clamp(value * gain_scale_.load(std::memory_order::relaxed) / 100.f, -30.f, 30.f));
            to_update_empty_flags_[band].signal();
        } else if (parameter_ID.startsWith(zlp::PQ::kID)) {
            empty_[band].setQ(value);
            to_update_empty_flags_[band].signal();
        } else if (parameter_ID.startsWith(zlp::PDynamicON::kID)) {
            dynamic_ons_[band].store(value > .5f, std::memory_order::relaxed);
            to_update_dynamic_ons_.signal();
        } else if (parameter_ID.startsWith(zlp::PTargetGain::kID)) {
            original_target_gains_[band].store(value, std::memory_order::relaxed);
            target_gains_[band].store(
                std::clamp(value * gain_scale_.load(std::memory_order::relaxed) / 100.f, -30.f, 30.f),
                std::memory_order::relaxed);
            to_update_target_gain_flags_[band].signal();
        } else if (parameter_ID.startsWith(zlp::PSpecResolution::kID)) {
            to_update_fft_resolution_.signal();
        }
    }

    void ResponsePanel::updateCurveParas() {
        // update filter status
        if (to_update_filter_status_.check()) {
            for (size_t band = 0; band < zlp::kBandNum; ++band) {
                const auto new_filter_status = filter_status_[band].load(std::memory_order::relaxed);
                if (c_filter_status_[band] != new_filter_status) {
                    c_filter_status_[band] = new_filter_status;
                    to_update_base_y_flags_[band] = true;
                    to_update_lr_flags_[static_cast<size_t>(c_lr_modes_[band])] = true;
                }
            }
            to_update_lr_modes_.signal();
        }
        // update dynamic ons
        if (to_update_dynamic_ons_.check()) {
            for (size_t band = 0; band < zlp::kBandNum; ++band) {
                const auto dynamic_on = dynamic_ons_[band].load(std::memory_order::relaxed);
                if (c_dynamic_ons_[band] != dynamic_on) {
                    c_dynamic_ons_[band] = dynamic_ons_[band].load(std::memory_order::relaxed);
                    to_update_lr_flags_[static_cast<size_t>(c_lr_modes_[band])] = true;
                    dynamic_mags_[band] = base_mags_[band];
                }
            }
            message_to_update_panels_.signal();
        }
        // update lr modes for summing
        if (to_update_lr_modes_.check()) {
            for (auto& indices : on_lr_indices_) {
                indices.clear();
            }
            std::fill(is_lr_not_off_flags_.begin(), is_lr_not_off_flags_.end(), false);
            for (size_t band = 0; band < zlp::kBandNum; ++band) {
                const auto lr_mode = lr_modes_[band].load(std::memory_order::relaxed);
                if (lr_mode != c_lr_modes_[band]) {
                    to_update_lr_flags_[static_cast<size_t>(c_lr_modes_[band])] = true;
                    to_update_lr_flags_[static_cast<size_t>(lr_mode)] = true;
                    c_lr_modes_[band] = lr_mode;
                }
                if (c_filter_status_[band] != zlp::FilterStatus::kOff) {
                    if (c_filter_status_[band] == zlp::FilterStatus::kOn) {
                        on_lr_indices_[static_cast<size_t>(lr_mode)].emplace_back(band);
                    }
                    is_lr_not_off_flags_[static_cast<size_t>(lr_mode)] = true;
                }
            }
            message_to_update_panels_.signal();
        }
        // update sample rate
        const bool sr_changed = std::abs(sample_rate_.load(std::memory_order::relaxed) - c_sample_rate_) > 1.0;
        const bool fft_res_changed = to_update_fft_resolution_.check();
        if (sr_changed) {
            const auto sample_rate = sample_rate_.load(std::memory_order::relaxed);
            c_sample_rate_ = sample_rate;
            c_slider_max_ = freq_helper::getSliderMax(sample_rate);
            if (sample_rate < 40000.0) {
                return;
            }
            fft_max_ = freq_helper::getFFTMax(sample_rate);
            for (auto& f : ideal_) {
                f.prepare(sample_rate);
            }
            const auto max_log_value = std::log(fft_max_ * 0.1) / static_cast<double>(kFFTSizeOverWidth);
            const auto interval_log_value = max_log_value / static_cast<double>(kNumPoints - 1);
            const auto freq_scale = 20.0 * std::numbers::pi / sample_rate;
            for (size_t i = 0; i < kNumPoints; ++i) {
                ws_[i] = static_cast<float>(std::exp(interval_log_value * static_cast<double>(i)) * freq_scale);
            }
            std::fill(to_update_base_y_flags_.begin(), to_update_base_y_flags_.end(), true);
        }
        if ((sr_changed || fft_res_changed || ws_dsp_.empty()) && c_sample_rate_ >= 40000.0) {
            const auto fft_resolution = p_ref_.parameters_.getRawParameterValue(zlp::PSpecResolution::kID)->load(std::memory_order::relaxed);
            size_t fft_low_order;
            if (c_sample_rate_ < 50000.0) { fft_low_order = 12; }
            else if (c_sample_rate_ < 100000.0) { fft_low_order = 13; }
            else if (c_sample_rate_ < 200000.0) { fft_low_order = 14; }
            else if (c_sample_rate_ < 400000.0) { fft_low_order = 15; }
            else { fft_low_order = 16; }

            const size_t order = fft_low_order - 1 + static_cast<size_t>(std::round(fft_resolution));
            const size_t fft_size = 1ULL << order;
            const size_t num_bin_effective = fft_size / 2;

            ws_dsp_.resize(num_bin_effective + 1);
            zldsp::filter::IdealBase<float>::calculateWs(ws_dsp_);
            ws_dsp_.resize(num_bin_effective);
            delta_dsp_.resize(num_bin_effective);
        }
        // update width & xs
        if (to_update_bound_.check()) {
            c_width_ = width_.load(std::memory_order::relaxed);
            c_height_ = height_.load(std::memory_order::relaxed);
            c_font_size_ = font_size_.load(std::memory_order::relaxed);
            if (c_width_ < 1.f || c_height_ < 1.f) {
                return;
            }
            const auto interval_x_value = c_width_ / static_cast<float>(kNumPoints - 1);
            for (size_t i = 0; i < kNumPoints; ++i) {
                xs_[i] = static_cast<float>(i) * interval_x_value;
            }
            c_eq_max_db_idx_ = -1.f;
        }
        // update maximum db
        if (const auto eq_max_db_idx = eq_max_db_idx_ref_.load(std::memory_order::relaxed);
            std::abs(eq_max_db_idx - c_eq_max_db_idx_) > .1f) {
            c_eq_max_db_idx_ = eq_max_db_idx;
            const auto z = base_.getCurveDBScale(static_cast<size_t>(std::round(eq_max_db_idx)));
            const auto h = c_height_ - static_cast<float>(getBottomAreaHeight(c_font_size_));
            const auto padding = c_font_size_ * kDraggerScale;
            const auto h1 = h * .5f;
            const auto h2 = h - padding;
            c_k_ = (h1 - h2) / z;
            c_b_ = h1;
            std::fill(to_update_base_y_flags_.begin(), to_update_base_y_flags_.end(), true);
        }
        // update db update flags
        for (size_t band = 0; band < zlp::kBandNum; ++band) {
            to_update_base_y_flags_[band] = to_update_base_y_flags_[band]
                || to_update_empty_flags_[band].check();
            to_update_target_y_flags_[band] = to_update_base_y_flags_[band]
                || to_update_target_gain_flags_[band].check();
            const auto lr = c_lr_modes_[band];
            to_update_lr_flags_[static_cast<size_t>(lr)] = to_update_lr_flags_[static_cast<size_t>(lr)]
                || to_update_base_y_flags_[band] || c_dynamic_ons_[band];
        }
    }

    bool ResponsePanel::updateCurveMags() {
        for (size_t band = 0; band < zlp::kBandNum; ++band) {
            if (c_filter_status_[band] != zlp::FilterStatus::kOff) {
                if (to_update_base_y_flags_[band]) {
                    auto para = empty_[band].getParas();
                    para.freq = std::min(para.freq, c_slider_max_);
                    ideal_[band].forceUpdate(para);
                    ideal_[band].updateMagnitudeSquare(ws_, base_mags_[band]);
                    zldsp::vector::sqr_mag_to_db(base_mags_[band].data(), base_mags_[band].size());
                    dynamic_mags_[band] = base_mags_[band];
                    const auto center_w = para.freq * (2.0 * std::numbers::pi / c_sample_rate_);
                    const float center_square_magnitude = zldsp::chore::squareGainToDecibels(
                        ideal_[band].getCenterMagnitudeSquare(static_cast<float>(center_w)));
                    const auto [left_x, center_x, right_x] = getLeftCenterRightX(para);

                    points_[band][0].store(static_cast<float>(center_x), std::memory_order::relaxed);
                    points_[band][1].store(static_cast<float>(left_x), std::memory_order::relaxed);
                    points_[band][2].store(static_cast<float>(right_x), std::memory_order::relaxed);
                    points_[band][3].store(c_k_ * center_square_magnitude + c_b_, std::memory_order::relaxed);
                    para.gain = original_base_gains_[band].load(std::memory_order::relaxed);
                    points_[band][4].store(c_k_ * getButtonMag(para) + c_b_, std::memory_order::relaxed);

                    if (threadShouldExit()) {
                        return false;
                    }
                    message_to_update_draggers_[band].signal();
                    message_to_update_draggers_total_.signal();
                }
                if (to_update_target_y_flags_[band]) {
                    const auto target_gain = target_gains_[band].load(std::memory_order::relaxed);
                    ideal_[band].setGain(target_gain);
                    ideal_[band].updateCoeffs();
                    ideal_[band].updateMagnitudeSquare(ws_, target_mags_[band]);
                    zldsp::vector::sqr_mag_to_db(target_mags_[band].data(), target_mags_[band].size());
                    auto para = ideal_[band].getParas();
                    para.gain = original_target_gains_[band].load(std::memory_order::relaxed);
                    points_[band][5].store(c_k_ * getButtonMag(para) + c_b_, std::memory_order::relaxed);

                    if (threadShouldExit()) {
                        return false;
                    }
                    message_to_update_target_dragger_.signal();
                    message_to_update_draggers_total_.signal();
                }
                // Dynamic EQ is disabled in zlseq
            } else {
                points_[band][4].store(-10000.f, std::memory_order::relaxed);
            }
        }
        return true;
    }

    float ResponsePanel::getButtonMag(const zldsp::filter::FilterParameters& para) {
        if (para.filter_type == zldsp::filter::kPeak) {
            return static_cast<float>(para.gain);
        } else if (para.filter_type == zldsp::filter::kLowShelf
            || para.filter_type == zldsp::filter::kHighShelf
            || para.filter_type == zldsp::filter::kTiltShelf
            || para.filter_type == zldsp::filter::kFlatTilt) {
            return static_cast<float>(0.5 * para.gain);
        } else {
            return 0.f;
        }
    }

    std::tuple<float, float, float> ResponsePanel::getLeftCenterRightX(zldsp::filter::FilterParameters para) const {
        const auto freq_to_x_scale = 1.0 / std::log(
            fft_max_ * 0.1) * static_cast<double>(c_width_) * static_cast<double>(
            kFFTSizeOverWidth);
        const auto center_x = std::log(para.freq / 10.0) * freq_to_x_scale;

        switch (para.filter_type) {
        case zldsp::filter::kPeak:
        case zldsp::filter::kBandPass:
        case zldsp::filter::kNotch:
        default: {
            const auto bandwidth = para.freq / para.q;
            const auto left_f = 0.5 * bandwidth * (std::sqrt(4.0 * para.q * para.q + 1.0) - 1.0);
            const auto left_x = std::log(left_f / 10.0) * freq_to_x_scale;
            const auto right_f = left_f + bandwidth;
            const auto right_x = std::log(right_f / 10.0) * freq_to_x_scale;
            return std::make_tuple(static_cast<float>(left_x),
                                   static_cast<float>(center_x),
                                   static_cast<float>(right_x));
        }
        case zldsp::filter::kAllPass: {
            if (para.order == 1) {
                para.q = std::sqrt(2) * 0.5;
            }
            para.order = 2;
            const auto bandwidth = para.freq / para.q;
            const auto left_f = 0.5 * bandwidth * (std::sqrt(4.0 * para.q * para.q + 1.0) - 1.0);
            const auto left_x = std::log(left_f / 10.0) * freq_to_x_scale;
            const auto right_f = left_f + bandwidth;
            const auto right_x = std::log(right_f / 10.0) * freq_to_x_scale;
            return std::make_tuple(static_cast<float>(left_x),
                                   static_cast<float>(center_x),
                                   static_cast<float>(right_x));
        }
        case zldsp::filter::kTiltShelf:
        case zldsp::filter::kFlatTilt: {
            const auto fixed_q = std::sqrt(2.0) * 0.03125;
            const auto bandwidth = para.freq / fixed_q;
            const auto left_f = 0.5 * bandwidth * (std::sqrt(4.0 * fixed_q * fixed_q + 1.0) - 1.0);
            const auto left_x = std::log(left_f / 10.0) * freq_to_x_scale;
            const auto right_f = left_f + bandwidth;
            const auto right_x = std::log(right_f / 10.0) * freq_to_x_scale;
            return std::make_tuple(static_cast<float>(left_x),
                                   static_cast<float>(center_x),
                                   static_cast<float>(right_x));
        }
        case zldsp::filter::kLowShelf:
        case zldsp::filter::kHighPass: {
            return std::make_tuple(0.f, static_cast<float>(center_x), static_cast<float>(center_x));
        }
        case zldsp::filter::kHighShelf:
        case zldsp::filter::kLowPass: {
            return std::make_tuple(static_cast<float>(center_x), static_cast<float>(center_x), c_width_);
        }
        }
    }
}
